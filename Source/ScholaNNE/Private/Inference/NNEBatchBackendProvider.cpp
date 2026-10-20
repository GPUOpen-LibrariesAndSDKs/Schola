// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "Inference/NNEBatchBackendProvider.h"
#include "NNE.h"
#include "NNERuntimeCPU.h"
#include "LogScholaNNE.h"
#include "Misc/ConfigCacheIni.h"
#include <atomic>
#include "Misc/ScopeExit.h"
#include "Spaces/DictSpace.h"

namespace
{
using namespace UE::NNE;

struct FBackendCounters
{
    std::atomic<int64> ReshapeCalls{0}, PaddingRows{0}, StagingGrowthEvents{0}, ReshapeNs{0}, ServiceNs{0};
};
class FNNEBatchContext : public IBatchInferenceContext
{
public:
    FNNEBatchContext(TSharedPtr<IModelInstanceCPU> InInstance, const FNPCRowLayout& InLayout, int32 InFixed, bool bInPad, TSharedRef<FBackendCounters> InCounters)
        : Instance(MoveTemp(InInstance)), Layout(InLayout), FixedBatch(InFixed), bPad(bInPad), Counters(InCounters)
    {
        InputStorage.SetNum(Layout.Inputs.Num()); OutputStorage.SetNum(Layout.Outputs.Num());
        Inputs.SetNum(Layout.Inputs.Num()); Outputs.SetNum(Layout.Outputs.Num());
        Shapes.SetNum(Layout.Inputs.Num());
    }
    virtual bool RunBatch(const float* Observations, int32 Rows, float* Actions) override
    {
        if (Busy.test_and_set()) return false;
        ON_SCOPE_EXIT { Busy.clear(); };
        const double ServiceStart = FPlatformTime::Seconds();
        const int32 Batch = FixedBatch > 0 ? FixedBatch : bPad ? int32(FMath::RoundUpToPowerOfTwo(uint32(Rows))) : Rows;
        if (Rows <= 0 || Rows > Batch) return false;
        if (LastBatch != Batch)
        {
            const double ShapeStart = FPlatformTime::Seconds();
            const auto Descs = Instance->GetInputTensorDescs();
            for (int32 I = 0; I < Descs.Num(); ++I)
            {
                TArray<uint32, TInlineAllocator<8>> Dims;
                for (int32 D : Descs[I].GetShape().GetData()) Dims.Add(uint32(D));
                Dims[0] = Batch;
                Shapes[I] = FTensorShape::Make(Dims);
            }
            if (Instance->SetInputTensorShapes(Shapes) != EResultStatus::Ok) return false;
            Counters->ReshapeNs.fetch_add(int64((FPlatformTime::Seconds()-ShapeStart)*1e9));
            Counters->ReshapeCalls.fetch_add(1);
            LastBatch = Batch;
        }
        Stage(Layout.Inputs, InputStorage, Inputs, Batch);
        Stage(Layout.Outputs, OutputStorage, Outputs, Batch);
        for (int32 T = 0; T < Layout.Inputs.Num(); ++T)
        {
            const auto& E = Layout.Inputs[T];
            auto* Dest = InputStorage[T].GetData();
            FMemory::Memzero(Dest, Inputs[T].SizeInBytes);
            for (int32 R = 0; R < Rows; ++R)
                for (int32 D = 0; D < E.Width; ++D)
                {
                    const float V = Observations[R * Layout.ObsSize + E.Offset + D];
                    const int32 Index = R * E.Width + D;
                    if (!FMath::IsFinite(V)) return false;
                    switch (E.DType)
                    {
                    case ENPCRowDType::Float: reinterpret_cast<float*>(Dest)[Index] = V; break;
                    case ENPCRowDType::Int64:
                        if (FMath::Abs(V) > 16777216.f || FMath::FloorToFloat(V) != V) return false;
                        reinterpret_cast<int64*>(Dest)[Index] = int64(V); break;
                    case ENPCRowDType::Bool:
                        if (V != 0.f && V != 1.f) return false;
                        reinterpret_cast<uint8*>(Dest)[Index] = V != 0.f; break;
                    }
                }
        }
        if (Instance->RunSync(Inputs, Outputs) != EResultStatus::Ok) return false;
        // ORT resolves symbolic outputs during its first RunSync after a reshape.
        const auto OutputShapes = Instance->GetOutputTensorShapes();
        if (OutputShapes.Num() != Layout.Outputs.Num()) return false;
        for (int32 I = 0; I < OutputShapes.Num(); ++I)
            if (OutputShapes[I].GetData().IsEmpty() || OutputShapes[I].GetData()[0] != uint32(Batch) ||
                OutputShapes[I].Volume() != uint64(Batch) * Layout.Outputs[I].Width) return false;
        for (int32 T = 0; T < Layout.Outputs.Num(); ++T)
        {
            const auto& E = Layout.Outputs[T];
            const auto* Source = OutputStorage[T].GetData();
            for (int32 R = 0; R < Rows; ++R)
                for (int32 D = 0; D < E.Width; ++D)
                {
                    const int32 Index = R * E.Width + D;
                    float V = 0;
                    switch (E.DType)
                    {
                    case ENPCRowDType::Float: V = reinterpret_cast<const float*>(Source)[Index]; break;
                    case ENPCRowDType::Int64:
                        { const int64 Integer = reinterpret_cast<const int64*>(Source)[Index];
                          if (Integer < -16777216 || Integer > 16777216) return false;
                          V = float(Integer); break; }
                    case ENPCRowDType::Bool: V = reinterpret_cast<const uint8*>(Source)[Index] ? 1.f : 0.f; break;
                    }
                    if (!FMath::IsFinite(V)) return false;
                    Actions[R * Layout.ActSize + E.Offset + D] = V;
                }
        }
        Counters->PaddingRows.fetch_add(Batch-Rows);
        Counters->ServiceNs.fetch_add(int64((FPlatformTime::Seconds()-ServiceStart)*1e9));
        return true;
    }
private:
    // uint64 backing keeps typed accesses aligned, including int64.
    void Stage(const TArray<FNPCRowElement>& Elements, TArray<TArray<uint64>>& Storage,
                      TArray<FTensorBindingCPU>& Bindings, int32 Batch)
    {
        for (int32 I = 0; I < Elements.Num(); ++I)
        {
            const auto& E = Elements[I];
            const int32 Size = E.DType == ENPCRowDType::Int64 ? 8 : E.DType == ENPCRowDType::Bool ? 1 : 4;
            const uint64 Bytes = uint64(Batch) * E.Width * Size;
            const int32 PreviousCapacity=Storage[I].Max();
            Storage[I].SetNumUninitialized((Bytes + 7) / 8, EAllowShrinking::No);
            if(Storage[I].Max()>PreviousCapacity)Counters->StagingGrowthEvents.fetch_add(1);
            Bindings[I] = {Storage[I].GetData(), Bytes};
        }
    }
    std::atomic_flag Busy = ATOMIC_FLAG_INIT;
    TSharedPtr<IModelInstanceCPU> Instance;
    FNPCRowLayout Layout;
    int32 FixedBatch = 0, LastBatch = 0;
    bool bPad = false;
    TSharedRef<FBackendCounters> Counters;
    TArray<TArray<uint64>> InputStorage, OutputStorage;
    TArray<FTensorBindingCPU> Inputs, Outputs;
    TArray<FTensorShape> Shapes;
};

class FNNEBatchBackend : public IBatchInferenceBackend
{
public:
    TSharedPtr<IModelCPU> Model;
    FNPCRowLayout Layout;
    FString Runtime;
    int32 Limit = 0, FixedBatch = 0;
    bool bPad = false;
    TSharedRef<FBackendCounters> Counters = MakeShared<FBackendCounters>();
    virtual int32 GetObservationSize() const override { return Layout.ObsSize; }
    virtual int32 GetActionSize() const override { return Layout.ActSize; }
    virtual int32 GetMaxBatchSize() const override { return Limit; }
    virtual FString GetName() const override { return Runtime; }
    virtual FBatchInferenceBackendStats GetBackendStats() const override
    {
        return {Counters->ReshapeCalls.load(), Counters->PaddingRows.load(), double(Counters->ReshapeNs.load())/1e9, double(Counters->ServiceNs.load())/1e9, Counters->StagingGrowthEvents.load()};
    }
    virtual TUniquePtr<IBatchInferenceContext> CreateContext() override
    {
        check(IsInGameThread());
        auto Instance = Model->CreateModelInstanceCPU();
        if (!Instance) return nullptr;
        return MakeUnique<FNNEBatchContext>(Instance, Layout, FixedBatch, bPad, Counters);
    }
};

bool Describe(TConstArrayView<FTensorDesc> Descs, TArray<FNPCRowElement>& Elements, int32& Size,
              int32& FixedBatch, bool bBasic, FString& Error)
{
    Size = 0;
    if (Descs.IsEmpty() || (bBasic && Descs.Num() != 1))
    { Error = TEXT("BasicCpu requires a single float input and output"); return false; }
    for (const auto& D : Descs)
    {
        const auto Dims = D.GetShape().GetData();
        if (D.GetName().StartsWith(TEXT("state_in")) || D.GetName().StartsWith(TEXT("state_out")))
        { Error = TEXT("Recurrent models are unsupported"); return false; }
        if (Dims.Num() < 1 || (bBasic && Dims.Num() != 2) || (Dims[0] != -1 && Dims[0] <= 0))
        { Error = TEXT("Model needs a leading batch dimension"); return false; }
        if (Dims[0] > 0)
        {
            if (FixedBatch && FixedBatch != Dims[0]) { Error = TEXT("Inconsistent fixed batch dimensions"); return false; }
            FixedBatch = Dims[0];
        }
        int64 Width = 1;
        for (int32 I = 1; I < Dims.Num(); ++I)
        {
            if (Dims[I] <= 0 || Width > MAX_int32 / Dims[I])
            { Error = TEXT("Dynamic non-batch dimensions or oversized tensor"); return false; }
            Width *= Dims[I];
        }
        ENPCRowDType Type;
        switch (D.GetDataType())
        {
        case ENNETensorDataType::Float: Type = ENPCRowDType::Float; break;
        case ENNETensorDataType::Int64: Type = ENPCRowDType::Int64; break;
        case ENNETensorDataType::Boolean: Type = ENPCRowDType::Bool; break;
        default: Error = TEXT("Unsupported tensor dtype (expected float, int64, bool)"); return false;
        }
        if ((bBasic && Type != ENPCRowDType::Float) || Width > MAX_int32 - Size)
        { Error = TEXT("Invalid BasicCpu dtype or row size"); return false; }
        Elements.Add({D.GetName(), Type, Size, int32(Width)});
        Size += int32(Width);
    }
    return true;
}
}

FString UNNEBatchBackendProvider::GetBucketKey() const
{
    return FString::Printf(TEXT("%s:%s:%s:%d:%d"), *GetPathNameSafe(ModelData),
        ModelData ? *ModelData->GetFileId().ToString() : TEXT("none"), *RuntimeName, MaxBatchSize, int32(bPadToPowerOfTwo));
}

TSharedPtr<IBatchInferenceBackend> UNNEBatchBackendProvider::CreateBackend(const FInteractionDefinition& Definition,
    FNPCRowLayout& OutLayout, FString& OutError)
{
    check(IsInGameThread());
    OutError.Reset(); OutLayout = {};
    const bool bBasic = RuntimeName == TEXT("NNERuntimeBasicCpu");
    if (!ModelData || MaxBatchSize < 0 || (RuntimeName != TEXT("NNERuntimeORTCpu") && !bBasic))
    { OutError = TEXT("Select ORT CPU or BasicCpu and a model asset"); return nullptr; }
    auto Runtime = UE::NNE::GetRuntime<INNERuntimeCPU>(RuntimeName);
    if (!Runtime.IsValid()) { OutError = TEXT("CPU runtime plugin is not enabled"); return nullptr; }
    auto Backend = MakeShared<FNNEBatchBackend>();
    Backend->Model = Runtime->CreateModelCPU(ModelData);
    if (!Backend->Model) { OutError = TEXT("Runtime cannot create model (check model file format)"); return nullptr; }
    auto Instance = Backend->Model->CreateModelInstanceCPU();
    if (!Instance || !Describe(Instance->GetInputTensorDescs(), OutLayout.Inputs, OutLayout.ObsSize,
            Backend->FixedBatch, bBasic, OutError) ||
        !Describe(Instance->GetOutputTensorDescs(), OutLayout.Outputs, OutLayout.ActSize,
            Backend->FixedBatch, bBasic, OutError)) return nullptr;
    OutLayout.UpdateHash();
    if (!FNPCRowCodec::Create(Definition, OutLayout, OutError)) return nullptr;
    if (!Definition.ObsSpaceDefn.GetPtr<FDictSpace>() && OutLayout.Inputs[0].Name != TEXT("obs"))
        UE_LOG(LogScholaNNE, Warning, TEXT("Root observation tensor 'obs' not found; binding tensor 0 (%s)"), *OutLayout.Inputs[0].Name);
    if (!Definition.ActionSpaceDefn.GetPtr<FDictSpace>() && OutLayout.Outputs[0].Name != TEXT("action"))
        UE_LOG(LogScholaNNE, Warning, TEXT("Root action tensor 'action' not found; binding tensor 0 (%s)"), *OutLayout.Outputs[0].Name);
    Backend->bPad = bPadToPowerOfTwo;
    Backend->Layout = OutLayout; Backend->Runtime = RuntimeName;
    Backend->Limit = Backend->FixedBatch > 0 ? Backend->FixedBatch : MaxBatchSize;
    if (MaxBatchSize > 0 && Backend->Limit > MaxBatchSize) Backend->Limit = MaxBatchSize;
    FString Options;
    GConfig->GetString(TEXT("/Script/NNERuntimeORT.NNERuntimeORTSettings"), TEXT("EditorThreadingOptions"), Options, GEngineIni);
    UE_LOG(LogScholaNNE, Display, TEXT("Background backend %s; editor threading config: %s; game default intra-op=1"), *RuntimeName, *Options);
    return Backend;
}
