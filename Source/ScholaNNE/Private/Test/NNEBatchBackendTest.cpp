// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#if WITH_DEV_AUTOMATION_TESTS && WITH_EDITOR
// Fixtures are built from embedded bytes, which needs editor-only UNNEModelData::Init.
#include "Misc/AutomationTest.h"
#include "Inference/NNEBatchBackendProvider.h"
#include "Inference/BackgroundInferenceScheduler.h"
#include "Spaces/PointAllocator.h"
#include "NNE.h"
#include "NNERuntimeCPU.h"
#include <atomic>
#include "BackgroundModels.inl"

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNNEBackgroundEquivalence, "Schola.Inference.Background.NNE.BatchEquivalence",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FNNEBackgroundEquivalence::RunTest(const FString&)
{
    for (int32 Case = 0; Case < 4; ++Case)
    {
        const bool bBasic = Case == 3, bFixed = Case == 1;
        const bool bPadded = Case == 2;
        auto* Provider = NewObject<UNNEBatchBackendProvider>();
        Provider->RuntimeName = bBasic ? TEXT("NNERuntimeBasicCpu") : TEXT("NNERuntimeORTCpu");
        Provider->MaxBatchSize = 64;
        Provider->bPadToPowerOfTwo = bPadded;
        Provider->ModelData = NewObject<UNNEModelData>();
        Provider->ModelData->Init(bBasic ? TEXT("ubnne") : TEXT("onnx"),
            bBasic ? MakeArrayView(BackgroundBasicCpu) : bFixed ? MakeArrayView(BackgroundFixedONNX) : MakeArrayView(BackgroundTypedONNX));
        auto Box = TInstancedStruct<FSpace>::Make<FBoxSpace>(TArray<float>{-100,-100}, TArray<float>{100,100});
        FInteractionDefinition Definition(Box, Box);
        if (!bBasic)
        {
            FDictSpace Obs, Act;
            // Deliberately reverse insertion order relative to model tensor order.
            Obs.Spaces.Add(TEXT("z"), TInstancedStruct<FSpace>::Make<FMultiDiscreteSpace>(TArray<int>{3,4})); Obs.Spaces.Add(TEXT("x"), Box);
            Act.Spaces.Add(TEXT("positive"), TInstancedStruct<FSpace>::Make<FMultiBinarySpace>(2));
            Act.Spaces.Add(TEXT("choice"), TInstancedStruct<FSpace>::Make<FDiscreteSpace>(2));
            Act.Spaces.Add(TEXT("sum"), Box);
            Definition.ObsSpaceDefn = TInstancedStruct<FSpace>::Make<FDictSpace>(Obs);
            Definition.ActionSpaceDefn = TInstancedStruct<FSpace>::Make<FDictSpace>(Act);
        }
        FNPCRowLayout Layout;
        FString Error;
        auto Backend = Provider->CreateBackend(Definition, Layout, Error);
        if (!TestTrue(Provider->RuntimeName + TEXT(": ") + Error, Backend.IsValid())) continue;
        auto Codec = FNPCRowCodec::Create(Definition, Layout, Error);
        if (!TestTrue(TEXT("Codec"), Codec.IsSet())) continue;
        // Warm the largest staging buffers, then vary every smaller batch without growth.
        auto WarmContext=Backend->CreateContext();
        const int32 WarmRows=bFixed?16:64;
        TArray<float> WarmInput,WarmOutput;WarmInput.Init(0,WarmRows*Layout.ObsSize);WarmOutput.Init(0,WarmRows*Layout.ActSize);
        TestTrue(TEXT("Warm staging"),WarmContext->RunBatch(WarmInput.GetData(),WarmRows,WarmOutput.GetData()));
        const int64 GrowthBefore=Backend->GetBackendStats().StagingGrowthEvents;
        for(int32 BatchRows=1;BatchRows<=WarmRows;++BatchRows)
            TestTrue(TEXT("Vary warmed staging"),WarmContext->RunBatch(WarmInput.GetData(),BatchRows,WarmOutput.GetData()));
        TestEqual(TEXT("No provider staging growth after maximum warm-up"),Backend->GetBackendStats().StagingGrowthEvents,GrowthBefore);
        // Independent serial RunSync oracle with the original model tensor bindings.
        auto Runtime = UE::NNE::GetRuntime<INNERuntimeCPU>(Provider->RuntimeName);
        auto Model = Runtime->CreateModelCPU(Provider->ModelData);
        auto Serial = Model->CreateModelInstanceCPU();
        const int32 SerialBatch = bFixed ? 16 : 1;
        const uint32 Dims[] = {uint32(SerialBatch),2};
        TArray<UE::NNE::FTensorShape> Shapes;
        for (int32 I = 0; I < (bBasic ? 1 : 2); ++I) Shapes.Add(UE::NNE::FTensorShape::Make(MakeArrayView(Dims)));
        TestTrue(TEXT("Serial shape"), Serial->SetInputTensorShapes(Shapes) == UE::NNE::EResultStatus::Ok);
        FNPCBackgroundSchedulerSettings Settings;
        Settings.MaxBatchSize = 64; Settings.WorkerCount = 2;
        auto Scheduler = FNPCBackgroundScheduler::Create(Settings, Backend.ToSharedRef(), &Error);
        if (!TestTrue(TEXT("Scheduler"), Scheduler.IsValid())) continue;
        for (int32 Rows = 1; Rows <= 129; ++Rows)
        {
            TArray<float> Input, Expected;
            Input.SetNumUninitialized(Rows * Layout.ObsSize);
            Expected.SetNumUninitialized(Rows * Layout.ActSize);
            for (int32 R = 0; R < Rows; ++R)
            {
                float X[32] = {float(R % 7) - 3.f, float(R % 11) - 5.f};
                int64 Z[32] = {1,2};
                float Y[32] = {}; int64 Choice[16] = {}; uint8 Positive[32] = {};
                TArray<UE::NNE::FTensorBindingCPU> Ins = {{X,uint64(SerialBatch)*2*sizeof(float)}};
                TArray<UE::NNE::FTensorBindingCPU> Outs = {{Y,uint64(SerialBatch)*2*sizeof(float)}};
                if (!bBasic) { Ins.Add({Z,uint64(SerialBatch)*2*sizeof(int64)}); Outs.Add({Choice,uint64(SerialBatch)*sizeof(int64)}); Outs.Add({Positive,uint64(SerialBatch)*2}); }
                if (!TestTrue(TEXT("Serial RunSync"), Serial->RunSync(Ins, Outs) == UE::NNE::EResultStatus::Ok)) return false;
                FMemory::Memcpy(Input.GetData() + R * Layout.ObsSize, X, 2*sizeof(float));
                FMemory::Memcpy(Expected.GetData() + R * Layout.ActSize, Y, 2*sizeof(float));
                if (!bBasic)
                {
                    Input[R * Layout.ObsSize + 2] = float(Z[0]); Input[R * Layout.ObsSize + 3] = float(Z[1]);
                    Expected[R * Layout.ActSize + 2] = float(Choice[0]);
                    Expected[R * Layout.ActSize + 3] = float(Positive[0]);
                    Expected[R * Layout.ActSize + 4] = float(Positive[1]);
                }
            }
            FNPCBackgroundRequest Request;
            TestTrue(TEXT("Admitted"), Scheduler->Submit(Input, {}, Request) == ENPCBackgroundStatus::Pending);
            if (!TestTrue(TEXT("Completed"), Scheduler->Wait(Request, 5) == ENPCBackgroundStatus::Completed)) return false;
            FNPCBackgroundResult Result;
            TestTrue(TEXT("Consumed"), Scheduler->Consume(Request, Result) == ENPCBackgroundStatus::Completed);
            for (int32 I = 0; I < Expected.Num(); ++I)
                if (!TestTrue(TEXT("Batched equals serial within tolerance"),
                    FMath::Abs(Result.Actions[I] - Expected[I]) <= 1e-5f + 1e-4f * FMath::Abs(Expected[I]))) return false;
            TInstancedStruct<FPoint> Action;
            TestTrue(TEXT("Typed outputs decode"), Codec->Decode(MakeArrayView(Result.Actions).Slice(0,Layout.ActSize), Action));
        }
    }
    return true;
}
#endif

#if WITH_DEV_AUTOMATION_TESTS && WITH_EDITOR
namespace
{
FInteractionDefinition TypedFixtureDefinition()
{
    auto Box = TInstancedStruct<FSpace>::Make<FBoxSpace>(TArray<float>{-100,-100}, TArray<float>{100,100});
    FDictSpace Obs, Act;
    Obs.Spaces.Add(TEXT("x"), Box); Obs.Spaces.Add(TEXT("z"), TInstancedStruct<FSpace>::Make<FMultiDiscreteSpace>(TArray<int>{3,4}));
    Act.Spaces.Add(TEXT("sum"), Box); Act.Spaces.Add(TEXT("choice"), TInstancedStruct<FSpace>::Make<FDiscreteSpace>(2));
    Act.Spaces.Add(TEXT("positive"), TInstancedStruct<FSpace>::Make<FMultiBinarySpace>(2));
    return FInteractionDefinition(TInstancedStruct<FSpace>::Make<FDictSpace>(Obs), TInstancedStruct<FSpace>::Make<FDictSpace>(Act));
}

/** Records any context entered by two threads at once, or used by more than one thread over its lifetime. */
struct FContextUseDetector
{
    std::atomic<bool> bOverlap{false}, bMigrated{false};
    std::atomic<int32> Contexts{0};
};
class FDetectingContext : public IBatchInferenceContext
{
public:
    FDetectingContext(TUniquePtr<IBatchInferenceContext> InInner, TSharedRef<FContextUseDetector> InDetector)
        : Inner(MoveTemp(InInner)), Detector(InDetector) {}
    virtual bool RunBatch(const float* Observations, int32 Rows, float* Actions) override
    {
        if (Entered.fetch_add(1) != 0) Detector->bOverlap = true;
        uint32 Expected = 0;
        const uint32 Self = FPlatformTLS::GetCurrentThreadId();
        if (!Owner.compare_exchange_strong(Expected, Self) && Expected != Self) Detector->bMigrated = true;
        const bool bOk = Inner->RunBatch(Observations, Rows, Actions);
        Entered.fetch_sub(1);
        return bOk;
    }
private:
    TUniquePtr<IBatchInferenceContext> Inner;
    TSharedRef<FContextUseDetector> Detector;
    std::atomic<int32> Entered{0};
    std::atomic<uint32> Owner{0};
};
class FDetectingBackend : public IBatchInferenceBackend
{
public:
    FDetectingBackend(TSharedRef<IBatchInferenceBackend> InInner, TSharedRef<FContextUseDetector> InDetector) : Inner(InInner), Detector(InDetector) {}
    virtual int32 GetObservationSize() const override { return Inner->GetObservationSize(); }
    virtual int32 GetActionSize() const override { return Inner->GetActionSize(); }
    virtual int32 GetMaxBatchSize() const override { return Inner->GetMaxBatchSize(); }
    virtual FString GetName() const override { return Inner->GetName(); }
    virtual TUniquePtr<IBatchInferenceContext> CreateContext() override
    {
        auto Context = Inner->CreateContext();
        if (!Context) return nullptr;
        ++Detector->Contexts;
        return MakeUnique<FDetectingContext>(MoveTemp(Context), Detector);
    }
private:
    TSharedRef<IBatchInferenceBackend> Inner;
    TSharedRef<FContextUseDetector> Detector;
};
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNNEBackgroundContextOwnership, "Schola.Inference.Background.NNE.ContextsNeverShared",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FNNEBackgroundContextOwnership::RunTest(const FString&)
{
    auto* Provider = NewObject<UNNEBatchBackendProvider>();
    Provider->RuntimeName = TEXT("NNERuntimeORTCpu");
    Provider->MaxBatchSize = 8;
    Provider->ModelData = NewObject<UNNEModelData>();
    Provider->ModelData->Init(TEXT("onnx"), MakeArrayView(BackgroundTypedONNX));
    FNPCRowLayout Layout; FString Error;
    auto Backend = Provider->CreateBackend(TypedFixtureDefinition(), Layout, Error);
    if (!TestTrue(TEXT("Backend: ") + Error, Backend.IsValid())) return false;
    for (const ENPCBackgroundDispatch Dispatch : {ENPCBackgroundDispatch::SharedQueue, ENPCBackgroundDispatch::WorkStealing, ENPCBackgroundDispatch::RowRoundRobin})
    {
        auto Detector = MakeShared<FContextUseDetector>();
        FNPCBackgroundSchedulerSettings Settings;
        Settings.WorkerCount = 4; Settings.MaxBatchSize = 8; Settings.Dispatch = Dispatch;
        Settings.MaxInFlightBatchesPerWorker = 2;
        auto Scheduler = FNPCBackgroundScheduler::Create(Settings, MakeShared<FDetectingBackend>(Backend.ToSharedRef(), Detector), &Error);
        if (!TestTrue(TEXT("Scheduler: ") + Error, Scheduler.IsValid())) return false;
        TArray<float> Input; Input.Init(1.f, 8 * Layout.ObsSize);
        TArray<FNPCBackgroundRequest> Requests;
        for (int32 I = 0; I < 400; ++I)
        {
            FNPCBackgroundRequest Request;
            if (Scheduler->Submit(MakeArrayView(Input).Left((I % 8 + 1) * Layout.ObsSize), {}, Request) == ENPCBackgroundStatus::Pending) Requests.Add(Request);
        }
        FNPCBackgroundResult Result;
        bool bCompleted = true;
        for (const auto& Request : Requests)
            bCompleted &= Scheduler->Wait(Request, 10) == ENPCBackgroundStatus::Completed && Scheduler->Consume(Request, Result) == ENPCBackgroundStatus::Completed;
        TestTrue(TEXT("All batches complete"), bCompleted && Requests.Num() > 0);
        TestEqual(TEXT("One NNE context per worker"), Detector->Contexts.load(), 4);
        TestFalse(TEXT("No NNE context entered concurrently"), Detector->bOverlap.load());
        TestFalse(TEXT("Each NNE context stays on one worker thread"), Detector->bMigrated.load());
        Scheduler->Shutdown();
    }
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNNEBackgroundRejectsRecurrent, "Schola.Inference.Background.NNE.RejectsRecurrentModels",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FNNEBackgroundRejectsRecurrent::RunTest(const FString&)
{
    auto* Provider = NewObject<UNNEBatchBackendProvider>();
    Provider->RuntimeName = TEXT("NNERuntimeORTCpu");
    Provider->ModelData = NewObject<UNNEModelData>();
    Provider->ModelData->Init(TEXT("onnx"), MakeArrayView(BackgroundRecurrentONNX));
    auto Box = TInstancedStruct<FSpace>::Make<FBoxSpace>(TArray<float>{-100,-100}, TArray<float>{100,100});
    FDictSpace Obs, Act;
    Obs.Spaces.Add(TEXT("x"), Box); Obs.Spaces.Add(TEXT("state_in"), Box);
    Act.Spaces.Add(TEXT("action"), Box); Act.Spaces.Add(TEXT("state_out"), Box);
    FInteractionDefinition Definition(TInstancedStruct<FSpace>::Make<FDictSpace>(Obs), TInstancedStruct<FSpace>::Make<FDictSpace>(Act));
    FNPCRowLayout Layout; FString Error;
    TestFalse(TEXT("Recurrent model rejected at CreateBackend"), Provider->CreateBackend(Definition, Layout, Error).IsValid());
    TestTrue(TEXT("Rejection names the recurrent signature: ") + Error, Error.Contains(TEXT("Recurrent")));
    return true;
}
#endif
