// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "Inference/NPCRowCodec.h"
#include "Spaces/PointAllocator.h"

void FNPCRowLayout::UpdateHash()
{
    Hash = HashCombineFast(GetTypeHash(ObsSize), GetTypeHash(ActSize));
    for (const auto* Elements : {&Inputs, &Outputs})
    {
        Hash = HashCombineFast(Hash, GetTypeHash(Elements->Num()));
        for (const auto& E : *Elements)
        {
            Hash = HashCombineFast(Hash, GetTypeHash(E.Name));
            Hash = HashCombineFast(Hash, GetTypeHash(uint8(E.DType)));
            Hash = HashCombineFast(Hash, GetTypeHash(E.Offset));
            Hash = HashCombineFast(Hash, GetTypeHash(E.Width));
        }
    }
}

bool FNPCRowCodec::Build(const TInstancedStruct<FSpace>& Space, const TArray<FNPCRowElement>& Elements,
                        int32 RowSize, TArray<FBinding>& Bindings, bool& bDict, FString& Error)
{
    if (!Space.IsValid() || RowSize <= 0) { Error = TEXT("Missing or empty space/row"); return false; }
    const auto* Dict = Space.GetPtr<FDictSpace>();
    bDict = Dict != nullptr;
    if (Elements.Num() != (Dict ? Dict->Spaces.Num() : 1))
    { Error = TEXT("Tensor count does not match space"); return false; }
    TSet<FString> Names;
    int32 Offset = 0;
    for (const auto& E : Elements)
    {
        if (E.Name.StartsWith(TEXT("state_in")) || E.Name.StartsWith(TEXT("state_out")))
        { Error = TEXT("Recurrent tensors are unsupported"); return false; }
        if (E.DType != ENPCRowDType::Float && E.DType != ENPCRowDType::Int64 && E.DType != ENPCRowDType::Bool)
        { Error = TEXT("Unsupported tensor dtype"); return false; }
        const auto* Leaf = Dict ? Dict->Spaces.Find(E.Name) : &Space;
        if (!Leaf || !Leaf->IsValid() || Names.Contains(E.Name))
        { Error = TEXT("Missing or duplicate tensor name"); return false; }
        Names.Add(E.Name);
        FBinding B;
        B.Key = E.Name;
        B.Offset = Offset;
        if (auto* Box = Leaf->GetPtr<FBoxSpace>())
        { B.Type = ESpaceType::Box; B.Width = Box->Dimensions.Num(); B.Shape = Box->Shape; }
        else if (auto* Discrete = Leaf->GetPtr<FDiscreteSpace>())
        { B.Type = ESpaceType::Discrete; B.Width = 1; B.High.Add(Discrete->High); }
        else if (auto* MultiDiscrete = Leaf->GetPtr<FMultiDiscreteSpace>())
        { B.Type = ESpaceType::MultiDiscrete; B.Width = MultiDiscrete->High.Num(); B.High = MultiDiscrete->High; }
        else if (auto* Binary = Leaf->GetPtr<FMultiBinarySpace>())
        { B.Type = ESpaceType::MultiBinary; B.Width = Binary->Shape; }
        else { Error = TEXT("Text and nested Dict spaces are unsupported"); return false; }
        for (int32 High : B.High)
            if (High <= 0 || High > 16777217)
            { Error = TEXT("Discrete range cannot be represented exactly in float rows"); return false; }
        if (B.Width <= 0 || E.Offset != Offset || E.Width != B.Width || B.Width > RowSize - Offset)
        { Error = TEXT("Tensor width/offset does not match index-valued space"); return false; }
        if ((B.Type == ESpaceType::Box && E.DType != ENPCRowDType::Float) ||
            ((B.Type == ESpaceType::Discrete || B.Type == ESpaceType::MultiDiscrete) && E.DType == ENPCRowDType::Bool))
        { Error = TEXT("Tensor dtype is incompatible with space"); return false; }
        Offset += B.Width;
        Bindings.Add(MoveTemp(B));
    }
    if (Offset != RowSize) { Error = TEXT("Row size does not match tensors"); return false; }
    return true;
}

TOptional<FNPCRowCodec> FNPCRowCodec::Create(const FInteractionDefinition& Definition,
                                           const FNPCRowLayout& Layout, FString& OutError)
{
    FNPCRowCodec C;
    C.ObsSize = Layout.ObsSize; C.ActSize = Layout.ActSize;
    OutError.Reset();
    if (!Build(Definition.ObsSpaceDefn, Layout.Inputs, C.ObsSize, C.Inputs, C.bInputDict, OutError) ||
        !Build(Definition.ActionSpaceDefn, Layout.Outputs, C.ActSize, C.Outputs, C.bOutputDict, OutError))
        return {};
    return C;
}

bool FNPCRowCodec::ValidateValues(const FBinding& B, TConstArrayView<float> Values)
{
    for (int32 I = 0; I < Values.Num(); ++I)
    {
        const float V = Values[I];
        if (!FMath::IsFinite(V)) return false;
        if (B.Type == ESpaceType::MultiBinary && V != 0.f && V != 1.f) return false;
        if (B.High.Num() && (V < 0.f || double(V) >= B.High[I] || FMath::FloorToFloat(V) != V)) return false;
    }
    return true;
}

bool FNPCRowCodec::Encode(const TInstancedStruct<FPoint>& Observation, TArrayView<float> Row) const
{
    if (Row.Num() != ObsSize || !Observation.IsValid()) return false;
    auto* Dict = bInputDict ? Observation.GetPtr<FDictPoint>() : nullptr;
    if (bInputDict && (!Dict || Dict->Points.Num() != Inputs.Num())) return false;
    for (const auto& B : Inputs)
    {
        const auto* P = Dict ? Dict->Points.Find(B.Key) : &Observation;
        if (!P) return false;
        auto Dest = Row.Slice(B.Offset, B.Width);
        switch (B.Type)
        {
        case ESpaceType::Box:
            { auto* V = P->GetPtr<FBoxPoint>(); if (!V || V->Values.Num() != B.Width) return false;
              FMemory::Memcpy(Dest.GetData(), V->Values.GetData(), B.Width * sizeof(float)); break; }
        case ESpaceType::Discrete:
            { auto* V = P->GetPtr<FDiscretePoint>(); if (!V || V->Value < 0 || V->Value >= B.High[0]) return false;
              Dest[0] = float(V->Value); break; }
        case ESpaceType::MultiDiscrete:
            { auto* V = P->GetPtr<FMultiDiscretePoint>(); if (!V || V->Values.Num() != B.Width) return false;
              for (int32 I = 0; I < B.Width; ++I) {
                  if (V->Values[I] < 0 || V->Values[I] >= B.High[I]) return false;
                  Dest[I] = float(V->Values[I]); } break; }
        case ESpaceType::MultiBinary:
            { auto* V = P->GetPtr<FMultiBinaryPoint>(); if (!V || V->Values.Num() != B.Width) return false;
              for (int32 I = 0; I < B.Width; ++I) Dest[I] = V->Values[I] ? 1.f : 0.f; break; }
        default: return false;
        }
        if (!ValidateValues(B, Dest)) return false;
    }
    return true;
}

bool FNPCRowCodec::Decode(TConstArrayView<float> Row, TInstancedStruct<FPoint>& Action) const
{
    if (Row.Num() != ActSize) return false;
    for (const auto& B : Outputs)
        if (!ValidateValues(B, Row.Slice(B.Offset, B.Width))) return false;
    if (bOutputDict && !Action.GetPtr<FDictPoint>()) Action.InitializeAs<FDictPoint>();
    if (bOutputDict)
    {
        auto& Points = Action.GetMutable<FDictPoint>().Points;
        for (auto It = Points.CreateIterator(); It; ++It)
            if (!Outputs.ContainsByPredicate([&](const FBinding& B) { return B.Key == It.Key(); })) It.RemoveCurrent();
    }
    for (const auto& B : Outputs)
    {
        auto& P = bOutputDict ? Action.GetMutable<FDictPoint>().Points.FindOrAdd(B.Key) : Action;
        const auto Values = Row.Slice(B.Offset, B.Width);
        switch (B.Type)
        {
        case ESpaceType::Box:
            { if (!P.GetPtr<FBoxPoint>()) P.InitializeAs<FBoxPoint>();
              auto& V = P.GetMutable<FBoxPoint>(); V.Values.SetNumUninitialized(B.Width);
              if (V.Shape != B.Shape) V.Shape = B.Shape;
              FMemory::Memcpy(V.Values.GetData(), Values.GetData(), B.Width * sizeof(float)); break; }
        case ESpaceType::Discrete:
            { if (!P.GetPtr<FDiscretePoint>()) P.InitializeAs<FDiscretePoint>();
              P.GetMutable<FDiscretePoint>().Value = int32(Values[0]); break; }
        case ESpaceType::MultiDiscrete:
            { if (!P.GetPtr<FMultiDiscretePoint>()) P.InitializeAs<FMultiDiscretePoint>();
              auto& V = P.GetMutable<FMultiDiscretePoint>().Values; V.SetNumUninitialized(B.Width);
              for (int32 I = 0; I < B.Width; ++I) V[I] = int32(Values[I]); break; }
        case ESpaceType::MultiBinary:
            { if (!P.GetPtr<FMultiBinaryPoint>()) P.InitializeAs<FMultiBinaryPoint>();
              auto& V = P.GetMutable<FMultiBinaryPoint>().Values; V.SetNumUninitialized(B.Width);
              for (int32 I = 0; I < B.Width; ++I) V[I] = Values[I] != 0.f; break; }
        default: return false;
        }
    }
    return true;
}
