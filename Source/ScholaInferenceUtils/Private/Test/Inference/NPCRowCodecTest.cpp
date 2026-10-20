// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#if WITH_DEV_AUTOMATION_TESTS
#include "Misc/AutomationTest.h"
#include <limits>
#include "Inference/NPCRowCodec.h"
#include "Spaces/PointAllocator.h"
#include "Spaces/TextSpace.h"

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNPCRowCodecTest, "Schola.Inference.Background.Codec.RoundTrips",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FNPCRowCodecTest::RunTest(const FString&)
{
    TArray<TInstancedStruct<FSpace>> Spaces;
    Spaces.Add(TInstancedStruct<FSpace>::Make<FBoxSpace>(TArray<float>{-10,-10}, TArray<float>{10,10}));
    Spaces.Add(TInstancedStruct<FSpace>::Make<FDiscreteSpace>(9));
    Spaces.Add(TInstancedStruct<FSpace>::Make<FMultiDiscreteSpace>(TArray<int>{3,7}));
    Spaces.Add(TInstancedStruct<FSpace>::Make<FMultiBinarySpace>(3));
    TArray<TInstancedStruct<FPoint>> Points;
    Points.Add(TInstancedStruct<FPoint>::Make<FBoxPoint>(TArray<float>{2,3}));
    Points.Add(TInstancedStruct<FPoint>::Make<FDiscretePoint>(8));
    Points.Add(TInstancedStruct<FPoint>::Make<FMultiDiscretePoint>(TArray<int>{2,6}));
    Points.Add(TInstancedStruct<FPoint>::Make<FMultiBinaryPoint>(TArray<bool>{true,false,true}));
    const int32 Widths[] = {2,1,2,3};
    for (int32 I = 0; I < Spaces.Num(); ++I)
    {
        FNPCRowLayout L;
        L.ObsSize = L.ActSize = Widths[I];
        L.Inputs = L.Outputs = {{TEXT("value"), ENPCRowDType::Float, 0, Widths[I]}};
        FString Error;
        auto C = FNPCRowCodec::Create(FInteractionDefinition(Spaces[I], Spaces[I]), L, Error);
        if (!TestTrue(TEXT("Create codec: ") + Error, C.IsSet())) continue;
        TArray<float> Row; Row.SetNumUninitialized(Widths[I]);
        TInstancedStruct<FPoint> Out;
        TestTrue(TEXT("Encode"), C->Encode(Points[I], Row));
        TestTrue(TEXT("Decode"), C->Decode(Row, Out));
        // Compare transport to avoid Box shape metadata differences.
        TArray<float> Again; Again.SetNumUninitialized(Widths[I]);
        TestTrue(TEXT("Encode decoded"), C->Encode(Out, Again));
        TestTrue(TEXT("Index-valued round trip"), Row == Again);
        const auto* Storage = Out.GetMemory();
        for (int32 Repeat = 0; Repeat < 100; ++Repeat) TestTrue(TEXT("Reuse decode"), C->Decode(Row, Out));
        TestTrue(TEXT("Point storage reused"), Storage == Out.GetMemory());
        Row[0] = std::numeric_limits<float>::infinity();
        auto Previous = Out;
        TestFalse(TEXT("Reject nonfinite output"), C->Decode(Row, Out));
        TestTrue(TEXT("Failed decode preserves output"), Previous == Out);
    }
    FDictSpace Dict;
    Dict.Spaces.Add(TEXT("discrete"), Spaces[1]);
    Dict.Spaces.Add(TEXT("box"), Spaces[0]);
    FDictPoint DictPoint;
    DictPoint.Points.Add(TEXT("discrete"), Points[1]);
    DictPoint.Points.Add(TEXT("box"), Points[0]);
    auto DS = TInstancedStruct<FSpace>::Make<FDictSpace>(Dict);
    auto DP = TInstancedStruct<FPoint>::Make<FDictPoint>(DictPoint);
    FNPCRowLayout L;
    L.ObsSize = L.ActSize = 3;
    L.Inputs = L.Outputs = {{TEXT("box"), ENPCRowDType::Float, 0, 2}, {TEXT("discrete"), ENPCRowDType::Int64, 2, 1}};
    FString Error;
    auto C = FNPCRowCodec::Create(FInteractionDefinition(DS, DS), L, Error);
    if (!TestTrue(TEXT("Dict codec"), C.IsSet())) return false;
    TArray<float> Row; Row.SetNumUninitialized(3);
    TestTrue(TEXT("Dict encode"), C->Encode(DP, Row));
    TestTrue(TEXT("Model tensor order"), Row == TArray<float>({2,3,8}));
    TInstancedStruct<FPoint> Out;
    TestTrue(TEXT("Dict decode"), C->Decode(Row, Out));
    TestTrue(TEXT("Dict roundtrip"), C->Encode(Out, Row));
    L.Inputs[0].Name = TEXT("missing");
    TestFalse(TEXT("Missing key rejected"), FNPCRowCodec::Create(FInteractionDefinition(DS, DS), L, Error).IsSet());
    L.Inputs[0].Name = TEXT("state_in0");
    TestFalse(TEXT("Recurrent rejected"), FNPCRowCodec::Create(FInteractionDefinition(DS, DS), L, Error).IsSet());
    auto Text = TInstancedStruct<FSpace>::Make<FTextSpace>();
    L.ObsSize = L.ActSize = 1;
    L.Inputs = L.Outputs = {{TEXT("text"), ENPCRowDType::Float, 0, 1}};
    TestFalse(TEXT("Text rejected"), FNPCRowCodec::Create(FInteractionDefinition(Text, Text), L, Error).IsSet());
    return true;
}
#endif

#if WITH_DEV_AUTOMATION_TESTS
namespace
{
// Every heap buffer a decoded point owns. Decode has no other allocation site, so unchanged
// buffers across repeated calls mean no allocation after warm-up.
void CollectPointBuffers(const TInstancedStruct<FPoint>& Point, TArray<const void*>& Out)
{
    Out.Add(Point.GetMemory());
    if (const auto* Box = Point.GetPtr<FBoxPoint>()) { Out.Add(Box->Values.GetData()); Out.Add(Box->Shape.GetData()); }
    else if (const auto* Multi = Point.GetPtr<FMultiDiscretePoint>()) Out.Add(Multi->Values.GetData());
    else if (const auto* Binary = Point.GetPtr<FMultiBinaryPoint>()) Out.Add(Binary->Values.GetData());
    else if (const auto* Dict = Point.GetPtr<FDictPoint>())
        for (const auto& Pair : Dict->Points) { Out.Add(&Pair); Out.Add(*Pair.Key); CollectPointBuffers(Pair.Value, Out); }
}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNPCRowCodecSteadyStateTest, "Schola.Inference.Background.Codec.NoAllocationAfterWarmup",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FNPCRowCodecSteadyStateTest::RunTest(const FString&)
{
    FDictSpace Dict;
    Dict.Spaces.Add(TEXT("binary"), TInstancedStruct<FSpace>::Make<FMultiBinarySpace>(3));
    Dict.Spaces.Add(TEXT("box"), TInstancedStruct<FSpace>::Make<FBoxSpace>(TArray<float>{-10,-10}, TArray<float>{10,10}));
    Dict.Spaces.Add(TEXT("discrete"), TInstancedStruct<FSpace>::Make<FDiscreteSpace>(9));
    Dict.Spaces.Add(TEXT("multi"), TInstancedStruct<FSpace>::Make<FMultiDiscreteSpace>(TArray<int>{3,7}));
    FDictPoint Observation;
    Observation.Points.Add(TEXT("binary"), TInstancedStruct<FPoint>::Make<FMultiBinaryPoint>(TArray<bool>{true,false,true}));
    Observation.Points.Add(TEXT("box"), TInstancedStruct<FPoint>::Make<FBoxPoint>(TArray<float>{2,3}));
    Observation.Points.Add(TEXT("discrete"), TInstancedStruct<FPoint>::Make<FDiscretePoint>(8));
    Observation.Points.Add(TEXT("multi"), TInstancedStruct<FPoint>::Make<FMultiDiscretePoint>(TArray<int>{2,6}));
    const auto Space = TInstancedStruct<FSpace>::Make<FDictSpace>(Dict);
    const auto Point = TInstancedStruct<FPoint>::Make<FDictPoint>(Observation);
    FNPCRowLayout L;
    L.ObsSize = L.ActSize = 8;
    L.Inputs = L.Outputs = {{TEXT("box"), ENPCRowDType::Float, 0, 2}, {TEXT("discrete"), ENPCRowDType::Int64, 2, 1},
        {TEXT("multi"), ENPCRowDType::Int64, 3, 2}, {TEXT("binary"), ENPCRowDType::Bool, 5, 3}};
    FString Error;
    auto C = FNPCRowCodec::Create(FInteractionDefinition(Space, Space), L, Error);
    if (!TestTrue(TEXT("Create codec: ") + Error, C.IsSet())) return false;
    TArray<float> Row; Row.SetNumUninitialized(8);
    TInstancedStruct<FPoint> Action;
    TestTrue(TEXT("Warm encode"), C->Encode(Point, Row));
    TestTrue(TEXT("Warm decode"), C->Decode(Row, Action));
    TArray<const void*> Warm, Steady;
    CollectPointBuffers(Action, Warm);
    const float* RowData = Row.GetData();
    bool bOk = true;
    for (int32 Repeat = 0; Repeat < 1000; ++Repeat)
    {
        Row[1] = float(Repeat % 7);
        bOk &= C->Decode(Row, Action) && C->Encode(Action, Row);
    }
    TestTrue(TEXT("Steady-state encode/decode"), bOk);
    CollectPointBuffers(Action, Steady);
    TestEqual(TEXT("Buffer count unchanged"), Steady.Num(), Warm.Num());
    TestTrue(TEXT("Decode reuses every point buffer after warm-up"), Steady == Warm);
    TestTrue(TEXT("Encode writes into the caller's row"), Row.GetData() == RowData);
    return true;
}
#endif
