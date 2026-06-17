// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#include "Misc/AutomationTest.h"

#include "NNEUtils/NNEBuffer.h"
#include "NNEUtils/NNEPointCreator.h"
#include "Points/MultiDiscretePoint.h"
#include "Spaces/MultiDiscreteSpace.h"

#define TestEqualExactFloat(TestMessage, Actual, Expected) TestEqual(TestMessage, (float)Actual, (float)Expected, 0.0001f)

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNNEPointCreatorDiscreteBufferTest, "Schola.Policies.NNE.NNEPointCreator.DiscreteBuffer Test", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FNNEPointCreatorDiscreteBufferTest::RunTest(const FString& Parameters)
{
	FMultiDiscreteSpace DiscreteSpace({5,});
    
    TArray<int64> BufferData = {2,};
    FNNEMultiDiscreteBuffer NNEBuffer(DiscreteSpace.GetNumDimensions());
    NNEBuffer.Buffer = BufferData;
    
    TInstancedStruct<FSpace> SpaceStruct;
    SpaceStruct.InitializeAs<FMultiDiscreteSpace>(DiscreteSpace);
    
    TInstancedStruct<FPoint> OutputPoint;
	OutputPoint.InitializeAs<FMultiDiscretePoint>();
	OutputPoint.GetMutable<FMultiDiscretePoint>().Values.Init(0, 1);
    
    TInstancedStruct<FNNEPointBuffer> BufferStruct;
    BufferStruct.InitializeAs<FNNEMultiDiscreteBuffer>(NNEBuffer);
    
    FNNEPointCreator::CreatePoint(BufferStruct, OutputPoint, SpaceStruct);
    
    const FMultiDiscretePoint& ResultPoint = OutputPoint.Get<FMultiDiscretePoint>();
    TestEqual(TEXT("MultiDiscretePoint should have one dimension"), ResultPoint.Values.Num(), 1);
    TestEqual(TEXT("MultiDiscretePoint value should be 2"), ResultPoint.Values[0], 2);
    
    return true;
}
