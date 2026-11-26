// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "Misc/AutomationTest.h"

#include "IREEUtils/IREEBuffer.h"
#include "IREEUtils/IREEPointCreator.h"
#include "Points/MultiDiscretePoint.h"
#include "Spaces/MultiDiscreteSpace.h"

#if WITH_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIREEPointCreatorDiscreteBufferTest, "Schola.Policies.IREE.IREEPointCreator.DiscreteBuffer Test", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FIREEPointCreatorDiscreteBufferTest::RunTest(const FString& Parameters)
{
	FMultiDiscreteSpace DiscreteSpace({ 5 });

	TArray<int64>			 BufferData = { 2 };
	FIREEMultiDiscreteBuffer IREEBuffer(DiscreteSpace.GetNumDimensions());
	IREEBuffer.Buffer = BufferData;

	TInstancedStruct<FSpace> SpaceStruct;
	SpaceStruct.InitializeAs<FMultiDiscreteSpace>(DiscreteSpace);

	TInstancedStruct<FPoint> OutputPoint;
	OutputPoint.InitializeAs<FMultiDiscretePoint>();
	OutputPoint.GetMutable<FMultiDiscretePoint>().Values.Init(0, 1);

	TInstancedStruct<FIREEPointBuffer> BufferStruct;
	BufferStruct.InitializeAs<FIREEMultiDiscreteBuffer>(IREEBuffer);

	FIREEPointCreator::CreatePoint(BufferStruct, OutputPoint, SpaceStruct);

	const FMultiDiscretePoint& ResultPoint = OutputPoint.Get<FMultiDiscretePoint>();
	TestEqual(TEXT("MultiDiscretePoint should have one dimension"), ResultPoint.Values.Num(), 1);
	TestEqual(TEXT("MultiDiscretePoint value should be 2"), ResultPoint.Values[0], 2);

	return true;
}

#endif // WITH_AUTOMATION_TESTS
