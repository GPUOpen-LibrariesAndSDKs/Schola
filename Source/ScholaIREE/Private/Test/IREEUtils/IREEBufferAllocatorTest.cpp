// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "IREEUtils/IREEBuffer.h"
#include "IREEUtils/IREEBufferAllocator.h"
#include "Spaces/BoxSpace.h"
#include "Spaces/DictSpace.h"
#include "Spaces/DiscreteSpace.h"
#include "Spaces/MultiBinarySpace.h"
#include "Spaces/MultiDiscreteSpace.h"

#if WITH_AUTOMATION_TESTS

#define TestEqualExactFloat(TestMessage, Actual, Expected) TestEqual(TestMessage, (float)Actual, (float)Expected, 0.0001f)

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIREEBufferAllocatorBoxSpaceTest, "Schola.Policies.IREE.IREEBufferAllocator.BoxSpace Test", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FIREEBufferAllocatorBoxSpaceTest::RunTest(const FString& Parameters)
{
	FBoxSpace BoxSpace;
	BoxSpace.Add(-1.0, 1.0);
	BoxSpace.Add(-2.0, 2.0);

	TInstancedStruct<FIREEPointBuffer> IREEBuffer;
	FIREEBufferAllocator::AllocateBuffer(TInstancedStruct<FSpace>::Make<FBoxSpace>(BoxSpace), IREEBuffer);

	TestTrue(TEXT("IREEBuffer is FIREEBoxBuffer"), IREEBuffer.GetPtr<FIREEBoxBuffer>() != nullptr);

	const FIREEBoxBuffer& BoxBuffer = IREEBuffer.Get<FIREEBoxBuffer>();
	TestEqual(TEXT("BoxBuffer.Buffer.Num() == 2"), BoxBuffer.Buffer.Num(), 2);
	TestEqualExactFloat(TEXT("BoxBuffer.Buffer[0] == 0.0f"), BoxBuffer.Buffer[0], 0.0f);
	TestEqualExactFloat(TEXT("BoxBuffer.Buffer[1] == 0.0f"), BoxBuffer.Buffer[1], 0.0f);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIREEBufferAllocatorComplexHierarchicalTest, "Schola.Policies.IREE.IREEBufferAllocator.Complex Hierarchical Test", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FIREEBufferAllocatorComplexHierarchicalTest::RunTest(const FString& Parameters)
{
	// agent_state {
	//   observation { position: Box(3), velocity: Box(2), sensors { vision: Box(12288), audio: MultiDiscrete(10), touch: MultiBinary(8) } }
	//   actions { movement: Discrete(4), interaction: MultiBinary(2) }
	// }
	FDictSpace AgentStateSpace;
	FDictSpace ObservationSpace;

	FBoxSpace PositionSpace;
	PositionSpace.Add(-10.0f, 10.0f);
	PositionSpace.Add(-10.0f, 10.0f);
	PositionSpace.Add(0.0f, 5.0f);
	ObservationSpace.Spaces.Add("position", TInstancedStruct<FSpace>::Make<FBoxSpace>(PositionSpace));

	FBoxSpace VelocitySpace;
	VelocitySpace.Add(-5.0f, 5.0f);
	VelocitySpace.Add(-5.0f, 5.0f);
	ObservationSpace.Spaces.Add("velocity", TInstancedStruct<FSpace>::Make<FBoxSpace>(VelocitySpace));

	FDictSpace SensorsSpace;

	FBoxSpace VisionSpace;
	for (int i = 0; i < 12288; ++i)
	{
		VisionSpace.Add(0.0f, 255.0f);
	}
	SensorsSpace.Spaces.Add("vision", TInstancedStruct<FSpace>::Make<FBoxSpace>(VisionSpace));

	FMultiDiscreteSpace AudioSpace;
	for (int i = 0; i < 10; ++i)
	{
		AudioSpace.Add(100);
	}
	SensorsSpace.Spaces.Add("audio", TInstancedStruct<FSpace>::Make<FMultiDiscreteSpace>(AudioSpace));

	FMultiBinarySpace TouchSpace(8);
	SensorsSpace.Spaces.Add("touch", TInstancedStruct<FSpace>::Make<FMultiBinarySpace>(TouchSpace));

	ObservationSpace.Spaces.Add("sensors", TInstancedStruct<FSpace>::Make<FDictSpace>(SensorsSpace));
	AgentStateSpace.Spaces.Add("observation", TInstancedStruct<FSpace>::Make<FDictSpace>(ObservationSpace));

	FDictSpace ActionSpace;

	FDiscreteSpace MovementSpace(4);
	ActionSpace.Spaces.Add("movement", TInstancedStruct<FSpace>::Make<FDiscreteSpace>(MovementSpace));

	FMultiBinarySpace InteractionSpace(2);
	ActionSpace.Spaces.Add("interaction", TInstancedStruct<FSpace>::Make<FMultiBinarySpace>(InteractionSpace));

	AgentStateSpace.Spaces.Add("actions", TInstancedStruct<FSpace>::Make<FDictSpace>(ActionSpace));

	TInstancedStruct<FIREEPointBuffer> AgentBuffer;
	FIREEBufferAllocator::AllocateBuffer(TInstancedStruct<FSpace>::Make<FDictSpace>(AgentStateSpace), AgentBuffer);

	TestTrue(TEXT("AgentBuffer is FIREEDictBuffer"), AgentBuffer.GetPtr<FIREEDictBuffer>() != nullptr);

	const FIREEDictBuffer& AgentDict = AgentBuffer.Get<FIREEDictBuffer>();
	TestEqual(TEXT("AgentDict has 2 top-level keys"), AgentDict.Buffers.Num(), 2);
	TestTrue(TEXT("AgentDict contains 'observation' key"), AgentDict.Buffers.Contains("observation"));
	TestTrue(TEXT("AgentDict contains 'actions' key"), AgentDict.Buffers.Contains("actions"));

	const TInstancedStruct<FIREEPointBuffer>& ObservationBuffer = AgentDict.Buffers["observation"];
	TestTrue(TEXT("ObservationBuffer is FIREEDictBuffer"), ObservationBuffer.GetPtr<FIREEDictBuffer>() != nullptr);

	const FIREEDictBuffer& ObservationDict = ObservationBuffer.Get<FIREEDictBuffer>();
	TestEqual(TEXT("ObservationDict has 3 keys"), ObservationDict.Buffers.Num(), 3);

	const TInstancedStruct<FIREEPointBuffer>& PositionBuffer = ObservationDict.Buffers["position"];
	TestTrue(TEXT("PositionBuffer is FIREEBoxBuffer"), PositionBuffer.GetPtr<FIREEBoxBuffer>() != nullptr);
	TestEqual(TEXT("PositionBuffer has 3 elements"), PositionBuffer.Get<FIREEBoxBuffer>().Buffer.Num(), 3);

	const TInstancedStruct<FIREEPointBuffer>& VelocityBuffer = ObservationDict.Buffers["velocity"];
	TestTrue(TEXT("VelocityBuffer is FIREEBoxBuffer"), VelocityBuffer.GetPtr<FIREEBoxBuffer>() != nullptr);
	TestEqual(TEXT("VelocityBuffer has 2 elements"), VelocityBuffer.Get<FIREEBoxBuffer>().Buffer.Num(), 2);

	const TInstancedStruct<FIREEPointBuffer>& SensorsBuffer = ObservationDict.Buffers["sensors"];
	TestTrue(TEXT("SensorsBuffer is FIREEDictBuffer"), SensorsBuffer.GetPtr<FIREEDictBuffer>() != nullptr);

	const FIREEDictBuffer& SensorsDict = SensorsBuffer.Get<FIREEDictBuffer>();
	TestEqual(TEXT("SensorsDict has 3 keys"), SensorsDict.Buffers.Num(), 3);

	const TInstancedStruct<FIREEPointBuffer>& VisionBuffer = SensorsDict.Buffers["vision"];
	TestTrue(TEXT("VisionBuffer is FIREEBoxBuffer"), VisionBuffer.GetPtr<FIREEBoxBuffer>() != nullptr);
	TestEqual(TEXT("VisionBuffer has 12288 elements"), VisionBuffer.Get<FIREEBoxBuffer>().Buffer.Num(), 12288);

	const TInstancedStruct<FIREEPointBuffer>& AudioBuffer = SensorsDict.Buffers["audio"];
	TestTrue(TEXT("AudioBuffer is FIREEMultiDiscreteBuffer"), AudioBuffer.GetPtr<FIREEMultiDiscreteBuffer>() != nullptr);
	TestEqual(TEXT("AudioBuffer has 10 elements"), AudioBuffer.Get<FIREEMultiDiscreteBuffer>().Buffer.Num(), 10);

	const TInstancedStruct<FIREEPointBuffer>& TouchBuffer = SensorsDict.Buffers["touch"];
	TestTrue(TEXT("TouchBuffer is FIREEMultiBinaryBuffer"), TouchBuffer.GetPtr<FIREEMultiBinaryBuffer>() != nullptr);
	TestEqual(TEXT("TouchBuffer has 8 elements"), TouchBuffer.Get<FIREEMultiBinaryBuffer>().Buffer.Num(), 8);

	const TInstancedStruct<FIREEPointBuffer>& ActionsBuffer = AgentDict.Buffers["actions"];
	TestTrue(TEXT("ActionsBuffer is FIREEDictBuffer"), ActionsBuffer.GetPtr<FIREEDictBuffer>() != nullptr);

	const FIREEDictBuffer& ActionsDict = ActionsBuffer.Get<FIREEDictBuffer>();
	TestEqual(TEXT("ActionsDict has 2 keys"), ActionsDict.Buffers.Num(), 2);

	const TInstancedStruct<FIREEPointBuffer>& MovementBuffer = ActionsDict.Buffers["movement"];
	TestTrue(TEXT("MovementBuffer is FIREEDiscreteBuffer"), MovementBuffer.GetPtr<FIREEDiscreteBuffer>() != nullptr);
	TestEqual(TEXT("MovementBuffer has 1 element"), MovementBuffer.Get<FIREEDiscreteBuffer>().Buffer.Num(), 1);

	const TInstancedStruct<FIREEPointBuffer>& InteractionBuffer = ActionsDict.Buffers["interaction"];
	TestTrue(TEXT("InteractionBuffer is FIREEMultiBinaryBuffer"), InteractionBuffer.GetPtr<FIREEMultiBinaryBuffer>() != nullptr);
	TestEqual(TEXT("InteractionBuffer has 2 elements"), InteractionBuffer.Get<FIREEMultiBinaryBuffer>().Buffer.Num(), 2);

	TestEqualExactFloat(TEXT("Position[0] initialized to 0"), PositionBuffer.Get<FIREEBoxBuffer>().Buffer[0], 0.0f);
	TestEqualExactFloat(TEXT("Vision[12287] initialized to 0"), VisionBuffer.Get<FIREEBoxBuffer>().Buffer[12287], 0.0f);
	TestEqual(TEXT("Audio[0] initialized to 0"), AudioBuffer.Get<FIREEMultiDiscreteBuffer>().Buffer[0], (int64)0);
	TestEqual(TEXT("Touch[0] initialized to false"), TouchBuffer.Get<FIREEMultiBinaryBuffer>().Buffer[0], false);
	TestEqual(TEXT("Movement[0] initialized to 0"), MovementBuffer.Get<FIREEDiscreteBuffer>().Buffer[0], (int64)0);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIREEBufferAllocatorMultiDiscreteSpaceTest, "Schola.Policies.IREE.IREEBufferAllocator.MultiDiscreteSpace Test", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FIREEBufferAllocatorMultiDiscreteSpaceTest::RunTest(const FString& Parameters)
{
	FMultiDiscreteSpace DiscreteSpace;
	DiscreteSpace.Add(2);
	DiscreteSpace.Add(3);

	TInstancedStruct<FIREEPointBuffer> IREEBuffer;
	FIREEBufferAllocator::AllocateBuffer(TInstancedStruct<FSpace>::Make<FMultiDiscreteSpace>(DiscreteSpace), IREEBuffer);

	TestTrue(TEXT("IREEBuffer is FIREEMultiDiscreteBuffer"), IREEBuffer.GetPtr<FIREEMultiDiscreteBuffer>() != nullptr);

	const FIREEMultiDiscreteBuffer& DiscreteBuffer = IREEBuffer.Get<FIREEMultiDiscreteBuffer>();
	TestEqual(TEXT("DiscreteBuffer.Buffer.Num() == 2"), DiscreteBuffer.Buffer.Num(), 2);
	TestEqual(TEXT("DiscreteBuffer.Buffer[0] == 0"), DiscreteBuffer.Buffer[0], (int64)0);
	TestEqual(TEXT("DiscreteBuffer.Buffer[1] == 0"), DiscreteBuffer.Buffer[1], (int64)0);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIREEBufferAllocatorBinarySpaceTest, "Schola.Policies.IREE.IREEBufferAllocator.BinarySpace Test", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FIREEBufferAllocatorBinarySpaceTest::RunTest(const FString& Parameters)
{
	FMultiBinarySpace BinarySpace(3);

	TInstancedStruct<FIREEPointBuffer> IREEBuffer;
	FIREEBufferAllocator::AllocateBuffer(TInstancedStruct<FSpace>::Make<FMultiBinarySpace>(BinarySpace), IREEBuffer);

	TestTrue(TEXT("IREEBuffer is FIREEMultiBinaryBuffer"), IREEBuffer.GetPtr<FIREEMultiBinaryBuffer>() != nullptr);

	const FIREEMultiBinaryBuffer& BinaryBuffer = IREEBuffer.Get<FIREEMultiBinaryBuffer>();
	TestEqual(TEXT("BinaryBuffer.Buffer.Num() == 3"), BinaryBuffer.Buffer.Num(), 3);
	for (bool Value : BinaryBuffer.Buffer)
	{
		TestFalse(TEXT("BinaryBuffer initialized to false"), Value);
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIREEBufferAllocatorNestedDictSpaceTest, "Schola.Policies.IREE.IREEBufferAllocator.Nested DictSpace Test", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FIREEBufferAllocatorNestedDictSpaceTest::RunTest(const FString& Parameters)
{
	FDictSpace OuterDictSpace;
	FBoxSpace  OuterBoxSpace;
	OuterBoxSpace.Add(-1.0, 1.0);
	OuterDictSpace.Spaces.Add("outer_box", TInstancedStruct<FSpace>::Make<FBoxSpace>(OuterBoxSpace));

	FDictSpace InnerDictSpace;
	FBoxSpace  InnerBoxSpace;
	InnerBoxSpace.Add(-2.0, 2.0);
	InnerDictSpace.Spaces.Add("inner_box", TInstancedStruct<FSpace>::Make<FBoxSpace>(InnerBoxSpace));

	OuterDictSpace.Spaces.Add("inner_dict", TInstancedStruct<FSpace>::Make<FDictSpace>(InnerDictSpace));

	TInstancedStruct<FIREEPointBuffer> IREEBuffer;
	FIREEBufferAllocator::AllocateBuffer(TInstancedStruct<FSpace>::Make<FDictSpace>(OuterDictSpace), IREEBuffer);

	TestTrue(TEXT("IREEBuffer is FIREEDictBuffer"), IREEBuffer.GetPtr<FIREEDictBuffer>() != nullptr);

	const FIREEDictBuffer& OuterDictBuffer = IREEBuffer.Get<FIREEDictBuffer>();
	TestEqual(TEXT("OuterDictBuffer.Buffers.Num() == 2"), OuterDictBuffer.Buffers.Num(), 2);
	TestTrue(TEXT("OuterBoxBuffer is FIREEBoxBuffer"), OuterDictBuffer.Buffers["outer_box"].GetPtr<FIREEBoxBuffer>() != nullptr);

	const TInstancedStruct<FIREEPointBuffer>& InnerDictBuffer = OuterDictBuffer.Buffers["inner_dict"];
	TestTrue(TEXT("InnerDictBuffer is FIREEDictBuffer"), InnerDictBuffer.GetPtr<FIREEDictBuffer>() != nullptr);
	TestTrue(TEXT("InnerDict contains 'inner_box' key"), InnerDictBuffer.Get<FIREEDictBuffer>().Buffers.Contains("inner_box"));

	return true;
}

#undef TestEqualExactFloat

#endif // WITH_AUTOMATION_TESTS
