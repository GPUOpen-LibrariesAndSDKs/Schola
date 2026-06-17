// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "NNEUtils/NNEBuffer.h"
#include "Spaces/BoxSpace.h"
#include "Spaces/DictSpace.h"
#include "Spaces/DiscreteSpace.h"
#include "Spaces/MultiBinarySpace.h"
#include "Spaces/MultiDiscreteSpace.h"
#include "NNEUtils/NNEBufferAllocator.h"

#define TestEqualExactFloat(TestMessage, Actual, Expected) TestEqual(TestMessage, (float)Actual, (float)Expected, 0.0001f)

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNNEBufferAllocatorBoxSpaceTest, "Schola.Policies.NNE.NNEBufferAllocator.BoxSpace Test", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FNNEBufferAllocatorBoxSpaceTest::RunTest(const FString& Parameters)
{
    FBoxSpace BoxSpace;
    BoxSpace.Add(-1.0, 1.0);
    BoxSpace.Add(-2.0, 2.0);

    TInstancedStruct<FNNEPointBuffer> NNEBuffer;
    FNNEBufferAllocator::AllocateBuffer(TInstancedStruct<FSpace>::Make<FBoxSpace>(BoxSpace), NNEBuffer);

    TestTrue(TEXT("NNEBuffer is FNNEBoxBuffer"), NNEBuffer.GetPtr<FNNEBoxBuffer>() != nullptr);
    
    const FNNEBoxBuffer& BoxBuffer = NNEBuffer.Get<FNNEBoxBuffer>();
    TestEqual(TEXT("BoxBuffer.Buffer.Num() == 2"), BoxBuffer.Buffer.Num(), 2);
    TestEqualExactFloat(TEXT("BoxBuffer.Buffer[0] == 0.0f"), BoxBuffer.Buffer[0], 0.0f);
    TestEqualExactFloat(TEXT("BoxBuffer.Buffer[1] == 0.0f"), BoxBuffer.Buffer[1], 0.0f);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNNEBufferAllocatorComplexHierarchicalTest, "Schola.Policies.NNE.NNEBufferAllocator.Complex Hierarchical Test", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FNNEBufferAllocatorComplexHierarchicalTest::RunTest(const FString& Parameters)
{
    // Create a complex hierarchical space structure representing an agent state
    // Structure:
    // agent_state {
    //   observation {
    //     position -> BoxSpace(3D: x,y,z)
    //     velocity -> BoxSpace(2D: vx,vy)
    //     sensors {
    //       vision -> BoxSpace(64x64x3 RGB image = 12288 elements)
    //       audio -> DiscreteSpace(10 frequency bins)
    //       touch -> BinarySpace(8 touch sensors)
    //     }
    //   }
    //   actions {
    //     movement -> DiscreteSpace(4 directions)
    //     interaction -> BinarySpace(2 buttons)
    //   }
    // }

    // Create the main agent state dictionary
    FDictSpace AgentStateSpace;

    FDictSpace ObservationSpace;
    
    // Position: 3D vector (x, y, z)
    FBoxSpace PositionSpace;
    PositionSpace.Add(-10.0f, 10.0f);  // x
    PositionSpace.Add(-10.0f, 10.0f);  // y
    PositionSpace.Add(0.0f, 5.0f);     // z
    ObservationSpace.Spaces.Add("position", TInstancedStruct<FSpace>::Make<FBoxSpace>(PositionSpace));
    
    // Velocity: 2D vector (vx, vy)
    FBoxSpace VelocitySpace;
    VelocitySpace.Add(-5.0f, 5.0f);    // vx
    VelocitySpace.Add(-5.0f, 5.0f);    // vy
    ObservationSpace.Spaces.Add("velocity", TInstancedStruct<FSpace>::Make<FBoxSpace>(VelocitySpace));
    
    // Sensors sub-dictionary
    FDictSpace SensorsSpace;
    
    // Vision: RGB image 64x64x3 = 12288 elements
    FBoxSpace VisionSpace;
    for (int i = 0; i < 12288; ++i)
    {
        VisionSpace.Add(0.0f, 255.0f);  // RGB values 0-255
    }
    SensorsSpace.Spaces.Add("vision", TInstancedStruct<FSpace>::Make<FBoxSpace>(VisionSpace));
    
    // Audio: 10 frequency bins
    FMultiDiscreteSpace AudioSpace;
    for (int i = 0; i < 10; ++i)
    {
        AudioSpace.Add(100);  // Each bin can have values 0-99
    }
    SensorsSpace.Spaces.Add("audio", TInstancedStruct<FSpace>::Make<FMultiDiscreteSpace>(AudioSpace));
    
    // Touch: 8 binary sensors
    FMultiBinarySpace TouchSpace(8);
	SensorsSpace.Spaces.Add("touch", TInstancedStruct<FSpace>::Make<FMultiBinarySpace>(TouchSpace));
    
    ObservationSpace.Spaces.Add("sensors", TInstancedStruct<FSpace>::Make<FDictSpace>(SensorsSpace));
    AgentStateSpace.Spaces.Add("observation", TInstancedStruct<FSpace>::Make<FDictSpace>(ObservationSpace));

    FDictSpace ActionSpace;
    
    // Movement: 4 directions (up, down, left, right)
	FDiscreteSpace MovementSpace(4); // 0=up, 1=down, 2=left, 3=right
    ActionSpace.Spaces.Add("movement", TInstancedStruct<FSpace>::Make<FDiscreteSpace>(MovementSpace));
    
    // Interaction: 2 binary buttons (grab, release)
    FMultiBinarySpace InteractionSpace(2);
	ActionSpace.Spaces.Add("interaction", TInstancedStruct<FSpace>::Make<FMultiBinarySpace>(InteractionSpace));
    
    AgentStateSpace.Spaces.Add("actions", TInstancedStruct<FSpace>::Make<FDictSpace>(ActionSpace));

    TInstancedStruct<FNNEPointBuffer> AgentBuffer;
    FNNEBufferAllocator::AllocateBuffer(TInstancedStruct<FSpace>::Make<FDictSpace>(AgentStateSpace), AgentBuffer);

    TestTrue(TEXT("AgentBuffer is FNNEDictBuffer"), AgentBuffer.GetPtr<FNNEDictBuffer>() != nullptr);
    
    const FNNEDictBuffer& AgentDict = AgentBuffer.Get<FNNEDictBuffer>();
    TestEqual(TEXT("AgentDict has 2 top-level keys"), AgentDict.Buffers.Num(), 2);
    TestTrue(TEXT("AgentDict contains 'observation' key"), AgentDict.Buffers.Contains("observation"));
    TestTrue(TEXT("AgentDict contains 'actions' key"), AgentDict.Buffers.Contains("actions"));

    const TInstancedStruct<FNNEPointBuffer>& ObservationBuffer = AgentDict.Buffers["observation"];
    TestTrue(TEXT("ObservationBuffer is FNNEDictBuffer"), ObservationBuffer.GetPtr<FNNEDictBuffer>() != nullptr);
    
    const FNNEDictBuffer& ObservationDict = ObservationBuffer.Get<FNNEDictBuffer>();
    TestEqual(TEXT("ObservationDict has 3 keys"), ObservationDict.Buffers.Num(), 3);
    TestTrue(TEXT("ObservationDict contains 'position'"), ObservationDict.Buffers.Contains("position"));
    TestTrue(TEXT("ObservationDict contains 'velocity'"), ObservationDict.Buffers.Contains("velocity"));
    TestTrue(TEXT("ObservationDict contains 'sensors'"), ObservationDict.Buffers.Contains("sensors"));

    // Verify position buffer (3D)
    const TInstancedStruct<FNNEPointBuffer>& PositionBuffer = ObservationDict.Buffers["position"];
    TestTrue(TEXT("PositionBuffer is FNNEBoxBuffer"), PositionBuffer.GetPtr<FNNEBoxBuffer>() != nullptr);
    TestEqual(TEXT("PositionBuffer has 3 elements"), PositionBuffer.Get<FNNEBoxBuffer>().Buffer.Num(), 3);

    // Verify velocity buffer (2D)
    const TInstancedStruct<FNNEPointBuffer>& VelocityBuffer = ObservationDict.Buffers["velocity"];
    TestTrue(TEXT("VelocityBuffer is FNNEBoxBuffer"), VelocityBuffer.GetPtr<FNNEBoxBuffer>() != nullptr);
    TestEqual(TEXT("VelocityBuffer has 2 elements"), VelocityBuffer.Get<FNNEBoxBuffer>().Buffer.Num(), 2);

    const TInstancedStruct<FNNEPointBuffer>& SensorsBuffer = ObservationDict.Buffers["sensors"];
    TestTrue(TEXT("SensorsBuffer is FNNEDictBuffer"), SensorsBuffer.GetPtr<FNNEDictBuffer>() != nullptr);
    
    const FNNEDictBuffer& SensorsDict = SensorsBuffer.Get<FNNEDictBuffer>();
    TestEqual(TEXT("SensorsDict has 3 keys"), SensorsDict.Buffers.Num(), 3);
    TestTrue(TEXT("SensorsDict contains 'vision'"), SensorsDict.Buffers.Contains("vision"));
    TestTrue(TEXT("SensorsDict contains 'audio'"), SensorsDict.Buffers.Contains("audio"));
    TestTrue(TEXT("SensorsDict contains 'touch'"), SensorsDict.Buffers.Contains("touch"));

    // Verify vision buffer (12288 elements for 64x64x3 RGB)
    const TInstancedStruct<FNNEPointBuffer>& VisionBuffer = SensorsDict.Buffers["vision"];
    TestTrue(TEXT("VisionBuffer is FNNEBoxBuffer"), VisionBuffer.GetPtr<FNNEBoxBuffer>() != nullptr);
    TestEqual(TEXT("VisionBuffer has 12288 elements"), VisionBuffer.Get<FNNEBoxBuffer>().Buffer.Num(), 12288);

    // Verify audio buffer (10 selected categorical values, one per audio bin)
    const TInstancedStruct<FNNEPointBuffer>& AudioBuffer = SensorsDict.Buffers["audio"];
    TestTrue(TEXT("AudioBuffer is FNNEMultiDiscreteBuffer"), AudioBuffer.GetPtr<FNNEMultiDiscreteBuffer>() != nullptr);
	TestEqual(TEXT("AudioBuffer has 10 elements"), AudioBuffer.Get<FNNEMultiDiscreteBuffer>().Buffer.Num(), 10);

    // Verify touch buffer (8 binary elements)
    const TInstancedStruct<FNNEPointBuffer>& TouchBuffer = SensorsDict.Buffers["touch"];
	TestTrue(TEXT("TouchBuffer is FNNEBinaryBuffer"), TouchBuffer.GetPtr<FNNEMultiBinaryBuffer>() != nullptr);
	TestEqual(TEXT("TouchBuffer has 8 elements"), TouchBuffer.Get<FNNEMultiBinaryBuffer>().Buffer.Num(), 8);

    const TInstancedStruct<FNNEPointBuffer>& ActionsBuffer = AgentDict.Buffers["actions"];
    TestTrue(TEXT("ActionsBuffer is FNNEDictBuffer"), ActionsBuffer.GetPtr<FNNEDictBuffer>() != nullptr);
    
    const FNNEDictBuffer& ActionsDict = ActionsBuffer.Get<FNNEDictBuffer>();
    TestEqual(TEXT("ActionsDict has 2 keys"), ActionsDict.Buffers.Num(), 2);
    TestTrue(TEXT("ActionsDict contains 'movement'"), ActionsDict.Buffers.Contains("movement"));
    TestTrue(TEXT("ActionsDict contains 'interaction'"), ActionsDict.Buffers.Contains("interaction"));

    // Verify movement buffer (single selected direction)
    const TInstancedStruct<FNNEPointBuffer>& MovementBuffer = ActionsDict.Buffers["movement"];
	TestTrue(TEXT("MovementBuffer is FNNEDiscreteBuffer"), MovementBuffer.GetPtr<FNNEDiscreteBuffer>() != nullptr);
	TestEqual(TEXT("MovementBuffer has 1 element"), MovementBuffer.Get<FNNEDiscreteBuffer>().Buffer.Num(), 1);

    // Verify interaction buffer (2 binary elements)
    const TInstancedStruct<FNNEPointBuffer>& InteractionBuffer = ActionsDict.Buffers["interaction"];
	TestTrue(TEXT("InteractionBuffer is FNNEBinaryBuffer"), InteractionBuffer.GetPtr<FNNEMultiBinaryBuffer>() != nullptr);
    TestEqual(TEXT("InteractionBuffer has 2 elements"), InteractionBuffer.Get<FNNEMultiBinaryBuffer>().Buffer.Num(), 2);

    // All buffers should be initialized with zeros (condensed representation)
    TestEqualExactFloat(TEXT("Position[0] initialized to 0"), PositionBuffer.Get<FNNEBoxBuffer>().Buffer[0], 0.0f);
    TestEqualExactFloat(TEXT("Vision[0] initialized to 0"), VisionBuffer.Get<FNNEBoxBuffer>().Buffer[0], 0.0f);
    TestEqualExactFloat(TEXT("Vision[12287] initialized to 0"), VisionBuffer.Get<FNNEBoxBuffer>().Buffer[12287], 0.0f);
    TestEqual(TEXT("Audio[0] initialized to 0"), AudioBuffer.Get<FNNEMultiDiscreteBuffer>().Buffer[0], (int64)0);
	TestEqualExactFloat(TEXT("Touch[0] initialized to 0"), TouchBuffer.Get<FNNEMultiBinaryBuffer>().Buffer[0], 0.0f);
    TestEqual(TEXT("Movement[0] initialized to 0"), MovementBuffer.Get<FNNEDiscreteBuffer>().Buffer[0], (int64)0);
	TestEqualExactFloat(TEXT("Interaction[0] initialized to 0"), InteractionBuffer.Get<FNNEMultiBinaryBuffer>().Buffer[0], 0.0f);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNNEBufferAllocatorMultiDiscreteSpaceTest, "Schola.Policies.NNE.NNEBufferAllocator.DiscreteSpace Test", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FNNEBufferAllocatorMultiDiscreteSpaceTest::RunTest(const FString& Parameters)
{
    FMultiDiscreteSpace DiscreteSpace;
    DiscreteSpace.Add(2);
    DiscreteSpace.Add(3);

    TInstancedStruct<FNNEPointBuffer> NNEBuffer;
    FNNEBufferAllocator::AllocateBuffer(TInstancedStruct<FSpace>::Make<FMultiDiscreteSpace>(DiscreteSpace), NNEBuffer);

    TestTrue(TEXT("NNEBuffer is FNNEDiscreteBuffer"), NNEBuffer.GetPtr<FNNEMultiDiscreteBuffer>() != nullptr);
    
    const FNNEMultiDiscreteBuffer& DiscreteBuffer = NNEBuffer.Get<FNNEMultiDiscreteBuffer>();
    TestEqual(TEXT("DiscreteBuffer.Buffer.Num() == 2"), DiscreteBuffer.Buffer.Num(), 2);
    TestEqual(TEXT("DiscreteBuffer.Buffer[0] == 0"), DiscreteBuffer.Buffer[0], (int64)0);
    TestEqual(TEXT("DiscreteBuffer.Buffer[1] == 0"), DiscreteBuffer.Buffer[1], (int64)0);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNNEBufferAllocatorBinarySpaceTest, "Schola.Policies.NNE.NNEBufferAllocator.BinarySpace Test", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FNNEBufferAllocatorBinarySpaceTest::RunTest(const FString& Parameters)
{
    FMultiBinarySpace BinarySpace(3); // 3 dimensions

    TInstancedStruct<FNNEPointBuffer> NNEBuffer;
    FNNEBufferAllocator::AllocateBuffer(TInstancedStruct<FSpace>::Make<FMultiBinarySpace>(BinarySpace), NNEBuffer);

    TestTrue(TEXT("NNEBuffer is FNNEBinaryBuffer"), NNEBuffer.GetPtr<FNNEMultiBinaryBuffer>() != nullptr);
    
    const FNNEMultiBinaryBuffer& BinaryBuffer = NNEBuffer.Get<FNNEMultiBinaryBuffer>();
    TestEqual(TEXT("BinaryBuffer.Buffer.Num() == 3"), BinaryBuffer.Buffer.Num(), 3);
    TestEqualExactFloat(TEXT("BinaryBuffer.Buffer[0] == 0.0f"), BinaryBuffer.Buffer[0], 0.0f);
    TestEqualExactFloat(TEXT("BinaryBuffer.Buffer[1] == 0.0f"), BinaryBuffer.Buffer[1], 0.0f);
    TestEqualExactFloat(TEXT("BinaryBuffer.Buffer[2] == 0.0f"), BinaryBuffer.Buffer[2], 0.0f);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNNEBufferAllocatorSimpleDictSpaceTest, "Schola.Policies.NNE.NNEBufferAllocator.Simple DictSpace Test", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FNNEBufferAllocatorSimpleDictSpaceTest::RunTest(const FString& Parameters)
{
    FDictSpace DictSpace;
    FBoxSpace BoxSpace;
    BoxSpace.Add(-1.0, 1.0);
    DictSpace.Spaces.Add("box", TInstancedStruct<FSpace>::Make<FBoxSpace>(BoxSpace));

    TInstancedStruct<FNNEPointBuffer> NNEBuffer;
    FNNEBufferAllocator::AllocateBuffer(TInstancedStruct<FSpace>::Make<FDictSpace>(DictSpace), NNEBuffer);

    TestTrue(TEXT("NNEBuffer is FNNEDictBuffer"), NNEBuffer.GetPtr<FNNEDictBuffer>() != nullptr);
    
    const FNNEDictBuffer& DictBuffer = NNEBuffer.Get<FNNEDictBuffer>();
    TestEqual(TEXT("DictBuffer.Buffers.Num() == 1"), DictBuffer.Buffers.Num(), 1);
    TestTrue(TEXT("DictBuffer contains 'box' key"), DictBuffer.Buffers.Contains("box"));
    
    const TInstancedStruct<FNNEPointBuffer>& BoxBuffer = DictBuffer.Buffers["box"];
    TestTrue(TEXT("BoxBuffer is FNNEBoxBuffer"), BoxBuffer.GetPtr<FNNEBoxBuffer>() != nullptr);
    TestEqual(TEXT("BoxBuffer.Buffer.Num() == 1"), BoxBuffer.Get<FNNEBoxBuffer>().Buffer.Num(), 1);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNNEBufferAllocatorNestedDictSpaceTest, "Schola.Policies.NNE.NNEBufferAllocator.Nested DictSpace Test", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FNNEBufferAllocatorNestedDictSpaceTest::RunTest(const FString& Parameters)
{
    FDictSpace OuterDictSpace;
    FBoxSpace OuterBoxSpace;
    OuterBoxSpace.Add(-1.0, 1.0);
    OuterDictSpace.Spaces.Add("outer_box", TInstancedStruct<FSpace>::Make<FBoxSpace>(OuterBoxSpace));

    FDictSpace InnerDictSpace;
    FBoxSpace InnerBoxSpace;
    InnerBoxSpace.Add(-2.0, 2.0);
    InnerDictSpace.Spaces.Add("inner_box", TInstancedStruct<FSpace>::Make<FBoxSpace>(InnerBoxSpace));
    
    OuterDictSpace.Spaces.Add("inner_dict", TInstancedStruct<FSpace>::Make<FDictSpace>(InnerDictSpace));

    TInstancedStruct<FNNEPointBuffer> NNEBuffer;
    FNNEBufferAllocator::AllocateBuffer(TInstancedStruct<FSpace>::Make<FDictSpace>(OuterDictSpace), NNEBuffer);

    TestTrue(TEXT("NNEBuffer is FNNEDictBuffer"), NNEBuffer.GetPtr<FNNEDictBuffer>() != nullptr);
    
    const FNNEDictBuffer& OuterDictBuffer = NNEBuffer.Get<FNNEDictBuffer>();
    TestEqual(TEXT("OuterDictBuffer.Buffers.Num() == 2"), OuterDictBuffer.Buffers.Num(), 2);
    TestTrue(TEXT("OuterDictBuffer contains 'outer_box' key"), OuterDictBuffer.Buffers.Contains("outer_box"));
    TestTrue(TEXT("OuterDictBuffer contains 'inner_dict' key"), OuterDictBuffer.Buffers.Contains("inner_dict"));
    
    // Test outer box buffer
    const TInstancedStruct<FNNEPointBuffer>& OuterBoxBuffer = OuterDictBuffer.Buffers["outer_box"];
    TestTrue(TEXT("OuterBoxBuffer is FNNEBoxBuffer"), OuterBoxBuffer.GetPtr<FNNEBoxBuffer>() != nullptr);
    
    // Test inner dict buffer
    const TInstancedStruct<FNNEPointBuffer>& InnerDictBuffer = OuterDictBuffer.Buffers["inner_dict"];
    TestTrue(TEXT("InnerDictBuffer is FNNEDictBuffer"), InnerDictBuffer.GetPtr<FNNEDictBuffer>() != nullptr);
    
    const FNNEDictBuffer& InnerDict = InnerDictBuffer.Get<FNNEDictBuffer>();
    TestEqual(TEXT("InnerDict.Buffers.Num() == 1"), InnerDict.Buffers.Num(), 1);
    TestTrue(TEXT("InnerDict contains 'inner_box' key"), InnerDict.Buffers.Contains("inner_box"));

    return true;
}
