// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "NNEUtils/NNEBuffer.h"
#include "NNEUtils/NNEBufferVisitor.h"

#define TestEqualExactFloat(TestMessage, Actual, Expected) TestEqual(TestMessage, (float)Actual, (float)Expected, 0.0001f)

// Minimal visitor for core behavior checks
struct FTestBufferVisitor : public FNNEBufferVisitor
{
	const FNNEDiscreteBuffer*	   DiscreteRef = nullptr;
	const FNNEMultiDiscreteBuffer* MultiDiscreteRef = nullptr;
	const FNNEBoxBuffer*		   BoxRef = nullptr;
	const FNNEMultiBinaryBuffer*   BinaryRef = nullptr;
	const FNNEDictBuffer*		   DictRef = nullptr;
	TArray<bool>				   CopiedBoolData;
	TArray<float>				   CopiedFloatData;
	TArray<int64>				   CopiedIntData;
	TArray<int64>				   CopiedMultiDiscreteData;

	void operator()(const FNNEDiscreteBuffer& InBuffer) override
	{
		DiscreteRef = &InBuffer;
		CopiedIntData = InBuffer.Buffer;
	}
	void operator()(const FNNEBoxBuffer& InBuffer) override
	{
		BoxRef = &InBuffer;
		CopiedFloatData = InBuffer.Buffer;
	}
	void operator()(const FNNEMultiBinaryBuffer& InBuffer) override
	{
		BinaryRef = &InBuffer;
		CopiedBoolData = InBuffer.Buffer;
	}
	void operator()(const FNNEDictBuffer& InBuffer) override
	{
		DictRef = &InBuffer;
	}

	void operator()(const FNNEMultiDiscreteBuffer& InBuffer) override
	{
		MultiDiscreteRef = &InBuffer;
		CopiedMultiDiscreteData = InBuffer.Buffer;
	}
};

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNNEStateBufferBasicsTest, "Schola.Policies.NNE.NNEBuffer.StateBuffer Basics", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FNNEStateBufferBasicsTest::RunTest(const FString& Parameters)
{
	// 3 sequences, dim=2
	FNNEStateBuffer StateBuffer({ -1, -1, 2 }, 3);

	// Fill with rows: [1,2] [3,4] [5,6]
	StateBuffer.StateBuffer[0] = 1.0f;
	StateBuffer.StateBuffer[1] = 2.0f;
	StateBuffer.StateBuffer[2] = 3.0f;
	StateBuffer.StateBuffer[3] = 4.0f;
	StateBuffer.StateBuffer[4] = 5.0f;
	StateBuffer.StateBuffer[5] = 6.0f;

	// Shift should drop first row and repeat last row
	StateBuffer.Shift();
	TestEqualExactFloat(TEXT("After shift: seq0_dim0 == 3"), StateBuffer.StateBuffer[0], 3.0f);
	TestEqualExactFloat(TEXT("After shift: seq0_dim1 == 4"), StateBuffer.StateBuffer[1], 4.0f);
	TestEqualExactFloat(TEXT("After shift: seq1_dim0 == 5"), StateBuffer.StateBuffer[2], 5.0f);
	TestEqualExactFloat(TEXT("After shift: seq1_dim1 == 6"), StateBuffer.StateBuffer[3], 6.0f);
	TestEqualExactFloat(TEXT("After shift: seq2_dim0 unchanged == 5"), StateBuffer.StateBuffer[4], 5.0f);
	TestEqualExactFloat(TEXT("After shift: seq2_dim1 unchanged == 6"), StateBuffer.StateBuffer[5], 6.0f);

	// Bindings
	UE::NNE::FTensorBindingCPU InputBinding = StateBuffer.MakeInputBinding();
	TestEqual(TEXT("Input binding size"), InputBinding.SizeInBytes, 3 * 2 * sizeof(float));
	TestEqual(TEXT("Input binding data"), InputBinding.Data, (void*)StateBuffer.StateBuffer.GetData());

	// Update should drop the first row, then SetNewestState writes the new state into the last row
	StateBuffer.StateBuffer = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
	StateBuffer.Update();
	StateBuffer.SetNewestState(TArray<float> { 7.0f, 8.0f });
	TestEqualExactFloat(TEXT("After update: seq0_dim0 == 3"), StateBuffer.StateBuffer[0], 3.0f);
	TestEqualExactFloat(TEXT("After update: seq1_dim1 == 6"), StateBuffer.StateBuffer[3], 6.0f);
	TestEqualExactFloat(TEXT("After update: seq2_dim0 == 7"), StateBuffer.StateBuffer[4], 7.0f);
	TestEqualExactFloat(TEXT("After update: seq2_dim1 == 8"), StateBuffer.StateBuffer[5], 8.0f);

	// Without a sequence dimension, Update leaves the buffer alone and SetNewestState replaces it
	FNNEStateBuffer FlatBuffer({ -1, 2 });
	FlatBuffer.StateBuffer = { 1.0f, 2.0f };
	FlatBuffer.Update();
	TestEqualExactFloat(TEXT("Flat update: unchanged"), FlatBuffer.StateBuffer[1], 2.0f);
	FlatBuffer.SetNewestState(TArray<float> { 9.0f, 10.0f });
	TestEqualExactFloat(TEXT("Flat set: dim0 == 9"), FlatBuffer.StateBuffer[0], 9.0f);
	TestEqualExactFloat(TEXT("Flat set: dim1 == 10"), FlatBuffer.StateBuffer[1], 10.0f);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNNEDiscreteBufferBasicsTest, "Schola.Policies.NNE.NNEBuffer.DiscreteBuffer Basics", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FNNEDiscreteBufferBasicsTest::RunTest(const FString& Parameters)
{
	FNNEDiscreteBuffer Buffer;
	Buffer.Buffer[0] = 2;
	FTestBufferVisitor Visitor;
	Buffer.Accept(Visitor);
	TestTrue(TEXT("Visitor captured DiscreteBuffer"), Visitor.DiscreteRef != nullptr);
	TestEqual(TEXT("Visitor copied size"), Visitor.CopiedIntData.Num(), 1);
	TestEqual(TEXT("Visitor copy matches [0]"), Visitor.CopiedIntData[0], (int64)2);

	// Binding
	UE::NNE::FTensorBindingCPU Binding = Buffer.MakeBinding();
	TestEqual(TEXT("Binding size"), Binding.SizeInBytes, sizeof(int64));
	TestEqual(TEXT("Binding data ptr"), Binding.Data, (void*)Buffer.Buffer.GetData());
	int64* BindingData = static_cast<int64*>(Binding.Data);
	TestEqual(TEXT("Binding data mirrors buffer [0]"), BindingData[0], (int64)2);

	// Copy independence
	FNNEDiscreteBuffer Copy = Buffer;
	Copy.Buffer[0] = 9;
	TestEqual(TEXT("Copy mutated [0]"), Copy.Buffer[0], (int64)9);
	TestEqual(TEXT("Original unaffected [0]"), Buffer.Buffer[0], (int64)2);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNNEBoxAndBinaryBufferTest, "Schola.Policies.NNE.NNEBuffer.BoxAndBinary Basics", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FNNEBoxAndBinaryBufferTest::RunTest(const FString& Parameters)
{
	// Box
	{
		FNNEBoxBuffer Box(5);
		for (int i = 0; i < 5; ++i)
		{
			TestEqualExactFloat(TEXT("Box init == 0"), Box.Buffer[i], 0.0f);
		}
		Box.Buffer[0] = 1.5f;
		Box.Buffer[2] = -2.7f;
		Box.Buffer[4] = 3.14f;
		TestEqualExactFloat(TEXT("Box[0]"), Box.Buffer[0], 1.5f);
		TestEqualExactFloat(TEXT("Box[2]"), Box.Buffer[2], -2.7f);
		TestEqualExactFloat(TEXT("Box[4]"), Box.Buffer[4], 3.14f);

		UE::NNE::FTensorBindingCPU Binding = Box.MakeBinding();
		TestEqual(TEXT("Box binding size"), Binding.SizeInBytes, 5 * sizeof(float));
		TestEqual(TEXT("Box binding ptr"), Binding.Data, (void*)Box.Buffer.GetData());
	}

	// Binary
	{
		FNNEMultiBinaryBuffer Binary(4);
		Binary.Buffer[0] = true;
		Binary.Buffer[1] = false;
		Binary.Buffer[2] = true;
		Binary.Buffer[3] = false;

		UE::NNE::FTensorBindingCPU Binding = Binary.MakeBinding();
		TestEqual(TEXT("Binary binding size"), Binding.SizeInBytes, 4 * sizeof(bool));
		TestEqual(TEXT("Binary binding ptr"), Binding.Data, (void*)Binary.Buffer.GetData());
		TestEqual(TEXT("Binary[0]"), Binary.Buffer[0], true);
		TestEqual(TEXT("Binary[1]"), Binary.Buffer[1], false);
		TestEqual(TEXT("Binary[2]"), Binary.Buffer[2], true);
		TestEqual(TEXT("Binary[3]"), Binary.Buffer[3], false);
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNNEMultiDiscreteBufferBasicsTest, "Schola.Policies.NNE.NNEBuffer.MultiDiscreteBuffer Basics", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FNNEMultiDiscreteBufferBasicsTest::RunTest(const FString& Parameters)
{
	FNNEMultiDiscreteBuffer Buffer(2);
	Buffer.Buffer[0] = 1;
	Buffer.Buffer[1] = 2;

	FTestBufferVisitor Visitor;
	Buffer.Accept(Visitor);
	TestTrue(TEXT("Visitor captured MultiDiscreteBuffer"), Visitor.MultiDiscreteRef != nullptr);
	TestEqual(TEXT("Visitor copied size"), Visitor.CopiedMultiDiscreteData.Num(), 2);
	TestEqual(TEXT("Visitor copy matches [0]"), Visitor.CopiedMultiDiscreteData[0], (int64)1);
	TestEqual(TEXT("Visitor copy matches [1]"), Visitor.CopiedMultiDiscreteData[1], (int64)2);

	UE::NNE::FTensorBindingCPU Binding = Buffer.MakeBinding();
	TestEqual(TEXT("Binding size"), Binding.SizeInBytes, 2 * sizeof(int64));
	TestEqual(TEXT("Binding data ptr"), Binding.Data, (void*)Buffer.Buffer.GetData());
	int64* BindingData = static_cast<int64*>(Binding.Data);
	TestEqual(TEXT("Binding data mirrors buffer [0]"), BindingData[0], (int64)1);
	TestEqual(TEXT("Binding data mirrors buffer [1]"), BindingData[1], (int64)2);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNNEDictBufferBasicTest, "Schola.Policies.NNE.NNEBuffer.DictBuffer Basics", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FNNEDictBufferBasicTest::RunTest(const FString& Parameters)
{
	TInstancedStruct<FNNEPointBuffer> Pos;
	Pos.InitializeAs<FNNEBoxBuffer>(3);
	Pos.GetMutable<FNNEBoxBuffer>().Buffer[0] = 1.0f;
	Pos.GetMutable<FNNEBoxBuffer>().Buffer[1] = 2.0f;
	Pos.GetMutable<FNNEBoxBuffer>().Buffer[2] = 3.0f;

	TInstancedStruct<FNNEPointBuffer> Act;
	Act.InitializeAs<FNNEDiscreteBuffer>();
	Act.GetMutable<FNNEDiscreteBuffer>().Buffer[0] = 2;
	TMap<FString, TInstancedStruct<FNNEPointBuffer>> Map;
	Map.Add("position", Pos);
	Map.Add("action", Act);

	FNNEDictBuffer Dict(Map);
	TestEqual(TEXT("Entry count"), Dict.Buffers.Num(), 2);
	TestTrue(TEXT("Has position"), Dict.Buffers.Contains("position"));
	TestTrue(TEXT("Has action"), Dict.Buffers.Contains("action"));

	const FNNEBoxBuffer& PosRef = Dict.Buffers["position"].Get<FNNEBoxBuffer>();
	TestEqualExactFloat(TEXT("Pos[0]"), PosRef.Buffer[0], 1.0f);
	TestEqualExactFloat(TEXT("Pos[1]"), PosRef.Buffer[1], 2.0f);
	TestEqualExactFloat(TEXT("Pos[2]"), PosRef.Buffer[2], 3.0f);

	const FNNEDiscreteBuffer& ActRef = Dict.Buffers["action"].Get<FNNEDiscreteBuffer>();
	TestEqualExactFloat(TEXT("Act[0]"), ActRef.Buffer[0], 2);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNNEDiscreteBufferInstancedStructTest, "Schola.Policies.NNE.NNEBuffer.DiscreteBuffer InstancedStruct", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FNNEDiscreteBufferInstancedStructTest::RunTest(const FString& Parameters)
{
	TInstancedStruct<FNNEPointBuffer> BufferInstance;
	BufferInstance.InitializeAs<FNNEDiscreteBuffer>();

	FNNEDiscreteBuffer& Discrete = BufferInstance.GetMutable<FNNEDiscreteBuffer>();
	Discrete.Buffer[0] = 5;

	FTestBufferVisitor Visitor;
	BufferInstance.Get<FNNEPointBuffer>().Accept(Visitor);
	TestTrue(TEXT("Visitor got Discrete"), Visitor.DiscreteRef != nullptr);
	TestEqual(TEXT("Visitor size"), Visitor.DiscreteRef->Buffer.Num(), 1);
	TestEqualExactFloat(TEXT("Visitor[0]"), Visitor.DiscreteRef->Buffer[0], 5);

	const FNNEDiscreteBuffer&  ConstBuffer = BufferInstance.Get<FNNEDiscreteBuffer>();
	UE::NNE::FTensorBindingCPU Binding = ConstBuffer.MakeBinding();
	TestEqual(TEXT("Binding ptr"), Binding.Data, (void*)ConstBuffer.Buffer.GetData());
	TestEqual(TEXT("Binding size"), Binding.SizeInBytes, sizeof(int64));
	return true;
}