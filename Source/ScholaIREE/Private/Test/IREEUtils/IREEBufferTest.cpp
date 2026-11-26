// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "IREEUtils/IREEBuffer.h"
#include "IREEUtils/IREEBufferVisitor.h"

#if WITH_AUTOMATION_TESTS

#define TestEqualExactFloat(TestMessage, Actual, Expected) TestEqual(TestMessage, (float)Actual, (float)Expected, 0.0001f)

// Minimal visitor for core behavior checks
struct FIREETestBufferVisitor : public FIREEBufferVisitor
{
	const FIREEDiscreteBuffer*		DiscreteRef = nullptr;
	const FIREEMultiDiscreteBuffer* MultiDiscreteRef = nullptr;
	const FIREEBoxBuffer*			BoxRef = nullptr;
	const FIREEMultiBinaryBuffer*	BinaryRef = nullptr;
	const FIREEDictBuffer*			DictRef = nullptr;
	TArray<bool>					CopiedBoolData;
	TArray<float>					CopiedFloatData;
	TArray<int64>					CopiedIntData;
	TArray<int64>					CopiedMultiDiscreteData;

	void operator()(const FIREEDiscreteBuffer& InBuffer) override
	{
		DiscreteRef = &InBuffer;
		CopiedIntData = InBuffer.Buffer;
	}
	void operator()(const FIREEBoxBuffer& InBuffer) override
	{
		BoxRef = &InBuffer;
		CopiedFloatData = InBuffer.Buffer;
	}
	void operator()(const FIREEMultiBinaryBuffer& InBuffer) override
	{
		BinaryRef = &InBuffer;
		CopiedBoolData = InBuffer.Buffer;
	}
	void operator()(const FIREEDictBuffer& InBuffer) override
	{
		DictRef = &InBuffer;
	}

	void operator()(const FIREEMultiDiscreteBuffer& InBuffer) override
	{
		MultiDiscreteRef = &InBuffer;
		CopiedMultiDiscreteData = InBuffer.Buffer;
	}
};

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIREEStateBufferBasicsTest, "Schola.Policies.IREE.IREEBuffer.StateBuffer Basics", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FIREEStateBufferBasicsTest::RunTest(const FString& Parameters)
{
	// 3 sequences, dim=2, plus one pending row
	FIREEStateBuffer StateBuffer({ -1, -1, 2 }, 3);
	TestEqual(TEXT("Sequence plus pending row"), StateBuffer.StateBuffer.Num(), 8);

	// The input binding covers the sequence and the output binding covers only the pending row
	FIREETensorBinding InputBinding = StateBuffer.MakeInputBinding();
	TestEqual(TEXT("Input binding size"), InputBinding.SizeInBytes, (uint64)(3 * 2 * sizeof(float)));
	TestEqual(TEXT("Input binding data"), InputBinding.Data, (void*)StateBuffer.StateBuffer.GetData());
	FIREETensorBinding OutputBinding = StateBuffer.MakeOutputBinding();
	TestEqual(TEXT("Output binding size"), OutputBinding.SizeInBytes, (uint64)(2 * sizeof(float)));
	TestEqual(TEXT("Output binding starts after the input"), OutputBinding.Data, (void*)(StateBuffer.StateBuffer.GetData() + 6));

	// Rows [1,2] [3,4] [5,6], pending [7,8]. Update drops the first row and commits the pending row
	StateBuffer.StateBuffer = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
	StateBuffer.Update();
	TestEqualExactFloat(TEXT("After update: seq0_dim0 == 3"), StateBuffer.StateBuffer[0], 3.0f);
	TestEqualExactFloat(TEXT("After update: seq1_dim1 == 6"), StateBuffer.StateBuffer[3], 6.0f);
	TestEqualExactFloat(TEXT("After update: seq2_dim0 == 7"), StateBuffer.StateBuffer[4], 7.0f);
	TestEqualExactFloat(TEXT("After update: seq2_dim1 == 8"), StateBuffer.StateBuffer[5], 8.0f);

	// Without a sequence dimension, Update replaces the state with the pending one
	FIREEStateBuffer FlatBuffer({ -1, 2 });
	TestEqual(TEXT("Flat state plus pending state"), FlatBuffer.StateBuffer.Num(), 4);
	FlatBuffer.StateBuffer = { 1.0f, 2.0f, 9.0f, 10.0f };
	FlatBuffer.Update();
	TestEqualExactFloat(TEXT("Flat update: dim0 == 9"), FlatBuffer.StateBuffer[0], 9.0f);
	TestEqualExactFloat(TEXT("Flat update: dim1 == 10"), FlatBuffer.StateBuffer[1], 10.0f);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIREEStateLayoutBasicsTest, "Schola.Policies.IREE.IREEBuffer.StateLayout Basics", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FIREEStateLayoutBasicsTest::RunTest(const FString& Parameters)
{
	// Batch -1 and sequence -1, max sequence length 3, state vector of 2
	FIREEStateLayout Layout(TArray<int32> { -1, -1, 2 }, 3);
	TestEqual(TEXT("Batch fixed to 1 and sequence set to max length"), Layout.Shape, TArray<int> { 1, 3, 2 });
	TestEqual(TEXT("SeqDim"), Layout.SeqDim, 1);
	TestEqual(TEXT("StateDimSize"), Layout.StateDimSize, 2);
	TestEqual(TEXT("NumElements"), Layout.GetNumElements(), 6);
	TestEqual(TEXT("BufferNumElements includes the pending state"), Layout.GetBufferNumElements(), 8);

	FIREEStateBuffer Buffer(Layout);
	TestEqual(TEXT("Buffer allocated from layout"), Buffer.StateBuffer.Num(), 8);
	TestEqual(TEXT("Buffer copies the layout's StateDimSize"), Buffer.StateDimSize, 2);
	for (float Value : Buffer.StateBuffer)
	{
		TestEqual(TEXT("Buffer is zeroed"), Value, 0.0f);
	}

	FIREEStateLayout FlatLayout(TArray<int32> { -1, 4 });
	TestFalse(TEXT("No sequence dimension"), FlatLayout.HasSequenceDimension());
	TestEqual(TEXT("Flat StateDimSize is the whole tensor"), FlatLayout.StateDimSize, 4);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIREEDiscreteBufferBasicsTest, "Schola.Policies.IREE.IREEBuffer.DiscreteBuffer Basics", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FIREEDiscreteBufferBasicsTest::RunTest(const FString& Parameters)
{
	FIREEDiscreteBuffer Buffer;
	Buffer.Buffer[0] = 2;
	FIREETestBufferVisitor Visitor;
	Buffer.Accept(Visitor);
	TestTrue(TEXT("Visitor captured DiscreteBuffer"), Visitor.DiscreteRef != nullptr);
	TestEqual(TEXT("Visitor copied size"), Visitor.CopiedIntData.Num(), 1);
	TestEqual(TEXT("Visitor copy matches [0]"), Visitor.CopiedIntData[0], (int64)2);

	// Binding
	FIREETensorBinding Binding = Buffer.MakeBinding();
	TestEqual(TEXT("Binding size"), Binding.SizeInBytes, (uint64)sizeof(int64));
	TestEqual(TEXT("Binding data ptr"), Binding.Data, (void*)Buffer.Buffer.GetData());
	int64* BindingData = static_cast<int64*>(Binding.Data);
	TestEqual(TEXT("Binding data mirrors buffer [0]"), BindingData[0], (int64)2);

	// Copy independence
	FIREEDiscreteBuffer Copy = Buffer;
	Copy.Buffer[0] = 9;
	TestEqual(TEXT("Copy mutated [0]"), Copy.Buffer[0], (int64)9);
	TestEqual(TEXT("Original unaffected [0]"), Buffer.Buffer[0], (int64)2);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIREEBoxAndBinaryBufferTest, "Schola.Policies.IREE.IREEBuffer.BoxAndBinary Basics", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FIREEBoxAndBinaryBufferTest::RunTest(const FString& Parameters)
{
	// Box
	{
		FIREEBoxBuffer Box(5);
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

		FIREETensorBinding Binding = Box.MakeBinding();
		TestEqual(TEXT("Box binding size"), Binding.SizeInBytes, (uint64)(5 * sizeof(float)));
		TestEqual(TEXT("Box binding ptr"), Binding.Data, (void*)Box.Buffer.GetData());
	}

	// Binary
	{
		FIREEMultiBinaryBuffer Binary(4);
		Binary.Buffer[0] = true;
		Binary.Buffer[1] = false;
		Binary.Buffer[2] = true;
		Binary.Buffer[3] = false;

		FIREETensorBinding Binding = Binary.MakeBinding();
		TestEqual(TEXT("Binary binding size"), Binding.SizeInBytes, (uint64)(4 * sizeof(bool)));
		TestEqual(TEXT("Binary binding ptr"), Binding.Data, (void*)Binary.Buffer.GetData());
		TestEqual(TEXT("Binary[0]"), Binary.Buffer[0], true);
		TestEqual(TEXT("Binary[1]"), Binary.Buffer[1], false);
		TestEqual(TEXT("Binary[2]"), Binary.Buffer[2], true);
		TestEqual(TEXT("Binary[3]"), Binary.Buffer[3], false);
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIREEMultiDiscreteBufferBasicsTest, "Schola.Policies.IREE.IREEBuffer.MultiDiscreteBuffer Basics", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FIREEMultiDiscreteBufferBasicsTest::RunTest(const FString& Parameters)
{
	FIREEMultiDiscreteBuffer Buffer(2);
	Buffer.Buffer[0] = 1;
	Buffer.Buffer[1] = 2;

	FIREETestBufferVisitor Visitor;
	Buffer.Accept(Visitor);
	TestTrue(TEXT("Visitor captured MultiDiscreteBuffer"), Visitor.MultiDiscreteRef != nullptr);
	TestEqual(TEXT("Visitor copied size"), Visitor.CopiedMultiDiscreteData.Num(), 2);
	TestEqual(TEXT("Visitor copy matches [0]"), Visitor.CopiedMultiDiscreteData[0], (int64)1);
	TestEqual(TEXT("Visitor copy matches [1]"), Visitor.CopiedMultiDiscreteData[1], (int64)2);

	FIREETensorBinding Binding = Buffer.MakeBinding();
	TestEqual(TEXT("Binding size"), Binding.SizeInBytes, (uint64)(2 * sizeof(int64)));
	TestEqual(TEXT("Binding data ptr"), Binding.Data, (void*)Buffer.Buffer.GetData());
	int64* BindingData = static_cast<int64*>(Binding.Data);
	TestEqual(TEXT("Binding data mirrors buffer [0]"), BindingData[0], (int64)1);
	TestEqual(TEXT("Binding data mirrors buffer [1]"), BindingData[1], (int64)2);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIREEDictBufferBasicTest, "Schola.Policies.IREE.IREEBuffer.DictBuffer Basics", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FIREEDictBufferBasicTest::RunTest(const FString& Parameters)
{
	TInstancedStruct<FIREEPointBuffer> Pos;
	Pos.InitializeAs<FIREEBoxBuffer>(3);
	Pos.GetMutable<FIREEBoxBuffer>().Buffer[0] = 1.0f;
	Pos.GetMutable<FIREEBoxBuffer>().Buffer[1] = 2.0f;
	Pos.GetMutable<FIREEBoxBuffer>().Buffer[2] = 3.0f;

	TInstancedStruct<FIREEPointBuffer> Act;
	Act.InitializeAs<FIREEDiscreteBuffer>();
	Act.GetMutable<FIREEDiscreteBuffer>().Buffer[0] = 2;
	TMap<FString, TInstancedStruct<FIREEPointBuffer>> Map;
	Map.Add("position", Pos);
	Map.Add("action", Act);

	FIREEDictBuffer Dict(Map);
	TestEqual(TEXT("Entry count"), Dict.Buffers.Num(), 2);
	TestTrue(TEXT("Has position"), Dict.Buffers.Contains("position"));
	TestTrue(TEXT("Has action"), Dict.Buffers.Contains("action"));

	const FIREEBoxBuffer& PosRef = Dict.Buffers["position"].Get<FIREEBoxBuffer>();
	TestEqualExactFloat(TEXT("Pos[0]"), PosRef.Buffer[0], 1.0f);
	TestEqualExactFloat(TEXT("Pos[1]"), PosRef.Buffer[1], 2.0f);
	TestEqualExactFloat(TEXT("Pos[2]"), PosRef.Buffer[2], 3.0f);

	const FIREEDiscreteBuffer& ActRef = Dict.Buffers["action"].Get<FIREEDiscreteBuffer>();
	TestEqualExactFloat(TEXT("Act[0]"), ActRef.Buffer[0], 2);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIREEDiscreteBufferInstancedStructTest, "Schola.Policies.IREE.IREEBuffer.DiscreteBuffer InstancedStruct", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FIREEDiscreteBufferInstancedStructTest::RunTest(const FString& Parameters)
{
	TInstancedStruct<FIREEPointBuffer> BufferInstance;
	BufferInstance.InitializeAs<FIREEDiscreteBuffer>();

	FIREEDiscreteBuffer& Discrete = BufferInstance.GetMutable<FIREEDiscreteBuffer>();
	Discrete.Buffer[0] = 5;

	FIREETestBufferVisitor Visitor;
	BufferInstance.Get<FIREEPointBuffer>().Accept(Visitor);
	TestTrue(TEXT("Visitor got Discrete"), Visitor.DiscreteRef != nullptr);
	TestEqual(TEXT("Visitor size"), Visitor.DiscreteRef->Buffer.Num(), 1);
	TestEqualExactFloat(TEXT("Visitor[0]"), Visitor.DiscreteRef->Buffer[0], 5);

	const FIREEDiscreteBuffer& ConstBuffer = BufferInstance.Get<FIREEDiscreteBuffer>();
	FIREETensorBinding	   Binding = ConstBuffer.MakeBinding();
	TestEqual(TEXT("Binding ptr"), Binding.Data, (void*)ConstBuffer.Buffer.GetData());
	TestEqual(TEXT("Binding size"), Binding.SizeInBytes, (uint64)sizeof(int64));
	return true;
}

#undef TestEqualExactFloat

#endif // WITH_AUTOMATION_TESTS
