// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "IREEUtils/IREEBindingCreator.h"
#include "IREEUtils/IREEBuffer.h"
#include "Spaces/BoxSpace.h"

#if WITH_AUTOMATION_TESTS

namespace
{
	FScholaIREETensorInfo MakeTestDesc(const FString& Name, const TArray<int32>& Shape, const FString& Dtype = TEXT("float32"))
	{
		FScholaIREETensorInfo Desc;
		Desc.Name = Name;
		Desc.Shape = Shape;
		Desc.Dtype = Dtype;
		return Desc;
	}
} // namespace

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIREEBindingCreatorConstructorTest, "Schola.Policies.IREE.IREEBindingCreator.Constructor Test", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FIREEBindingCreatorConstructorTest::RunTest(const FString& Parameters)
{
	TInstancedStruct<FIREEPointBuffer> TestBuffer;
	TestBuffer.InitializeAs<FIREEBoxBuffer>(3);

	TArray<FScholaIREETensorInfo> TensorDescs = { MakeTestDesc(TEXT("input"), { 3 }) };
	TArray<FIREETensorBinding> Bindings;
	Bindings.SetNum(1);

	FIREEBindingCreator Creator(TestBuffer, TensorDescs, Bindings);
	TestEqual(TEXT("bError should be false initially"), Creator.bError, false);
	TestEqual(TEXT("Index should be 0 initially"), Creator.Index, 0);
	TestEqual(TEXT("TensorDescs should be stored"), Creator.TensorDescs.Num(), 1);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIREEBindingCreatorBoxBufferTest, "Schola.Policies.IREE.IREEBindingCreator.BoxBuffer Test", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FIREEBindingCreatorBoxBufferTest::RunTest(const FString& Parameters)
{
	TInstancedStruct<FIREEPointBuffer> Holder;
	Holder.InitializeAs<FIREEBoxBuffer>(3);
	TArray<FScholaIREETensorInfo> TensorDescs = { MakeTestDesc(TEXT("input"), { 3 }) };
	TArray<FIREETensorBinding> Bindings;
	Bindings.SetNum(1);
	FIREEBindingCreator Creator(Holder, TensorDescs, Bindings);

	FIREEBoxBuffer BoxBuffer(3);
	Creator(BoxBuffer);

	TestEqual(TEXT("Binding size"), Bindings[0].SizeInBytes, (uint64)(3 * sizeof(float)));
	TestEqual(TEXT("Binding data ptr"), Bindings[0].Data, (void*)BoxBuffer.Buffer.GetData());
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIREEBindingCreatorBinaryBufferTest, "Schola.Policies.IREE.IREEBindingCreator.BinaryBuffer Test", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FIREEBindingCreatorBinaryBufferTest::RunTest(const FString& Parameters)
{
	TInstancedStruct<FIREEPointBuffer> Holder;
	Holder.InitializeAs<FIREEMultiBinaryBuffer>(2);
	TArray<FScholaIREETensorInfo> TensorDescs = { MakeTestDesc(TEXT("binary_input"), { 2 }) };
	TArray<FIREETensorBinding> Bindings;
	Bindings.SetNum(1);
	FIREEBindingCreator Creator(Holder, TensorDescs, Bindings);

	FIREEMultiBinaryBuffer BinaryBuffer(2);
	Creator(BinaryBuffer);

	TestEqual(TEXT("Binding size"), Bindings[0].SizeInBytes, (uint64)(2 * sizeof(bool)));
	TestEqual(TEXT("Binding data ptr"), Bindings[0].Data, (void*)BinaryBuffer.Buffer.GetData());
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIREEBindingCreatorDiscreteBufferTest, "Schola.Policies.IREE.IREEBindingCreator.DiscreteBuffer Test", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FIREEBindingCreatorDiscreteBufferTest::RunTest(const FString& Parameters)
{
	TInstancedStruct<FIREEPointBuffer> Holder;
	Holder.InitializeAs<FIREEDiscreteBuffer>();
	TArray<FScholaIREETensorInfo> TensorDescs = { MakeTestDesc(TEXT("discrete_input"), { 1 }, TEXT("int64")) };
	TArray<FIREETensorBinding> Bindings;
	Bindings.SetNum(1);
	FIREEBindingCreator Creator(Holder, TensorDescs, Bindings);

	FIREEDiscreteBuffer Buffer;
	Creator(Buffer);

	TestEqual(TEXT("Binding size"), Bindings[0].SizeInBytes, (uint64)sizeof(int64));
	TestEqual(TEXT("Binding data ptr"), Bindings[0].Data, (void*)Buffer.Buffer.GetData());
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIREEBindingCreatorDictSkipsStateTensorsTest, "Schola.Policies.IREE.IREEBindingCreator.Dict skips state tensors", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FIREEBindingCreatorDictSkipsStateTensorsTest::RunTest(const FString& Parameters)
{
	TMap<FString, TInstancedStruct<FIREEPointBuffer>> Map;
	TInstancedStruct<FIREEPointBuffer>				  Regular;
	Regular.InitializeAs<FIREEBoxBuffer>(2);
	Map.Add(TEXT("regular_input"), Regular);

	TInstancedStruct<FIREEPointBuffer> Root;
	Root.InitializeAs<FIREEDictBuffer>(FIREEDictBuffer(Map));

	TArray<FScholaIREETensorInfo> TensorDescs = {
		MakeTestDesc(TEXT("state_in_hidden"), { 64 }),
		MakeTestDesc(TEXT("regular_input"), { 2 }),
		MakeTestDesc(TEXT("state_out_cell"), { 64 }),
	};

	TArray<FIREETensorBinding> Bindings;
	Bindings.SetNum(3);
	FIREEBindingCreator Creator(Root, TensorDescs, Bindings);

	FIREEDictBuffer Dict(Map);
	Creator(Dict);

	TestEqual(TEXT("Should not error"), Creator.bError, false);
	TestEqual(TEXT("Processed all descs"), Creator.Index, 3);
	TestEqual(TEXT("State input binding untouched"), Bindings[0].SizeInBytes, (uint64)0);
	TestEqual(TEXT("Middle binding size"), Bindings[1].SizeInBytes, (uint64)(2 * sizeof(float)));
	TestEqual(TEXT("State output binding untouched"), Bindings[2].SizeInBytes, (uint64)0);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIREEBindingCreatorMissingKeyTest, "Schola.Policies.IREE.IREEBindingCreator.Missing key error", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FIREEBindingCreatorMissingKeyTest::RunTest(const FString& Parameters)
{
	TMap<FString, TInstancedStruct<FIREEPointBuffer>> Map;
	TInstancedStruct<FIREEPointBuffer>				  Buf;
	Buf.InitializeAs<FIREEBoxBuffer>(2);
	Map.Add(TEXT("existing_key"), Buf);

	TInstancedStruct<FIREEPointBuffer> Root;
	Root.InitializeAs<FIREEDictBuffer>(FIREEDictBuffer(Map));

	TArray<FScholaIREETensorInfo>	TensorDescs = { MakeTestDesc(TEXT("missing_key"), { 2 }) };
	TArray<FIREETensorBinding> Bindings;
	Bindings.SetNum(1);
	FIREEBindingCreator Creator(Root, TensorDescs, Bindings);

	FIREEDictBuffer Dict(Map);
	AddExpectedError(TEXT("key missing_key not found in dict buffer"), EAutomationExpectedErrorFlags::Contains, 1);
	Creator(Dict);
	TestEqual(TEXT("Should error on missing key"), Creator.bError, true);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIREEBindingCreatorOutOfBoundsTest, "Schola.Policies.IREE.IREEBindingCreator.Out of bounds bindings", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FIREEBindingCreatorOutOfBoundsTest::RunTest(const FString& Parameters)
{
	// Dict with a single buffer but two tensor descs; bindings array too small
	AddExpectedError(TEXT("binding index 1 out of range"), EAutomationExpectedErrorFlags::Contains, 1);
	TMap<FString, TInstancedStruct<FIREEPointBuffer>> Map;
	TInstancedStruct<FIREEPointBuffer>				  Buf;
	Buf.InitializeAs<FIREEBoxBuffer>(2);
	Map.Add(TEXT("input"), Buf);
	TInstancedStruct<FIREEPointBuffer> Root;
	Root.InitializeAs<FIREEDictBuffer>(FIREEDictBuffer(Map));

	TArray<FScholaIREETensorInfo> TensorDescs = {
		MakeTestDesc(TEXT("input"), { 2 }),
		MakeTestDesc(TEXT("input2"), { 2 }),
	};

	TArray<FIREETensorBinding> Bindings;
	Bindings.SetNum(1);
	FIREEBindingCreator Creator(Root, TensorDescs, Bindings);
	FIREEDictBuffer		Dict(Map);
	Creator(Dict);
	TestEqual(TEXT("Creator should be marked as having an error"), Creator.bError, true);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIREEBindingCreatorCreateBindingsSmokeTest, "Schola.Policies.IREE.IREEBindingCreator.CreateBindings Smoke", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FIREEBindingCreatorCreateBindingsSmokeTest::RunTest(const FString& Parameters)
{
	TInstancedStruct<FSpace> Space;
	Space.InitializeAs<FBoxSpace>();

	TInstancedStruct<FIREEPointBuffer> Buffer;
	Buffer.InitializeAs<FIREEBoxBuffer>(3);

	TArray<FScholaIREETensorInfo>	TensorDescs = { MakeTestDesc(TEXT("input"), { 3 }) };
	TArray<FIREETensorBinding> Bindings;
	Bindings.SetNum(1);
	const bool bError = FIREEBindingCreator::CreateBindings(Space, Buffer, TensorDescs, Bindings);

	TestEqual(TEXT("Should not report error"), bError, false);
	TestEqual(TEXT("Binding size"), Bindings[0].SizeInBytes, (uint64)(3 * sizeof(float)));
	TestEqual(TEXT("Binding data"), Bindings[0].Data, (void*)Buffer.Get<FIREEBoxBuffer>().Buffer.GetData());
	return true;
}

#endif // WITH_AUTOMATION_TESTS
