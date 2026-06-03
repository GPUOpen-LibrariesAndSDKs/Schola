// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "ProtobufUtils/ProtobufSerializer.h"

#include "Spaces/DictSpace.h"
#include "Spaces/MultiBinarySpace.h"
#include "Spaces/DiscreteSpace.h"
#include "Spaces/MultiDiscreteSpace.h"
#include "Spaces/BoxSpace.h"
#include "Spaces/TextSpace.h"

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufDiscreteSpaceSerializationTest, "Schola.Protobuf.Serialization.Spaces.Discrete", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufDiscreteSpaceSerializationTest::RunTest(const FString& Parameters)
{
	TInstancedStruct<FSpace> Inst;
	Inst.InitializeAs<FDiscreteSpace>();
	FDiscreteSpace* Space = Inst.GetMutablePtr<FDiscreteSpace>();
	Space->High = 7;

	Schola::Space OutProto;
	ProtobufSerializer::ToProto(Inst, &OutProto);

	TestTrue(TEXT("Discrete space serialized as discrete_space"), OutProto.has_discrete_space());
	if (OutProto.has_discrete_space())
	{
		TestEqual(TEXT("DiscreteSpace.high == 7"), OutProto.discrete_space().high(), 7);
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufMultiDiscreteSpaceSerializationTest, "Schola.Protobuf.Serialization.Spaces.MultiDiscrete", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufMultiDiscreteSpaceSerializationTest::RunTest(const FString& Parameters)
{
	TInstancedStruct<FSpace> Inst;
	Inst.InitializeAs<FMultiDiscreteSpace>();
	FMultiDiscreteSpace* Space = Inst.GetMutablePtr<FMultiDiscreteSpace>();
	Space->High = TArray<int>({3,4,5});

	Schola::Space OutProto;
	ProtobufSerializer::ToProto(Inst, &OutProto);

	TestTrue(TEXT("MultiDiscrete space serialized as multi_discrete_space"), OutProto.has_multi_discrete_space());
	if (OutProto.has_multi_discrete_space())
	{
		auto& Repeated = OutProto.multi_discrete_space().high();
		TestEqual(TEXT("MultiDiscrete high size == 3"), (int)Repeated.size(), 3);
		TestEqual(TEXT("High[0] == 3"), Repeated.Get(0), 3);
		TestEqual(TEXT("High[1] == 4"), Repeated.Get(1), 4);
		TestEqual(TEXT("High[2] == 5"), Repeated.Get(2), 5);
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufMultiBinarySpaceSerializationTest, "Schola.Protobuf.Serialization.Spaces.MultiBinary", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufMultiBinarySpaceSerializationTest::RunTest(const FString& Parameters)
{
	TInstancedStruct<FSpace> Inst;
	Inst.InitializeAs<FMultiBinarySpace>();
	FMultiBinarySpace* Space = Inst.GetMutablePtr<FMultiBinarySpace>();
	Space->Shape = 8;

	Schola::Space OutProto;
	ProtobufSerializer::ToProto(Inst, &OutProto);

	TestTrue(TEXT("MultiBinary space serialized as multi_binary_space"), OutProto.has_multi_binary_space());
	if (OutProto.has_multi_binary_space())
	{
		TestEqual(TEXT("MultiBinary.shape == 8"), OutProto.multi_binary_space().shape(), 8);
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufBoxSpaceSerializationTest, "Schola.Protobuf.Serialization.Spaces.Box", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufBoxSpaceSerializationTest::RunTest(const FString& Parameters)
{
	TInstancedStruct<FSpace> Inst;
	Inst.InitializeAs<FBoxSpace>();
	FBoxSpace* Space = Inst.GetMutablePtr<FBoxSpace>();

	// Add two dimensions
	FBoxSpaceDimension DimA;
	DimA.Low = -1.0f;
	DimA.High = 1.0f;
	FBoxSpaceDimension DimB;
	DimB.Low = 0.0f;
	DimB.High = 2.0f;
	Space->Dimensions = TArray<FBoxSpaceDimension>({DimA, DimB});
	Space->Shape = TArray<int>({2});

	Schola::Space OutProto;
	ProtobufSerializer::ToProto(Inst, &OutProto);

	TestTrue(TEXT("Box space serialized as box_space"), OutProto.has_box_space());
	if (OutProto.has_box_space())
	{
		auto& Dims = OutProto.box_space().dimensions();
		TestEqual(TEXT("BoxSpace has 2 dims"), (int)Dims.size(), 2);
		TestTrue(TEXT("Dimension 0 low == -1.0"), FMath::IsNearlyEqual((float)Dims.Get(0).low(), -1.0f));
		TestTrue(TEXT("Dimension 0 high == 1.0"), FMath::IsNearlyEqual((float)Dims.Get(0).high(), 1.0f));
		TestTrue(TEXT("Dimension 1 low == 0.0"), FMath::IsNearlyEqual((float)Dims.Get(1).low(), 0.0f));
		TestTrue(TEXT("Dimension 1 high == 2.0"), FMath::IsNearlyEqual((float)Dims.Get(1).high(), 2.0f));

		auto& Shape = OutProto.box_space().shape_dimensions();
		TestEqual(TEXT("BoxSpace.shape_dimensions size == 1"), (int)Shape.size(), 1);
		TestEqual(TEXT("BoxSpace.shape_dimensions[0] == 2"), Shape.Get(0), 2);
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufTextSpaceSerializationTest, "Schola.Protobuf.Serialization.Spaces.Text", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufTextSpaceSerializationTest::RunTest(const FString& Parameters)
{
	TInstancedStruct<FSpace> Inst;
	Inst.InitializeAs<FTextSpace>();
	FTextSpace* Space = Inst.GetMutablePtr<FTextSpace>();
	Space->MaxLength = 32;
	Space->bHasMinLength = true;
	Space->MinLength = 4;
	Space->Charset = TEXT("abc");

	Schola::Space OutProto;
	ProtobufSerializer::ToProto(Inst, &OutProto);

	TestTrue(TEXT("Text space serialized as text_space"), OutProto.has_text_space());
	if (OutProto.has_text_space())
	{
		TestEqual(TEXT("TextSpace.max_length == 32"), OutProto.text_space().max_length(), 32);
		TestTrue(TEXT("TextSpace has min_length set"), OutProto.text_space().has_min_length());
		TestEqual(TEXT("TextSpace.min_length == 4"), OutProto.text_space().min_length(), 4);
		TestTrue(TEXT("TextSpace has charset set"), OutProto.text_space().has_charset());
		TestEqual(TEXT("TextSpace.charset == abc"), FString(UTF8_TO_TCHAR(OutProto.text_space().charset().c_str())), FString(TEXT("abc")));
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufTextSpaceMinimalSerializationTest, "Schola.Protobuf.Serialization.Spaces.TextMinimal", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufTextSpaceMinimalSerializationTest::RunTest(const FString& Parameters)
{
	// When min length and charset are unset, the optional proto fields should stay unset.
	TInstancedStruct<FSpace> Inst;
	Inst.InitializeAs<FTextSpace>();
	FTextSpace* Space = Inst.GetMutablePtr<FTextSpace>();
	Space->MaxLength = 10;

	Schola::Space OutProto;
	ProtobufSerializer::ToProto(Inst, &OutProto);

	TestTrue(TEXT("Text space serialized as text_space"), OutProto.has_text_space());
	if (OutProto.has_text_space())
	{
		TestEqual(TEXT("TextSpace.max_length == 10"), OutProto.text_space().max_length(), 10);
		TestFalse(TEXT("TextSpace min_length unset"), OutProto.text_space().has_min_length());
		TestFalse(TEXT("TextSpace charset unset"), OutProto.text_space().has_charset());
	}

	return true;
}
