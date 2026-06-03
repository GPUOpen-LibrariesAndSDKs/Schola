// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "ProtobufUtils/ProtobufDeserializer.h"

#include "Spaces/DictSpace.h"
#include "Spaces/MultiBinarySpace.h"
#include "Spaces/DiscreteSpace.h"
#include "Spaces/MultiDiscreteSpace.h"
#include "Spaces/BoxSpace.h"
#include "Spaces/BoxSpaceDimension.h"
#include "Spaces/TextSpace.h"

#include <string>

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufDiscreteSpaceDeserializationTest, "Schola.Protobuf.Deserialization.Spaces.Discrete", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufDiscreteSpaceDeserializationTest::RunTest(const FString& Parameters)
{
	Schola::Space InProto;
	InProto.mutable_discrete_space()->set_high(7);

	TInstancedStruct<FSpace> Out;
	ProtobufDeserializer::FromProto(InProto, Out);

	const FDiscreteSpace* Space = Out.GetPtr<FDiscreteSpace>();
	TestTrue(TEXT("Discrete space deserialized as discrete_space"), Space != nullptr);
	if (Space)
	{
		TestEqual(TEXT("DiscreteSpace.high == 7"), Space->High, 7);
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufMultiDiscreteSpaceDeserializationTest, "Schola.Protobuf.Deserialization.Spaces.MultiDiscrete", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufMultiDiscreteSpaceDeserializationTest::RunTest(const FString& Parameters)
{
	Schola::Space InProto;
	auto* MultiProto = InProto.mutable_multi_discrete_space();
	MultiProto->add_high(3);
	MultiProto->add_high(4);
	MultiProto->add_high(5);

	TInstancedStruct<FSpace> Out;
	ProtobufDeserializer::FromProto(InProto, Out);

	const FMultiDiscreteSpace* Space = Out.GetPtr<FMultiDiscreteSpace>();
	TestTrue(TEXT("MultiDiscrete space deserialized as multi_discrete_space"), Space != nullptr);
	if (Space)
	{
		TestEqual(TEXT("MultiDiscrete high size == 3"), Space->High.Num(), 3);
		TestEqual(TEXT("High[0] == 3"), Space->High[0], 3);
		TestEqual(TEXT("High[1] == 4"), Space->High[1], 4);
		TestEqual(TEXT("High[2] == 5"), Space->High[2], 5);
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufMultiBinarySpaceDeserializationTest, "Schola.Protobuf.Deserialization.Spaces.MultiBinary", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufMultiBinarySpaceDeserializationTest::RunTest(const FString& Parameters)
{
	Schola::Space InProto;
	InProto.mutable_multi_binary_space()->set_shape(8);

	TInstancedStruct<FSpace> Out;
	ProtobufDeserializer::FromProto(InProto, Out);

	const FMultiBinarySpace* Space = Out.GetPtr<FMultiBinarySpace>();
	TestTrue(TEXT("MultiBinary space deserialized as multi_binary_space"), Space != nullptr);
	if (Space)
	{
		TestEqual(TEXT("MultiBinary.shape == 8"), Space->Shape, 8);
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufBoxSpaceDeserializationTest, "Schola.Protobuf.Deserialization.Spaces.Box", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufBoxSpaceDeserializationTest::RunTest(const FString& Parameters)
{
	Schola::Space InProto;
	auto* BoxProto = InProto.mutable_box_space();

	auto* DimA = BoxProto->add_dimensions();
	DimA->set_low(-1.0f);
	DimA->set_high(1.0f);
	auto* DimB = BoxProto->add_dimensions();
	DimB->set_low(0.0f);
	DimB->set_high(2.0f);

	BoxProto->add_shape_dimensions(2);

	TInstancedStruct<FSpace> Out;
	ProtobufDeserializer::FromProto(InProto, Out);

	const FBoxSpace* Space = Out.GetPtr<FBoxSpace>();
	TestTrue(TEXT("Box space deserialized as box_space"), Space != nullptr);
	if (Space)
	{
		TestEqual(TEXT("BoxSpace has 2 dims"), Space->Dimensions.Num(), 2);
		if (Space->Dimensions.Num() >= 2)
		{
			TestTrue(TEXT("Dimension 0 low == -1.0"), FMath::IsNearlyEqual(Space->Dimensions[0].Low, -1.0f));
			TestTrue(TEXT("Dimension 0 high == 1.0"), FMath::IsNearlyEqual(Space->Dimensions[0].High, 1.0f));
			TestTrue(TEXT("Dimension 1 low == 0.0"), FMath::IsNearlyEqual(Space->Dimensions[1].Low, 0.0f));
			TestTrue(TEXT("Dimension 1 high == 2.0"), FMath::IsNearlyEqual(Space->Dimensions[1].High, 2.0f));
		}

		TestEqual(TEXT("BoxSpace.shape size == 1"), Space->Shape.Num(), 1);
		if (Space->Shape.Num() > 0)
		{
			TestEqual(TEXT("BoxSpace.shape[0] == 2"), Space->Shape[0], 2);
		}
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufTextSpaceDeserializationTest, "Schola.Protobuf.Deserialization.Spaces.Text", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufTextSpaceDeserializationTest::RunTest(const FString& Parameters)
{
	Schola::Space InProto;
	auto* TextProto = InProto.mutable_text_space();
	TextProto->set_max_length(32);
	TextProto->set_min_length(4);
	TextProto->set_charset("abc");

	TInstancedStruct<FSpace> Out;
	ProtobufDeserializer::FromProto(InProto, Out);

	const FTextSpace* Space = Out.GetPtr<FTextSpace>();
	TestTrue(TEXT("Text space deserialized as text_space"), Space != nullptr);
	if (Space)
	{
		TestEqual(TEXT("TextSpace.MaxLength == 32"), Space->MaxLength, 32);
		TestTrue(TEXT("TextSpace.bHasMinLength true"), Space->bHasMinLength);
		TestEqual(TEXT("TextSpace.MinLength == 4"), Space->MinLength, 4);
		TestEqual(TEXT("TextSpace.Charset == abc"), Space->Charset, FString(TEXT("abc")));
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufTextSpaceMinimalDeserializationTest, "Schola.Protobuf.Deserialization.Spaces.TextMinimal", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufTextSpaceMinimalDeserializationTest::RunTest(const FString& Parameters)
{
	// Only max_length set: min length flag should be false and charset empty.
	Schola::Space InProto;
	InProto.mutable_text_space()->set_max_length(10);

	TInstancedStruct<FSpace> Out;
	ProtobufDeserializer::FromProto(InProto, Out);

	const FTextSpace* Space = Out.GetPtr<FTextSpace>();
	TestTrue(TEXT("Text space deserialized as text_space"), Space != nullptr);
	if (Space)
	{
		TestEqual(TEXT("TextSpace.MaxLength == 10"), Space->MaxLength, 10);
		TestFalse(TEXT("TextSpace.bHasMinLength false"), Space->bHasMinLength);
		TestTrue(TEXT("TextSpace.Charset empty"), Space->Charset.IsEmpty());
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufDictSpaceDeserializationTest, "Schola.Protobuf.Deserialization.Spaces.Dict", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufDictSpaceDeserializationTest::RunTest(const FString& Parameters)
{
	// Create inner discrete space proto
	Schola::Space InnerProto;
	InnerProto.mutable_discrete_space()->set_high(5);

	// Create dict space proto and add entry "inner"
	Schola::Space InProto;
	auto* DictProto = InProto.mutable_dict_space();
	(*DictProto->mutable_spaces())[std::string(TCHAR_TO_UTF8(TEXT("inner")))] = InnerProto;

	TInstancedStruct<FSpace> Out;
	ProtobufDeserializer::FromProto(InProto, Out);

	const FDictSpace* Dict = Out.GetPtr<FDictSpace>();
	TestTrue(TEXT("Dict space deserialized as dict_space"), Dict != nullptr);
	if (Dict)
	{
		const TInstancedStruct<FSpace>* Sub = Dict->Spaces.Find(TEXT("inner"));
		TestTrue(TEXT("Dict contains key 'inner'"), Sub != nullptr);
		if (Sub)
		{
			const FDiscreteSpace* Inner = Sub->GetPtr<FDiscreteSpace>();
			TestTrue(TEXT("Inner deserialized as discrete_space"), Inner != nullptr);
			if (Inner)
			{
				TestEqual(TEXT("Inner.high == 5"), Inner->High, 5);
			}
		}
	}

	return true;
}
