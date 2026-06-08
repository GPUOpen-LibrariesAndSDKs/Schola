// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "ProtobufUtils/ProtobufDeserializer.h"

#include "Points/Point.h"
#include "Points/DictPoint.h"
#include "Points/MultiBinaryPoint.h"
#include "Points/DiscretePoint.h"
#include "Points/MultiDiscretePoint.h"
#include "Points/BoxPoint.h"
#include "Points/TextPoint.h"

#include <string>

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufDiscretePointDeserializationTest, "Schola.Protobuf.Deserialization.Points.Discrete", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufDiscretePointDeserializationTest::RunTest(const FString& Parameters)
{
	Schola::Point InProto;
	InProto.mutable_discrete_point()->set_value(5);

	TInstancedStruct<FPoint> Out;
	ProtobufDeserializer::FromProto(InProto, Out);

	const FDiscretePoint* Discrete = Out.GetPtr<FDiscretePoint>();
	TestTrue(TEXT("Discrete point deserialized as discrete_point"), Discrete != nullptr);
	if (Discrete)
	{
		TestEqual(TEXT("Discrete value matches"), Discrete->Value, 5);
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufMultiDiscretePointDeserializationTest, "Schola.Protobuf.Deserialization.Points.MultiDiscrete", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufMultiDiscretePointDeserializationTest::RunTest(const FString& Parameters)
{
	Schola::Point InProto;
	auto* MultiProto = InProto.mutable_multi_discrete_point();
	MultiProto->add_values(1);
	MultiProto->add_values(2);
	MultiProto->add_values(3);

	TInstancedStruct<FPoint> Out;
	ProtobufDeserializer::FromProto(InProto, Out);

	const FMultiDiscretePoint* Multi = Out.GetPtr<FMultiDiscretePoint>();
	TestTrue(TEXT("MultiDiscrete point deserialized as multi_discrete_point"), Multi != nullptr);
	if (Multi)
	{
		TestEqual(TEXT("MultiDiscrete Values"), Multi->Values, TArray<int>({1, 2, 3}));
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufMultiBinaryPointDeserializationTest, "Schola.Protobuf.Deserialization.Points.MultiBinary", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufMultiBinaryPointDeserializationTest::RunTest(const FString& Parameters)
{
	Schola::Point InProto;
	auto* BinProto = InProto.mutable_multi_binary_point();
	BinProto->add_values(0);
	BinProto->add_values(1);
	BinProto->add_values(1);
	BinProto->add_values(0);

	TInstancedStruct<FPoint> Out;
	ProtobufDeserializer::FromProto(InProto, Out);

	const FMultiBinaryPoint* Multi = Out.GetPtr<FMultiBinaryPoint>();
	TestTrue(TEXT("MultiBinary point deserialized as multi_binary_point"), Multi != nullptr);
	if (Multi)
	{
		TestEqual(TEXT("MultiBinary Values"), Multi->Values, TArray<bool>({false, true, true, false}));
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufBoxPointDeserializationTest, "Schola.Protobuf.Deserialization.Points.Box", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufBoxPointDeserializationTest::RunTest(const FString& Parameters)
{
	Schola::Point InProto;
	auto* BoxProto = InProto.mutable_box_point();
	BoxProto->add_values(0.1f);
	BoxProto->add_values(0.2f);
	BoxProto->add_values(0.3f);

	TInstancedStruct<FPoint> Out;
	ProtobufDeserializer::FromProto(InProto, Out);

	const FBoxPoint* Box = Out.GetPtr<FBoxPoint>();
	TestTrue(TEXT("Box point deserialized as box_point"), Box != nullptr);
	if (Box)
	{
		TestEqual(TEXT("Box size == 3"), Box->Values.Num(), 3);
		TestTrue(TEXT("Box[0] approx 0.1"), FMath::IsNearlyEqual(Box->Values[0], 0.1f));
		TestTrue(TEXT("Box[1] approx 0.2"), FMath::IsNearlyEqual(Box->Values[1], 0.2f));
		TestTrue(TEXT("Box[2] approx 0.3"), FMath::IsNearlyEqual(Box->Values[2], 0.3f));
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufTextPointDeserializationTest, "Schola.Protobuf.Deserialization.Points.Text", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufTextPointDeserializationTest::RunTest(const FString& Parameters)
{
	Schola::Point InProto;
	InProto.mutable_text_point()->set_value("hello world");

	TInstancedStruct<FPoint> Out;
	ProtobufDeserializer::FromProto(InProto, Out);

	const FTextPoint* Text = Out.GetPtr<FTextPoint>();
	TestTrue(TEXT("Text point deserialized as text_point"), Text != nullptr);
	if (Text)
	{
		TestEqual(TEXT("Text value matches"), Text->Value, FString(TEXT("hello world")));
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufDictPointDeserializationTest, "Schola.Protobuf.Deserialization.Points.Dict", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufDictPointDeserializationTest::RunTest(const FString& Parameters)
{
	// Create inner discrete point proto
	Schola::Point InnerProto;
	InnerProto.mutable_discrete_point()->set_value(42);

	// Create dict point proto and add entry "inner"
	Schola::Point InProto;
	auto* DictProto = InProto.mutable_dict_point();
	(*DictProto->mutable_values())[std::string(TCHAR_TO_UTF8(TEXT("inner")))] = InnerProto;

	TInstancedStruct<FPoint> Out;
	ProtobufDeserializer::FromProto(InProto, Out);

	const FDictPoint* Dict = Out.GetPtr<FDictPoint>();
	TestTrue(TEXT("Dict point deserialized as dict_point"), Dict != nullptr);
	if (Dict)
	{
		const TInstancedStruct<FPoint>* Sub = Dict->Points.Find(TEXT("inner"));
		TestTrue(TEXT("Dict contains key 'inner'"), Sub != nullptr);
		if (Sub)
		{
			const FDiscretePoint* Inner = Sub->GetPtr<FDiscretePoint>();
			TestTrue(TEXT("Inner deserialized as discrete_point"), Inner != nullptr);
			if (Inner)
			{
				TestEqual(TEXT("Inner value == 42"), Inner->Value, 42);
			}
		}
	}

	return true;
}
