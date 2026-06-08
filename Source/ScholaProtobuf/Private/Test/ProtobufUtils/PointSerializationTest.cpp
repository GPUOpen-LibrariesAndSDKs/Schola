// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "ProtobufUtils/ProtobufSerializer.h"

#include "Points/Point.h"
#include "Points/DictPoint.h"
#include "Points/MultiBinaryPoint.h"
#include "Points/DiscretePoint.h"
#include "Points/MultiDiscretePoint.h"
#include "Points/BoxPoint.h"
#include "Points/TextPoint.h"

#include <string>

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufDiscretePointSerializationTest, "Schola.Protobuf.Serialization.Points.Discrete", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufDiscretePointSerializationTest::RunTest(const FString& Parameters)
{
	TInstancedStruct<FPoint> Inst;
	Inst.InitializeAs<FDiscretePoint>();
	FDiscretePoint* Discrete = Inst.GetMutablePtr<FDiscretePoint>();
	Discrete->Value = 5;

	Schola::Point OutProto;
	ProtobufSerializer::ToProto(Inst, &OutProto);

	TestTrue(TEXT("Discrete point serialized as discrete_point"), OutProto.has_discrete_point());
	if (OutProto.has_discrete_point())
	{
		TestEqual(TEXT("Discrete value matches"), OutProto.discrete_point().value(), 5);
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufMultiDiscretePointSerializationTest, "Schola.Protobuf.Serialization.Points.MultiDiscrete", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufMultiDiscretePointSerializationTest::RunTest(const FString& Parameters)
{
	TInstancedStruct<FPoint> Inst;
	Inst.InitializeAs<FMultiDiscretePoint>();
	FMultiDiscretePoint* Multi = Inst.GetMutablePtr<FMultiDiscretePoint>();
	Multi->Values = TArray<int>({1,2,3});

	Schola::Point OutProto;
	ProtobufSerializer::ToProto(Inst, &OutProto);

	TestTrue(TEXT("MultiDiscrete point serialized as multi_discrete_point"), OutProto.has_multi_discrete_point());
	if (OutProto.has_multi_discrete_point())
	{
		auto& Repeated = OutProto.multi_discrete_point().values();
		TestEqual(TEXT("MultiDiscrete size == 3"), (int)Repeated.size(), 3);
		TestEqual(TEXT("Value[0] == 1"), Repeated.Get(0), 1);
		TestEqual(TEXT("Value[1] == 2"), Repeated.Get(1), 2);
		TestEqual(TEXT("Value[2] == 3"), Repeated.Get(2), 3);
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufMultiBinaryPointSerializationTest, "Schola.Protobuf.Serialization.Points.MultiBinary", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufMultiBinaryPointSerializationTest::RunTest(const FString& Parameters)
{
	TInstancedStruct<FPoint> Inst;
	Inst.InitializeAs<FMultiBinaryPoint>();
	FMultiBinaryPoint* Multi = Inst.GetMutablePtr<FMultiBinaryPoint>();
	Multi->Values = TArray<bool>({false,true,true,false});

	Schola::Point OutProto;
	ProtobufSerializer::ToProto(Inst, &OutProto);

	TestTrue(TEXT("MultiBinary point serialized as multi_binary_point"), OutProto.has_multi_binary_point());
	if (OutProto.has_multi_binary_point())
	{
		auto& Repeated = OutProto.multi_binary_point().values();
		TestEqual(TEXT("MultiBinary size == 4"), (int)Repeated.size(), 4);
		TestEqual(TEXT("Value[0] == false"), Repeated.Get(0), false);
		TestEqual(TEXT("Value[1] == true"), Repeated.Get(1), true);
		TestEqual(TEXT("Value[2] == true"), Repeated.Get(2), true);
		TestEqual(TEXT("Value[3] == false"), Repeated.Get(3), false);
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufBoxPointSerializationTest, "Schola.Protobuf.Serialization.Points.Box", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufBoxPointSerializationTest::RunTest(const FString& Parameters)
{
	TInstancedStruct<FPoint> Inst;
	Inst.InitializeAs<FBoxPoint>();
	FBoxPoint* Box = Inst.GetMutablePtr<FBoxPoint>();
	Box->Values = TArray<float>({0.1f, 0.2f, 0.3f, 0.4f});
	Box->Shape = TArray<int>({ 2, 2 });
	Schola::Point OutProto;
	ProtobufSerializer::ToProto(Inst, &OutProto);

	TestTrue(TEXT("Box point serialized as box_point"), OutProto.has_box_point());
	if (OutProto.has_box_point())
	{
		auto& Repeated = OutProto.box_point().values();
		TestEqual(TEXT("Number of Serialized Values == 3"), (int)Repeated.size(), 4);
		TestTrue(TEXT("Box[0] approx 0.1"), FMath::IsNearlyEqual((float)Repeated.Get(0), 0.1f));
		TestTrue(TEXT("Box[1] approx 0.2"), FMath::IsNearlyEqual((float)Repeated.Get(1), 0.2f));
		TestTrue(TEXT("Box[2] approx 0.3"), FMath::IsNearlyEqual((float)Repeated.Get(2), 0.3f));
		TestTrue(TEXT("Box Shape Has 2 Dimensions"), OutProto.box_point().shape().size() == 2);
		TestEqual(TEXT("Box Shape First Dimension is 2"), OutProto.box_point().shape().Get(0), 2);
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufTextPointSerializationTest, "Schola.Protobuf.Serialization.Points.Text", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufTextPointSerializationTest::RunTest(const FString& Parameters)
{
	TInstancedStruct<FPoint> Inst;
	Inst.InitializeAs<FTextPoint>(FString(TEXT("hello world")));

	Schola::Point OutProto;
	ProtobufSerializer::ToProto(Inst, &OutProto);

	TestTrue(TEXT("Text point serialized as text_point"), OutProto.has_text_point());
	if (OutProto.has_text_point())
	{
		TestEqual(TEXT("Text value matches"), FString(UTF8_TO_TCHAR(OutProto.text_point().value().c_str())), FString(TEXT("hello world")));
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufDictPointSerializationTest, "Schola.Protobuf.Serialization.Points.Dict", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufDictPointSerializationTest::RunTest(const FString& Parameters)
{
	// Create inner discrete point
	TInstancedStruct<FPoint> Inner;
	Inner.InitializeAs<FDiscretePoint>(42);

	// Create dict point and add entry "inner"
	TInstancedStruct<FPoint> Inst;
	Inst.InitializeAs<FDictPoint>();
	FDictPoint* Dict = Inst.GetMutablePtr<FDictPoint>();
	Dict->Points.Add(TEXT("inner"), Inner);

	Schola::Point OutProto;
	ProtobufSerializer::ToProto(Inst, &OutProto);

	TestTrue(TEXT("Dict point serialized as dict_point"), OutProto.has_dict_point());
	if (OutProto.has_dict_point())
	{
		const auto& Map = OutProto.dict_point().values();
		auto It = Map.find(std::string("inner"));
		TestTrue(TEXT("Dict contains key 'inner'"), It != Map.end());
		if (It != Map.end())
		{
			const Schola::Point& InnerProto = It->second;
			TestTrue(TEXT("Inner serialized as discrete_point"), InnerProto.has_discrete_point());
			if (InnerProto.has_discrete_point())
			{
				TestEqual(TEXT("Inner value == 42"), InnerProto.discrete_point().value(), 42);
			}
		}
	}

	return true;
}
