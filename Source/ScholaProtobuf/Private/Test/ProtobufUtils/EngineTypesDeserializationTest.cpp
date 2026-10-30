// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "ProtobufUtils/ProtobufDeserializer.h"

#include <string>

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufStdStringToFStringDeserializationTest, "Schola.Protobuf.Deserialization.FStringToStdString", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufStdStringToFStringDeserializationTest::RunTest(const FString& Parameters)
{
	std::string In = std::string(TCHAR_TO_UTF8(*FString(TEXT("hello world"))));
	FString Out;
	ProtobufDeserializer::FromProto(In, Out);

	const FString Expected = FString(TEXT("hello world"));
	TestTrue(TEXT("Converted FString should match original"), Out == Expected);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufProtoMapToTMapFStringDeserializationTest, "Schola.Protobuf.Deserialization.Maps.ProtoMapToTMapFStringFString", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufProtoMapToTMapFStringDeserializationTest::RunTest(const FString& Parameters)
{
	Schola::AgentState Proto;
	(*Proto.mutable_info())[std::string(TCHAR_TO_UTF8(TEXT("alpha")))] = std::string(TCHAR_TO_UTF8(TEXT("one")));
	(*Proto.mutable_info())[std::string(TCHAR_TO_UTF8(TEXT("beta")))] = std::string(TCHAR_TO_UTF8(TEXT("two")));

	TMap<FString, FString> OutMap;
	ProtobufDeserializer::FromProto(Proto.info(), OutMap);

	auto ItA = OutMap.Find(TEXT("alpha"));
	auto ItB = OutMap.Find(TEXT("beta"));

	TestTrue(TEXT("alpha present in TMap"), ItA != nullptr);
	TestTrue(TEXT("beta present in TMap"), ItB != nullptr);
	if (ItA) TestTrue(TEXT("alpha == one"), *ItA == TEXT("one"));
	if (ItB) TestTrue(TEXT("beta == two"), *ItB == TEXT("two"));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufScalarFromProtoTest, "Schola.Protobuf.Deserialization.Scalar", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufScalarFromProtoTest::RunTest(const FString& Parameters)
{
	int Out = 0;
	ProtobufDeserializer::FromProto(42, Out);

	TestEqual(TEXT("Scalar identity conversion assigns the value"), Out, 42);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufRepeatedFieldToTArrayTest, "Schola.Protobuf.Deserialization.Arrays.RepeatedFieldToTArray", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufRepeatedFieldToTArrayTest::RunTest(const FString& Parameters)
{
	google::protobuf::RepeatedField<int> In;
	In.Add(1);
	In.Add(2);
	In.Add(3);

	TArray<int> Out;
	Out.Add(99);
	ProtobufDeserializer::FromProto(In, Out);

	TestFalse(TEXT("Sentinel value is cleared"), Out.Contains(99));
	TestEqual(TEXT("RepeatedField is copied in order"), Out, TArray<int>({1, 2, 3}));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufRepeatedPtrFieldToTArrayTest, "Schola.Protobuf.Deserialization.Arrays.RepeatedPtrFieldToTArray", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufRepeatedPtrFieldToTArrayTest::RunTest(const FString& Parameters)
{
	google::protobuf::RepeatedPtrField<std::string> In;
	*In.Add() = std::string(TCHAR_TO_UTF8(TEXT("alpha")));
	*In.Add() = std::string(TCHAR_TO_UTF8(TEXT("beta")));

	TArray<FString> Out;
	Out.Add(TEXT("sentinel"));
	ProtobufDeserializer::FromProto(In, Out);

	TestFalse(TEXT("Sentinel string is cleared"), Out.Contains(TEXT("sentinel")));
	TestEqual(TEXT("RepeatedPtrField size"), Out.Num(), 2);
	if (Out.Num() == 2)
	{
		TestEqual(TEXT("First element == alpha"), Out[0], FString(TEXT("alpha")));
		TestEqual(TEXT("Second element == beta"), Out[1], FString(TEXT("beta")));
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufProtoMapIntToTMapTest, "Schola.Protobuf.Deserialization.Maps.ProtoMapIntToTMapIntFString", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufProtoMapIntToTMapTest::RunTest(const FString& Parameters)
{
	google::protobuf::Map<int, std::string> In;
	In[1] = std::string(TCHAR_TO_UTF8(TEXT("one")));
	In[4] = std::string(TCHAR_TO_UTF8(TEXT("four")));

	TMap<int, FString> Out;
	Out.Add(99, TEXT("sentinel"));
	ProtobufDeserializer::FromProto(In, Out);

	TestFalse(TEXT("Sentinel key 99 is cleared"), Out.Contains(99));
	const FString* Value1 = Out.Find(1);
	const FString* Value4 = Out.Find(4);
	TestTrue(TEXT("Key 1 is present"), Value1 != nullptr);
	TestTrue(TEXT("Key 4 is present"), Value4 != nullptr);
	if (Value1) TestEqual(TEXT("Key 1 == one"), *Value1, FString(TEXT("one")));
	if (Value4) TestEqual(TEXT("Key 4 == four"), *Value4, FString(TEXT("four")));

	return true;
}
