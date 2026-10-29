// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "ProtobufUtils/ProtobufSerializer.h"

#include <string>

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufFStringToStdStringTest, "Schola.Protobuf.Serialization.FStringToStdString", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufFStringToStdStringTest::RunTest(const FString& Parameters)
{
	FString In = TEXT("hello world");
	std::string Out;
	ProtobufSerializer::ToProto(In, &Out);

	const std::string Expected = TCHAR_TO_UTF8(*In);
	TestTrue(TEXT("Converted std::string should match UTF-8 of FString"), Out == Expected);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufTMapFStringToProtoTest, "Schola.Protobuf.Serialization.Maps.TMapFStringFString", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufTMapFStringToProtoTest::RunTest(const FString& Parameters)
{
	TMap<FString, FString> UnrealMap;
	UnrealMap.Add(TEXT("alpha"), TEXT("one"));
	UnrealMap.Add(TEXT("beta"), TEXT("two"));

	Schola::AgentState Proto; // use agent state info map as a target
	ProtobufSerializer::ToProto(UnrealMap, Proto.mutable_info());

	const auto& Map = Proto.info();
	auto ItA = Map.find(std::string(TCHAR_TO_UTF8(TEXT("alpha"))));
	auto ItB = Map.find(std::string(TCHAR_TO_UTF8(TEXT("beta"))));

	TestTrue(TEXT("alpha present in proto map"), ItA != Map.end());
	TestTrue(TEXT("beta present in proto map"), ItB != Map.end());
	if (ItA != Map.end()) TestTrue(TEXT("alpha == one"), ItA->second == std::string(TCHAR_TO_UTF8(TEXT("one"))));
	if (ItB != Map.end()) TestTrue(TEXT("beta == two"), ItB->second == std::string(TCHAR_TO_UTF8(TEXT("two"))));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufScalarPointerToProtoTest, "Schola.Protobuf.Serialization.ScalarPointer", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufScalarPointerToProtoTest::RunTest(const FString& Parameters)
{
	int Out = 0;
	ProtobufSerializer::ToProto(42, &Out);

	TestEqual(TEXT("Scalar pointer conversion assigns the value"), Out, 42);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufScalarReferenceToProtoTest, "Schola.Protobuf.Serialization.ScalarReference", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufScalarReferenceToProtoTest::RunTest(const FString& Parameters)
{
	int Out = 0;
	ProtobufSerializer::ToProto(42, Out);

	TestEqual(TEXT("Scalar reference conversion assigns the value"), Out, 42);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufTArrayToRepeatedFieldTest, "Schola.Protobuf.Serialization.Arrays.TArrayToRepeatedField", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufTArrayToRepeatedFieldTest::RunTest(const FString& Parameters)
{
	TArray<int> In({1, 2, 3});
	google::protobuf::RepeatedField<int> Out;
	Out.Add(9);

	ProtobufSerializer::ToProto(In, &Out);

	TestEqual(TEXT("RepeatedField keeps the pre-existing element and appends"), Out.size(), 4);
	TestEqual(TEXT("Pre-existing element stays at index 0"), Out.Get(0), 9);
	TestEqual(TEXT("Appended value 1"), Out.Get(1), 1);
	TestEqual(TEXT("Appended value 2"), Out.Get(2), 2);
	TestEqual(TEXT("Appended value 3"), Out.Get(3), 3);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufTArrayToRepeatedPtrFieldTest, "Schola.Protobuf.Serialization.Arrays.TArrayToRepeatedPtrField", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufTArrayToRepeatedPtrFieldTest::RunTest(const FString& Parameters)
{
	TArray<FString> In({TEXT("alpha"), TEXT("beta")});
	google::protobuf::RepeatedPtrField<std::string> Out;

	ProtobufSerializer::ToProto(In, &Out);

	TestEqual(TEXT("RepeatedPtrField size"), Out.size(), 2);
	TestTrue(TEXT("First element is UTF-8 of alpha"), Out.Get(0) == std::string(TCHAR_TO_UTF8(TEXT("alpha"))));
	TestTrue(TEXT("Second element is UTF-8 of beta"), Out.Get(1) == std::string(TCHAR_TO_UTF8(TEXT("beta"))));

	return true;
}

// Test the ability to convert a value using a function pointer, by just capturing the value
namespace
{
	int CapturedInt = 0;

	void CaptureInt(int Value)
	{
		CapturedInt = Value;
	}
}


IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufFunctionPointerSetterToProtoTest, "Schola.Protobuf.Serialization.Setter", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufFunctionPointerSetterToProtoTest::RunTest(const FString& Parameters)
{
	CapturedInt = 0;
	ProtobufSerializer::ToProto(7, &CaptureInt);

	TestEqual(TEXT("Setter is invoked with the value"), CapturedInt, 7);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufTMapIntToProtoMapTest, "Schola.Protobuf.Serialization.Maps.TMapIntFString", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufTMapIntToProtoMapTest::RunTest(const FString& Parameters)
{
	TMap<int, FString> In;
	In.Add(1, TEXT("one"));
	In.Add(4, TEXT("four"));

	google::protobuf::Map<int, std::string> Out;
	Out[7] = std::string("keep");

	ProtobufSerializer::ToProto(In, &Out);

	auto It1 = Out.find(1);
	auto It4 = Out.find(4);
	auto It7 = Out.find(7);

	TestTrue(TEXT("Key 1 is present"), It1 != Out.end());
	TestTrue(TEXT("Key 4 is present"), It4 != Out.end());
	TestTrue(TEXT("Key 2 is absent"), Out.find(2) == Out.end());
	TestTrue(TEXT("Pre-existing key 7 is kept"), It7 != Out.end());
	if (It1 != Out.end()) TestTrue(TEXT("Key 1 == one"), It1->second == std::string(TCHAR_TO_UTF8(TEXT("one"))));
	if (It4 != Out.end()) TestTrue(TEXT("Key 4 == four"), It4->second == std::string(TCHAR_TO_UTF8(TEXT("four"))));
	if (It7 != Out.end()) TestTrue(TEXT("Key 7 == keep"), It7->second == std::string("keep"));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufTArrayWithKeysToProtoMapTest, "Schola.Protobuf.Serialization.Maps.TArrayWithKeysToMap", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufTArrayWithKeysToProtoMapTest::RunTest(const FString& Parameters)
{
	TArray<FString> Values({TEXT("A"), TEXT("B"), TEXT("C")});
	TArray<int> Keys({0, 2});
	google::protobuf::Map<int, std::string> Out;

	ProtobufSerializer::ToProto(Values, Keys, &Out);

	auto It0 = Out.find(0);
	auto It2 = Out.find(2);

	TestTrue(TEXT("Proto map size is 2"), (int)Out.size() == 2);
	TestTrue(TEXT("Key 0 is present"), It0 != Out.end());
	TestTrue(TEXT("Key 2 is present"), It2 != Out.end());
	TestTrue(TEXT("Key 1 is not inserted"), Out.find(1) == Out.end());
	if (It0 != Out.end()) TestTrue(TEXT("Key 0 indexes value A"), It0->second == std::string(TCHAR_TO_UTF8(TEXT("A"))));
	if (It2 != Out.end()) TestTrue(TEXT("Key 2 indexes value C"), It2->second == std::string(TCHAR_TO_UTF8(TEXT("C"))));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FProtobufTArrayToIntKeyedProtoMapTest, "Schola.Protobuf.Serialization.Maps.TArrayToIntKeyedMap", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FProtobufTArrayToIntKeyedProtoMapTest::RunTest(const FString& Parameters)
{
	TArray<FString> Values({TEXT("A"), TEXT("B")});
	google::protobuf::Map<int, std::string> Out;

	ProtobufSerializer::ToProto(Values, &Out);

	auto It0 = Out.find(0);
	auto It1 = Out.find(1);

	TestTrue(TEXT("Proto map size is 2"), (int)Out.size() == 2);
	TestTrue(TEXT("Key 0 is present"), It0 != Out.end());
	TestTrue(TEXT("Key 1 is present"), It1 != Out.end());
	if (It0 != Out.end()) TestTrue(TEXT("Key 0 == A"), It0->second == std::string(TCHAR_TO_UTF8(TEXT("A"))));
	if (It1 != Out.end()) TestTrue(TEXT("Key 1 == B"), It1->second == std::string(TCHAR_TO_UTF8(TEXT("B"))));

	return true;
}
