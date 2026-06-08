// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Points/TextPoint.h"

#if WITH_AUTOMATION_TESTS

// Constructor Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextPointDefaultConstructorTest, "Schola.Points.TextPoint.Default Constructor Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextPointDefaultConstructorTest::RunTest(const FString& Parameters)
{
	FTextPoint TextPoint = FTextPoint();

	TestTrue(TEXT("TextPoint.Value is empty"), TextPoint.Value.IsEmpty());

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextPointFromStringConstructorTest, "Schola.Points.TextPoint.From String Constructor Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextPointFromStringConstructorTest::RunTest(const FString& Parameters)
{
	FTextPoint TextPoint = FTextPoint(TEXT("hello world"));

	TestEqual(TEXT("TextPoint.Value == hello world"), TextPoint.Value, FString(TEXT("hello world")));

	return true;
}

// Method Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextPointToStringTest, "Schola.Points.TextPoint.ToString Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextPointToStringTest::RunTest(const FString& Parameters)
{
	FTextPoint TextPoint = FTextPoint(TEXT("hello world"));

	TestEqual(TEXT("TextPoint.ToString() == hello world"), TextPoint.ToString(), FString(TEXT("hello world")));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextPointResetTest, "Schola.Points.TextPoint.Reset Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextPointResetTest::RunTest(const FString& Parameters)
{
	FTextPoint TextPoint = FTextPoint(TEXT("hello world"));
	TextPoint.Reset();

	TestTrue(TEXT("TextPoint.Value is empty after Reset"), TextPoint.Value.IsEmpty());

	return true;
}

#endif
