// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Points/MultiDiscretePoint.h"

#if WITH_AUTOMATION_TESTS

// Constructor Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiDiscretePointFromArrayTest, "Schola.Points.MultiDiscretePoint.From Array Constructor Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiDiscretePointFromArrayTest::RunTest(const FString& Parameters)
{
    TArray<int> Values = { 1, 2, 3 };
	FMultiDiscretePoint DiscretePoint = FMultiDiscretePoint(Values);

    TestEqual(TEXT("DiscretePoint[0] == 1"), DiscretePoint[0], 1);
    TestEqual(TEXT("DiscretePoint[1] == 2"), DiscretePoint[1], 2);
    TestEqual(TEXT("DiscretePoint[2] == 3"), DiscretePoint[2], 3);
    
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiDiscretePointInitializerListConstructorTest, "Schola.Points.MultiDiscretePoint.InitializerList Constructor Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiDiscretePointInitializerListConstructorTest::RunTest(const FString& Parameters)
{
    FMultiDiscretePoint DiscretePoint = FMultiDiscretePoint({ 1, 2, 3 });
    
    TestEqual(TEXT("DiscretePoint[0] == 1"), DiscretePoint[0], 1);
    TestEqual(TEXT("DiscretePoint[1] == 2"), DiscretePoint[1], 2);
    TestEqual(TEXT("DiscretePoint[2] == 3"), DiscretePoint[2], 3);
    
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiDiscretePointRawPointerConstructorTest, "Schola.Points.MultiDiscretePoint.RawPointer Constructor Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiDiscretePointRawPointerConstructorTest::RunTest(const FString& Parameters)
{
    TArray<int> Values = { 1, 2, 3 };
    FMultiDiscretePoint DiscretePoint = FMultiDiscretePoint(Values.GetData(), 3);
    
    TestEqual(TEXT("DiscretePoint[0] == 1"), DiscretePoint[0], 1);
    TestEqual(TEXT("DiscretePoint[1] == 2"), DiscretePoint[1], 2);
    TestEqual(TEXT("DiscretePoint[2] == 3"), DiscretePoint[2], 3);
    
    return true;
}

// Method Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiDiscretePointAddTest, "Schola.Points.MultiDiscretePoint.Add Test ", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiDiscretePointAddTest::RunTest(const FString& Parameters)
{
	FMultiDiscretePoint DiscretePoint = FMultiDiscretePoint();
    DiscretePoint.Add(1);
    DiscretePoint.Add(2);

    TestEqual(TEXT("DiscretePoint[0] == 1"), DiscretePoint[0], 1);
    TestEqual(TEXT("DiscretePoint[1] == 2"), DiscretePoint[1], 2);
    
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiDiscretePointResetTest, "Schola.Points.MultiDiscretePoint.Reset Test ", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiDiscretePointResetTest::RunTest(const FString& Parameters)
{
	FMultiDiscretePoint DiscretePoint = FMultiDiscretePoint();
    DiscretePoint.Add(1);
    DiscretePoint.Add(2);
    DiscretePoint.Reset();

    TestEqual(TEXT("DiscretePoint.Values.Num() == 0"), DiscretePoint.Values.Num(), 0);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiDiscretePointToStringTest, "Schola.Points.MultiDiscretePoint.ToString", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiDiscretePointToStringTest::RunTest(const FString& Parameters)
{
	FMultiDiscretePoint DiscretePoint;
	DiscretePoint.Values = {1, 2, 3};
	TestEqual(TEXT("FMultiDiscretePoint::ToString bracketed integers"), DiscretePoint.ToString(), FString(TEXT("[1, 2, 3]")));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiDiscretePointToStringEmptyTest, "Schola.Points.MultiDiscretePoint.ToString.Empty", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiDiscretePointToStringEmptyTest::RunTest(const FString& Parameters)
{
	const FMultiDiscretePoint DiscretePoint;
	TestEqual(TEXT("FMultiDiscretePoint::ToString empty is empty brackets"), DiscretePoint.ToString(), FString(TEXT("[]")));
	return true;
}

#endif