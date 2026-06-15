// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "Points/MultiBinaryPoint.h"
#if WITH_AUTOMATION_TESTS

// Constructor Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiBinaryPointDefaultConstructorTest, "Schola.Points.MultiBinaryPoint.Constructor.Default", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiBinaryPointDefaultConstructorTest::RunTest(const FString& Parameters)
{
	FMultiBinaryPoint MultiBinaryPoint = FMultiBinaryPoint();

    TestEqual(TEXT("MultiBinaryPoint.Values.Num() == 0"), MultiBinaryPoint.Values.Num(), 0);
    
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiBinaryPointFromArrayTest, "Schola.Points.MultiBinaryPoint.Constructor.TArray", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiBinaryPointFromArrayTest::RunTest(const FString& Parameters)
{
    TArray<bool> Values = { true, false, true };
	FMultiBinaryPoint MultiBinaryPoint = FMultiBinaryPoint(Values);

    TestEqual(TEXT("MultiBinaryPoint.Values"), MultiBinaryPoint.Values, Values);
    
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiBinaryPointInitializerListConstructorTest, "Schola.Points.MultiBinaryPoint.InitializerList Constructor Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiBinaryPointInitializerListConstructorTest::RunTest(const FString& Parameters)
{
	FMultiBinaryPoint MultiBinaryPoint = FMultiBinaryPoint({ false, true, false, true });

    TestEqual(TEXT("MultiBinaryPoint.Values"), MultiBinaryPoint.Values, TArray<bool>({false, true, false, true}));
    
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiBinaryPointRawPointerConstructorTest, "Schola.Points.MultiBinaryPoint.RawPointer Constructor Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiBinaryPointRawPointerConstructorTest::RunTest(const FString& Parameters)
{
    TArray<bool> Values = { true, true, false };
	FMultiBinaryPoint MultiBinaryPoint = FMultiBinaryPoint(Values.GetData(), 3);

    TestEqual(TEXT("MultiBinaryPoint.Values"), MultiBinaryPoint.Values, Values);
    
    return true;
}

// Method Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiBinaryPointAddTest, "Schola.Points.MultiBinaryPoint.Add Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiBinaryPointAddTest::RunTest(const FString& Parameters)
{
	FMultiBinaryPoint MultiBinaryPoint = FMultiBinaryPoint();
    MultiBinaryPoint.Add(true);
    MultiBinaryPoint.Add(false);

    TestEqual(TEXT("MultiBinaryPoint.Values"), MultiBinaryPoint.Values, TArray<bool>({true, false}));
    
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiBinaryPointResetTest, "Schola.Points.MultiBinaryPoint.Reset Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiBinaryPointResetTest::RunTest(const FString& Parameters)
{
	FMultiBinaryPoint MultiBinaryPoint = FMultiBinaryPoint();
    MultiBinaryPoint.Add(true);
    MultiBinaryPoint.Add(false);
    MultiBinaryPoint.Reset();

    TestEqual(TEXT("MultiBinaryPoint.Values.Num() == 0"), MultiBinaryPoint.Values.Num(), 0);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiBinaryPointToStringWithValuesTest, "Schola.Points.MultiBinaryPoint.ToString.WithValues", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiBinaryPointToStringWithValuesTest::RunTest(const FString& Parameters)
{
	FMultiBinaryPoint MultiBinaryPoint;
	MultiBinaryPoint.Values = {true, false, true};
	TestEqual(TEXT("FMultiBinaryPoint::ToString lists bools as true/false"), MultiBinaryPoint.ToString(), FString(TEXT("[true, false, true]")));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiBinaryPointToStringEmptyTest, "Schola.Points.MultiBinaryPoint.ToString.Empty", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiBinaryPointToStringEmptyTest::RunTest(const FString& Parameters)
{
	const FMultiBinaryPoint MultiBinaryPoint;
	TestEqual(TEXT("FMultiBinaryPoint::ToString with no values"), MultiBinaryPoint.ToString(), FString(TEXT("[]")));
	return true;
}


#endif
