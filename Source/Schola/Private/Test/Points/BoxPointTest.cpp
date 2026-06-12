// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Points/BoxPoint.h"

#if WITH_AUTOMATION_TESTS

#define TestEqualExactFloat(TestMessage, Actual, Expected) TestEqual(TestMessage, (float)Actual, (float)Expected, 0.0001f)

// Constructor Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxPointDefaultConstructorTest, "Schola.Points.BoxPoint.Constructor.Default", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxPointDefaultConstructorTest::RunTest(const FString& Parameters)
{
    FBoxPoint BoxPoint = FBoxPoint();

    TestEqual(TEXT("BoxPoint.Values.Num() == 0"), BoxPoint.Values.Num(), 0);
    TestEqual(TEXT("BoxPoint.Shape.Num() == 0"), BoxPoint.Shape.Num(), 0);
    
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxPointTArrayConstructorTest, "Schola.Points.BoxPoint.TArray Constructor Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxPointTArrayConstructorTest::RunTest(const FString& Parameters)
{
    TArray<float> Values = { 1.0f, 2.0f, 3.0f, 4.0f };
    FBoxPoint BoxPoint = FBoxPoint(Values);

    TestEqual(TEXT("BoxPoint.Values"), BoxPoint.Values, Values);
    TestEqual(TEXT("BoxPoint.Shape"), BoxPoint.Shape, TArray<int>({4}));
    
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxPointInitializerListConstructorTest, "Schola.Points.BoxPoint.InitializerList Constructor Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxPointInitializerListConstructorTest::RunTest(const FString& Parameters)
{
    FBoxPoint BoxPoint = FBoxPoint({ 5.0f, 10.0f, 15.0f });

    TestEqual(TEXT("BoxPoint.Values"), BoxPoint.Values, TArray<float>({5.0f, 10.0f, 15.0f}));
    TestEqual(TEXT("BoxPoint.Shape"), BoxPoint.Shape, TArray<int>({3}));
    
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxPointTArrayWithShapeConstructorTest, "Schola.Points.BoxPoint.TArrayWithShape Constructor Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxPointTArrayWithShapeConstructorTest::RunTest(const FString& Parameters)
{
    TArray<float> Values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    TArray<int> Shape = { 2, 3 };
    FBoxPoint BoxPoint = FBoxPoint(Values, Shape);

    TestEqual(TEXT("BoxPoint.Values"), BoxPoint.Values, Values);
    TestEqual(TEXT("BoxPoint.Shape"), BoxPoint.Shape, Shape);
    
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxPointInvalidShapeTest, "Schola.Points.BoxPoint.Shape Test ", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxPointInvalidShapeTest::RunTest(const FString& Parameters)
{
    // TODO raise an error here potentially
    TArray<float> Values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    TArray<int> Shape = { 2, 3 };
    FBoxPoint BoxPoint = FBoxPoint(Values, Shape);
    
    return true;
}


IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxPointFromArrayTest, "Schola.Points.BoxPoint.Constructor.RawPointer", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxPointFromArrayTest::RunTest(const FString& Parameters)
{
    TArray<float> Values = { 1.0f, 2.0f, 3.0f };
	FBoxPoint BoxPoint = FBoxPoint(Values.GetData(),3);

    TestEqual(TEXT("BoxPoint.Values"), BoxPoint.Values, Values);
    TestEqual(TEXT("BoxPoint.Shape"), BoxPoint.Shape, TArray<int>({3}));
    
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxPointPreallocationConstructorTest, "Schola.Points.BoxPoint.Constructor.Preallocation", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxPointPreallocationConstructorTest::RunTest(const FString& Parameters)
{
    FBoxPoint BoxPoint = FBoxPoint(5);

    TestEqual(TEXT("BoxPoint.Values has 5 slack space"), BoxPoint.Values.GetSlack(), 5);
    
	return true;
}

// Method Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxPointAddTest, "Schola.Points.BoxPoint.Add Test ", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxPointAddTest::RunTest(const FString& Parameters)
{
    FBoxPoint BoxPoint = FBoxPoint();
    BoxPoint.Add(1.0f);
    BoxPoint.Add(2.0f);

    TestEqual(TEXT("BoxPoint.Values"), BoxPoint.Values, TArray<float>({1.0f, 2.0f}));
    
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxPointResetTest, "Schola.Points.BoxPoint.Reset Test ", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxPointResetTest::RunTest(const FString& Parameters)
{
    FBoxPoint BoxPoint = FBoxPoint();
    BoxPoint.Add(1.0f);
    BoxPoint.Add(2.0f);
    BoxPoint.Reset();

    TestEqual(TEXT("BoxPoint.Values.Num() == 0"), BoxPoint.Values.Num(), 0);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxPointToStringTest, "Schola.Points.BoxPoint.ToString", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxPointToStringTest::RunTest(const FString& Parameters)
{
	const FBoxPoint BoxPoint({1.5f, 2.0f});
	TestEqual(TEXT("FBoxPoint::ToString bracketed SanitizeFloat values"), BoxPoint.ToString(), FString(TEXT("[1.5, 2.0]")));
	return true;
}


#endif