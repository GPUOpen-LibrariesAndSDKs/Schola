// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Points/DiscretePoint.h"

#if WITH_AUTOMATION_TESTS

// Constructor Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDiscretePointDefaultConstructorTest, "Schola.Points.DiscretePoint.Default Constructor Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FDiscretePointDefaultConstructorTest::RunTest(const FString& Parameters)
{
	FDiscretePoint DiscretePoint = FDiscretePoint();

    TestEqual(TEXT("DiscretePoint.Value == 0"), DiscretePoint.Value, 0);
    
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDiscretePointCreationTest, "Schola.Points.DiscretePoint.IntValue Constructor Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FDiscretePointCreationTest::RunTest(const FString& Parameters)
{
	FDiscretePoint DiscretePoint = FDiscretePoint(3);

    TestEqual(TEXT("DiscretePoint.Value == 3"), DiscretePoint.Value, 3);
    
    return true;
}

// Method Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDiscretePointResetTest, "Schola.Points.DiscretePoint.Reset Test ", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FDiscretePointResetTest::RunTest(const FString& Parameters)
{
	FDiscretePoint DiscretePoint = FDiscretePoint(2);
	DiscretePoint.Reset();
    TestEqual(TEXT("DiscretePoint.Value == 0"), DiscretePoint.Value, 0);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDiscretePointToStringTest, "Schola.Points.DiscretePoint.ToString", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FDiscretePointToStringTest::RunTest(const FString& Parameters)
{
	const FDiscretePoint DiscretePoint(42);
	TestEqual(TEXT("FDiscretePoint::ToString formats integer value"), DiscretePoint.ToString(), FString(TEXT("42")));
	return true;
}

#endif