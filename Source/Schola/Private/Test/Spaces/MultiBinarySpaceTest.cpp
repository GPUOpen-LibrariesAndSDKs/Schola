// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "Spaces/MultiBinarySpace.h"
#include "Spaces/Space.h"
#include "Points/MultiBinaryPoint.h"
#include "Points/DiscretePoint.h"

#if WITH_AUTOMATION_TESTS

// Constructor Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiBinarySpaceDefaultTest, "Schola.Spaces.MultiBinarySpace.Default Constructor Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiBinarySpaceDefaultTest::RunTest(const FString& Parameters)
{
    FMultiBinarySpace MultiBinarySpace = FMultiBinarySpace();

    TestEqual(TEXT("MultiBinarySpace.Shape == 0"), MultiBinarySpace.Shape, 0);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiBinarySpaceShapeTest, "Schola.Spaces.MultiBinarySpace.Int Constructor Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiBinarySpaceShapeTest::RunTest(const FString& Parameters)
{
    FMultiBinarySpace MultiBinarySpace = FMultiBinarySpace(10);

    TestEqual(TEXT("MultiBinarySpace.Shape == 10"), MultiBinarySpace.Shape, 10);

    return true;
}

// Method Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiBinarySpaceMergeTest, "Schola.Spaces.MultiBinarySpace.Merge Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiBinarySpaceMergeTest::RunTest(const FString& Parameters)
{
    FMultiBinarySpace MultiBinarySpace = FMultiBinarySpace(10);
    FMultiBinarySpace OtherMultiBinarySpace = FMultiBinarySpace(5);

    MultiBinarySpace.Merge(OtherMultiBinarySpace);

    TestEqual(TEXT("MultiBinarySpace.Shape == 15"), MultiBinarySpace.Shape, 15);

    return true;
}

// Validate Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiBinarySpaceValidateSuccessTest, "Schola.Spaces.MultiBinarySpace.Validate Success Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiBinarySpaceValidateSuccessTest::RunTest(const FString& Parameters)
{
    FMultiBinarySpace MultiBinarySpace = FMultiBinarySpace(3);

    FMultiBinaryPoint MultiBinaryPoint = FMultiBinaryPoint({ true, false, true });
    TInstancedStruct<FPoint> Point = TInstancedStruct<FPoint>::Make<FMultiBinaryPoint>(MultiBinaryPoint);

    TestEqual(TEXT("MultiBinarySpace.Validate(Point) == ESpaceValidationResult::Success"), MultiBinarySpace.Validate(Point), ESpaceValidationResult::Success);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiBinarySpaceValidateWrongDataTypeTest, "Schola.Spaces.MultiBinarySpace.Validate Wrong Data Type Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiBinarySpaceValidateWrongDataTypeTest::RunTest(const FString& Parameters)
{
    FMultiBinarySpace MultiBinarySpace = FMultiBinarySpace(3);

    TInstancedStruct<FPoint> Point;
    Point.InitializeAs<FDiscretePoint>();

    TestEqual(TEXT("MultiBinarySpace.Validate(Point) == ESpaceValidationResult::WrongDataType"), MultiBinarySpace.Validate(Point), ESpaceValidationResult::WrongDataType);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiBinarySpaceValidateWrongDimensionsTest, "Schola.Spaces.MultiBinarySpace.Validate Wrong Dimensions Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiBinarySpaceValidateWrongDimensionsTest::RunTest(const FString& Parameters)
{
    FMultiBinarySpace MultiBinarySpace = FMultiBinarySpace(3);

    FMultiBinaryPoint MultiBinaryPoint = FMultiBinaryPoint({ true, false });
    TInstancedStruct<FPoint> Point = TInstancedStruct<FPoint>::Make<FMultiBinaryPoint>(MultiBinaryPoint);

    TestEqual(TEXT("MultiBinarySpace.Validate(Point) == ESpaceValidationResult::WrongDimensions"), MultiBinarySpace.Validate(Point), ESpaceValidationResult::WrongDimensions);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiBinarySpaceToStringTest, "Schola.Spaces.MultiBinarySpace.ToString", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiBinarySpaceToStringTest::RunTest(const FString& Parameters)
{
	FMultiBinarySpace MultiBinarySpace;
	MultiBinarySpace.Shape = 8;
	TestEqual(TEXT("FMultiBinarySpace::ToString"), MultiBinarySpace.ToString(), FString(TEXT("MultiBinarySpace(Shape=8)")));
	return true;
}
#endif
