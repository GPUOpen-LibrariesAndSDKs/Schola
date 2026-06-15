// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "Spaces/BoxSpace.h"
#include "Points/BoxPoint.h"
#include "Points/MultiBinaryPoint.h"
#if WITH_DEV_AUTOMATION_TESTS
#define TestEqualExactFloat(TestMessage, Actual, Expected) TestEqual(TestMessage, (float)Actual, (float)Expected, 0.0001f)

// Constructor Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceDefaultTest, "Schola.Spaces.BoxSpace.Default Constructor Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceDefaultTest::RunTest(const FString& Parameters)
{
    FBoxSpace BoxSpace = FBoxSpace();

    TestEqual(TEXT("BoxSpace.Dimensions.Num() == 0"), BoxSpace.Dimensions.Num(), 0);
    TestEqual(TEXT("BoxSpace.Shape.Num() == 0"), BoxSpace.Shape.Num(), 0);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceLowHighArrayCreationTest, "Schola.Spaces.BoxSpace.TArrays Constructor Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceLowHighArrayCreationTest::RunTest(const FString& Parameters)
{
    TArray<float> Low = { -1.0, -2.0, -3.0 };
    TArray<float> High = { 1.0, 2.0, 3.0 };

    FBoxSpace BoxSpace = FBoxSpace(Low, High);

    TestEqual(TEXT("BoxSpace.Shape.Num() == 0"), BoxSpace.Shape.Num(), 0);

    const TArray<FBoxSpaceDimension> ExpectedDimensions = {
        FBoxSpaceDimension(-1.f, 1.f),
        FBoxSpaceDimension(-2.f, 2.f),
        FBoxSpaceDimension(-3.f, 3.f),
    };
    TestEqual(TEXT("BoxSpace.Dimensions"), BoxSpace.Dimensions, ExpectedDimensions);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceLowHighArrayWithShapeTest, "Schola.Spaces.BoxSpace.TArrays With Shape Constructor Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceLowHighArrayWithShapeTest::RunTest(const FString& Parameters)
{
    TArray<float> Low = { -1.0, -2.0, -3.0, -4.0, -5.0, -6.0 };
    TArray<float> High = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    TArray<int> Shape = { 2, 3 };

    FBoxSpace BoxSpace = FBoxSpace(Low, High, Shape);

    const TArray<FBoxSpaceDimension> ExpectedDimensions = {
        FBoxSpaceDimension(-1.f, 1.f),
        FBoxSpaceDimension(-2.f, 2.f),
        FBoxSpaceDimension(-3.f, 3.f),
        FBoxSpaceDimension(-4.f, 4.f),
        FBoxSpaceDimension(-5.f, 5.f),
        FBoxSpaceDimension(-6.f, 6.f),
    };
    TestEqual(TEXT("BoxSpace.Dimensions"), BoxSpace.Dimensions, ExpectedDimensions);
    TestEqual(TEXT("BoxSpace.Shape"), BoxSpace.Shape, Shape);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceDimensionArrayCreationTest, "Schola.Spaces.BoxSpace.DimensionArray Constructor Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceDimensionArrayCreationTest::RunTest(const FString& Parameters)
{
    TArray<FBoxSpaceDimension> Dimensions = { FBoxSpaceDimension(-1.0, 1.0), FBoxSpaceDimension(-2.0, 2.0), FBoxSpaceDimension(-3.0, 3.0) };

    FBoxSpace BoxSpace = FBoxSpace(Dimensions);

	TestEqual(TEXT("BoxSpace.Shape == {3}"), BoxSpace.Shape, TArray<int>({ 3 }));
    TestEqual(TEXT("BoxSpace.Dimensions"), BoxSpace.Dimensions, Dimensions);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceDimensionArrayWithShapeTest, "Schola.Spaces.BoxSpace.DimensionArrayWithShape Constructor Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceDimensionArrayWithShapeTest::RunTest(const FString& Parameters)
{
    TArray<FBoxSpaceDimension> Dimensions = { 
        FBoxSpaceDimension(-1.0, 1.0), 
        FBoxSpaceDimension(-2.0, 2.0), 
        FBoxSpaceDimension(-3.0, 3.0),
        FBoxSpaceDimension(-4.0, 4.0)
    };
    TArray<int> Shape = { 2, 2 };

    FBoxSpace BoxSpace = FBoxSpace(Dimensions, Shape);

    TestEqual(TEXT("BoxSpace.Dimensions"), BoxSpace.Dimensions, Dimensions);
    TestEqual(TEXT("BoxSpace.Shape"), BoxSpace.Shape, Shape);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceInitializerListsTest, "Schola.Spaces.BoxSpace.InitializerLists Constructor Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceInitializerListsTest::RunTest(const FString& Parameters)
{
    FBoxSpace BoxSpace = FBoxSpace({ -5.0f, -10.0f }, { 5.0f, 10.0f });

    TestEqual(TEXT("BoxSpace.Dimensions.Num() == 2"), BoxSpace.Dimensions.Num(), 2);

    //Create an array of shape [2] by default
    TestEqual(TEXT("BoxSpace.Shape"), BoxSpace.Shape, TArray<int>({2}));

    const TArray<FBoxSpaceDimension> ExpectedDimensions = {
        FBoxSpaceDimension(-5.f, 5.f),
        FBoxSpaceDimension(-10.f, 10.f),
    };
    TestEqual(TEXT("BoxSpace.Dimensions"), BoxSpace.Dimensions, ExpectedDimensions);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceInitializerListsWithShapeTest, "Schola.Spaces.BoxSpace.InitializerListsWithShape Constructor Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceInitializerListsWithShapeTest::RunTest(const FString& Parameters)
{
    FBoxSpace BoxSpace = FBoxSpace({ 0.0f, 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f, 1.0f }, { 2, 2 });

    TestEqual(TEXT("BoxSpace.Dimensions.Num() == 4"), BoxSpace.Dimensions.Num(), 4);
    TestEqual(TEXT("BoxSpace.Shape"), BoxSpace.Shape, TArray<int>({2, 2}));

    const TArray<FBoxSpaceDimension> ExpectedDimensions = {
        FBoxSpaceDimension(0.f, 1.f),
        FBoxSpaceDimension(0.f, 1.f),
        FBoxSpaceDimension(0.f, 1.f),
        FBoxSpaceDimension(0.f, 1.f),
    };
    TestEqual(TEXT("BoxSpace.Dimensions"), BoxSpace.Dimensions, ExpectedDimensions);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceShapeConstructorTest, "Schola.Spaces.BoxSpace.Shape Constructor Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceShapeConstructorTest::RunTest(const FString& Parameters)
{
    TArray<int> Shape = { 3, 4, 5 };
    FBoxSpace BoxSpace = FBoxSpace(Shape);

    TestEqual(TEXT("BoxSpace.Shape"), BoxSpace.Shape, Shape);
    
    // This constructor creates a space with preallocated shape but uninitialized dimensions
    TestEqual(TEXT("BoxSpace.Dimensions.Num() == 60"), BoxSpace.Dimensions.Num(), 60); // 3*4*5 = 60

    return true;
}

// Method Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceAddDimensionTest, "Schola.Spaces.BoxSpace.Add Dimension Test ", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceAddDimensionTest::RunTest(const FString& Parameters)
{
    FBoxSpace BoxSpace = FBoxSpace();

    BoxSpace.Add(-1.0, 1.0);
    BoxSpace.Add(FBoxSpaceDimension(-2.0, 2.0));
    BoxSpace.Add(-3.0, 3.0);

    const TArray<FBoxSpaceDimension> ExpectedDimensions = {
        FBoxSpaceDimension(-1.f, 1.f),
        FBoxSpaceDimension(-2.f, 2.f),
        FBoxSpaceDimension(-3.f, 3.f),
    };
    TestEqual(TEXT("BoxSpace.Dimensions"), BoxSpace.Dimensions, ExpectedDimensions);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceNormalizedObservationSpaceTest, "Schola.Spaces.BoxSpace.Get Normalized Observation Space Test ", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceNormalizedObservationSpaceTest::RunTest(const FString& Parameters)
{
    FBoxSpace BoxSpace = FBoxSpace();
    BoxSpace.Add(-1.0, 1.0);
    BoxSpace.Add(-2.0, 2.0);
    BoxSpace.Add(-3.0, 3.0);

    FBoxSpace NormalizedBoxSpace = BoxSpace.GetNormalizedObservationSpace();

    const TArray<FBoxSpaceDimension> ExpectedNormalized = {
        FBoxSpaceDimension(0.f, 1.f),
        FBoxSpaceDimension(0.f, 1.f),
        FBoxSpaceDimension(0.f, 1.f),
    };
    TestEqual(TEXT("NormalizedBoxSpace.Dimensions"), NormalizedBoxSpace.Dimensions, ExpectedNormalized);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceFlattenedSizeTest, "Schola.Spaces.BoxSpace.Get Flattened Size Test ", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceFlattenedSizeTest::RunTest(const FString& Parameters)
{
    FBoxSpace BoxSpace = FBoxSpace();
    BoxSpace.Add(-1.0, 1.0);
    BoxSpace.Add(-2.0, 2.0);
    BoxSpace.Add(-3.0, 3.0);

    TestEqual(TEXT("BoxSpace.GetFlattenedSize() == 3"), BoxSpace.GetFlattenedSize(), 3);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceNormalizeObservationTest, "Schola.Spaces.BoxSpace.Normalize Observation Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceNormalizeObservationTest::RunTest(const FString& Parameters)
{
    FBoxSpace BoxSpace = FBoxSpace();
    BoxSpace.Add(0, 1.0);
    BoxSpace.Add(-2.0, 2.0);
    BoxSpace.Add(-3.0, 3.0);

    FBoxPoint BoxPoint = FBoxPoint();
    BoxPoint.Add(0.0);
    BoxPoint.Add(0.0);
    BoxPoint.Add(0.0);

    FBoxPoint NormalizedBoxPoint = BoxSpace.NormalizeObservation(BoxPoint);

    TestEqual(TEXT("NormalizedBoxPoint.Values"), NormalizedBoxPoint.Values, TArray<float>({0.f, 0.5f, 0.5f}));

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceValidateSuccessTest, "Schola.Spaces.BoxSpace.Validate Success Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceValidateSuccessTest::RunTest(const FString& Parameters)
{
    FBoxSpace BoxSpace = FBoxSpace();
    BoxSpace.Add(-1.0, 1.0);
    BoxSpace.Add(-2.0, 2.0);
    BoxSpace.Add(-3.0, 3.0);

    FBoxPoint BoxPoint = FBoxPoint({0.0f, 1.0f, 2.0f});
    TInstancedStruct<FPoint> Point = TInstancedStruct<FPoint>::Make<FBoxPoint>(BoxPoint);

    ESpaceValidationResult Result = BoxSpace.Validate(Point);
    TestTrue(TEXT("BoxSpace.Validate(Point) == ESpaceValidationResult::Success"), Result == ESpaceValidationResult::Success);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceValidateWrongDimensionsTest, "Schola.Spaces.BoxSpace.Validate Wrong Dimensions Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceValidateWrongDimensionsTest::RunTest(const FString& Parameters)
{
    FBoxSpace BoxSpace = FBoxSpace();
    BoxSpace.Add(-1.0, 1.0);
    BoxSpace.Add(-2.0, 2.0);

    FBoxPoint BoxPoint = FBoxPoint({0.0f});
    TInstancedStruct<FPoint> Point = TInstancedStruct<FPoint>::Make<FBoxPoint>(BoxPoint);

    ESpaceValidationResult Result = BoxSpace.Validate(Point);
    TestTrue(TEXT("BoxSpace.Validate(Point) == ESpaceValidationResult::WrongDimensions"), Result == ESpaceValidationResult::WrongDimensions);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceValidateWrongDimensionsShapeMismatchTest, "Schola.Spaces.BoxSpace.Validate Wrong Dimensions When Shape Mismatch Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceValidateWrongDimensionsShapeMismatchTest::RunTest(const FString& Parameters)
{
    TArray<float> Low = { 0.0f, 0.0f};
    TArray<float> High = { 1.0f, 1.0f};
    FBoxSpace BoxSpace = FBoxSpace(Low, High, { 1, 2 });

    TArray<float> Values = { 0.0f, 0.0f};
    FBoxPoint BoxPoint = FBoxPoint(Values, { 2, 1});
    TInstancedStruct<FPoint> Point = TInstancedStruct<FPoint>::Make<FBoxPoint>(BoxPoint);

    ESpaceValidationResult Result = BoxSpace.Validate(Point);
    TestTrue(TEXT("BoxSpace.Validate(Point) == ESpaceValidationResult::WrongDimensions when space shape {1,2} and point shape {2,1}"), Result == ESpaceValidationResult::WrongDimensions);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceValidateOutOfBoundsTest, "Schola.Spaces.BoxSpace.Validate Out Of Bounds Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceValidateOutOfBoundsTest::RunTest(const FString& Parameters)
{
    FBoxSpace BoxSpace = FBoxSpace();
    BoxSpace.Add(-1.0, 1.0);
    BoxSpace.Add(-2.0, 2.0);

    FBoxPoint BoxPoint = FBoxPoint({2.0f, 0.0f});
    TInstancedStruct<FPoint> Point = TInstancedStruct<FPoint>::Make<FBoxPoint>(BoxPoint);

    ESpaceValidationResult Result = BoxSpace.Validate(Point);
    TestTrue(TEXT("BoxSpace.Validate(Point) == ESpaceValidationResult::OutOfBounds"), Result == ESpaceValidationResult::OutOfBounds);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceValidateWrongDataTypeTest, "Schola.Spaces.BoxSpace.Validate Wrong Type of Point Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceValidateWrongDataTypeTest::RunTest(const FString& Parameters)
{
    FBoxSpace BoxSpace = FBoxSpace();
    BoxSpace.Add(-1.0, 1.0);
    BoxSpace.Add(-2.0, 2.0);
    BoxSpace.Add(-3.0, 3.0);

    TInstancedStruct<FPoint> Point;
    Point.InitializeAs<FMultiBinaryPoint>();

    ESpaceValidationResult Result = BoxSpace.Validate(Point);

    TestTrue(TEXT("BoxSpace.Validate(Point) == ESpaceValidationResult::WrongDataType"), Result == ESpaceValidationResult::WrongDataType);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceToStringNoShapeTest, "Schola.Spaces.BoxSpace.ToString.NoShape", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceToStringNoShapeTest::RunTest(const FString& Parameters)
{
	FBoxSpace BoxSpace;
	BoxSpace.Add(-1.0f, 1.0f);
	BoxSpace.Add(-2.0f, 2.0f);

	TestEqual(
		TEXT("FBoxSpace::ToString without shape"),
		BoxSpace.ToString(),
		FString(TEXT("BoxSpace(Dimensions={[-1, 1], [-2, 2]})")));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceToStringWithShapeTest, "Schola.Spaces.BoxSpace.ToString.WithShape", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceToStringWithShapeTest::RunTest(const FString& Parameters)
{
	TArray<float> Low = {-1.0f, -2.0f};
	TArray<float> High = {1.0f, 2.0f};
	TArray<int> Shape = {2};
	FBoxSpace BoxSpace(Low, High, Shape);

	TestEqual(
		TEXT("FBoxSpace::ToString with shape"),
		BoxSpace.ToString(),
		FString(TEXT("BoxSpace(Dimensions={[-1, 1], [-2, 2]}, Shape=[2])")));
	return true;
}
#endif