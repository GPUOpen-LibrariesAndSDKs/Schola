// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Spaces/MultiDiscreteSpace.h"
#include "Points/MultiBinaryPoint.h"
#include "Points/MultiDiscretePoint.h"
#if WITH_DEV_AUTOMATION_TESTS
#define TestEqualExactFloat(TestMessage, Actual, Expected) TestEqual(TestMessage, (float)Actual, (float)Expected, 0.0001f)

// Constructor Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiDiscreteSpaceDefaultTest, "Schola.Spaces.MultiDiscreteSpace.Constructor.Default", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiDiscreteSpaceDefaultTest::RunTest(const FString& Parameters)
{
    FMultiDiscreteSpace DiscreteSpace = FMultiDiscreteSpace();

    TestEqualExactFloat(TEXT("DiscreteSpace.High.Num() == 0"), DiscreteSpace.High.Num(), 0);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiDiscreteSpaceHighArrayCreationTest, "Schola.Spaces.MultiDiscreteSpace.Constructor.TArray", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiDiscreteSpaceHighArrayCreationTest::RunTest(const FString& Parameters)
{
    TArray<int> High = { 1, 2, 3 };

    FMultiDiscreteSpace DiscreteSpace = FMultiDiscreteSpace(High);

    TestEqual(TEXT("DiscreteSpace.High.Num() == 3"), DiscreteSpace.High.Num(), 3);

    //Check that the High Values Match
	TestEqualExactFloat(TEXT("DiscreteSpace.High[0] == 1"), DiscreteSpace.High[0], 1);
	TestEqualExactFloat(TEXT("DiscreteSpace.High[1] == 2"), DiscreteSpace.High[1], 2);
	TestEqualExactFloat(TEXT("DiscreteSpace.High[2] == 3"), DiscreteSpace.High[2], 3);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiDiscreteSpaceRawPointerConstructorTest, "Schola.Spaces.MultiDiscreteSpace.Constructor.RawPointer", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiDiscreteSpaceRawPointerConstructorTest::RunTest(const FString& Parameters)
{
    TArray<int> High = { 5, 10, 15, 20 };
    FMultiDiscreteSpace DiscreteSpace = FMultiDiscreteSpace(High.GetData(), 4);

    TestEqual(TEXT("DiscreteSpace.High.Num() == 4"), DiscreteSpace.High.Num(), 4);
    TestEqualExactFloat(TEXT("DiscreteSpace.High[0] == 5"), DiscreteSpace.High[0], 5);
	TestEqualExactFloat(TEXT("DiscreteSpace.High[1] == 10"), DiscreteSpace.High[1], 10);
	TestEqualExactFloat(TEXT("DiscreteSpace.High[2] == 15"), DiscreteSpace.High[2], 15);
	TestEqualExactFloat(TEXT("DiscreteSpace.High[3] == 20"), DiscreteSpace.High[3], 20);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiDiscreteSpaceCopyConstructorTest, "Schola.Spaces.MultiDiscreteSpace.Constructor.Copy", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiDiscreteSpaceCopyConstructorTest::RunTest(const FString& Parameters)
{
    FMultiDiscreteSpace OriginalSpace = FMultiDiscreteSpace();
    OriginalSpace.Add(3);
    OriginalSpace.Add(7);
    OriginalSpace.Add(11);

    FMultiDiscreteSpace CopiedSpace = FMultiDiscreteSpace(OriginalSpace);

    TestEqual(TEXT("CopiedSpace.High.Num() == 3"), CopiedSpace.High.Num(), 3);
    TestEqualExactFloat(TEXT("CopiedSpace.High[0] == 3"), CopiedSpace.High[0], 3);
	TestEqualExactFloat(TEXT("CopiedSpace.High[1] == 7"), CopiedSpace.High[1], 7);
	TestEqualExactFloat(TEXT("CopiedSpace.High[2] == 11"), CopiedSpace.High[2], 11);

    return true;
}

// Method Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiDiscreteSpaceAddTest, "Schola.Spaces.MultiDiscreteSpace.Add Test ", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiDiscreteSpaceAddTest::RunTest(const FString& Parameters)
{
    FMultiDiscreteSpace DiscreteSpace = FMultiDiscreteSpace();
    DiscreteSpace.Add(1);
    DiscreteSpace.Add(2);

    TestEqualExactFloat(TEXT("DiscreteSpace.High.Num() == 2"), DiscreteSpace.High.Num(), 2);

    //Check that the High Values Match
	TestEqualExactFloat(TEXT("DiscreteSpace.High[0] == 1"), DiscreteSpace.High[0], 1);
	TestEqualExactFloat(TEXT("DiscreteSpace.High[1] == 2"), DiscreteSpace.High[1], 2);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiDiscreteSpaceMergeTest, "Schola.Spaces.MultiDiscreteSpace.Merge Test ", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiDiscreteSpaceMergeTest::RunTest(const FString& Parameters)
{
    FMultiDiscreteSpace DiscreteSpace = FMultiDiscreteSpace();
    DiscreteSpace.Add(1);
    DiscreteSpace.Add(2);

    FMultiDiscreteSpace OtherDiscreteSpace = FMultiDiscreteSpace();
    OtherDiscreteSpace.Add(3);
    OtherDiscreteSpace.Add(4);

    DiscreteSpace.Merge(OtherDiscreteSpace);

    TestEqualExactFloat(TEXT("DiscreteSpace.High.Num() == 4"), DiscreteSpace.High.Num(), 4);

    //Check that the High Values Match
	TestEqualExactFloat(TEXT("DiscreteSpace.High[0] == 1"), DiscreteSpace.High[0], 1);
    TestEqualExactFloat(TEXT("DiscreteSpace.High[1] == 2"), DiscreteSpace.High[1], 2);
    TestEqualExactFloat(TEXT("DiscreteSpace.High[2] == 3"), DiscreteSpace.High[2], 3);
    TestEqualExactFloat(TEXT("DiscreteSpace.High[3] == 4"), DiscreteSpace.High[3], 4);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiDiscreteSpaceFlattenedSizeTest, "Schola.Spaces.MultiDiscreteSpace.FlattenedSize Test ", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiDiscreteSpaceFlattenedSizeTest::RunTest(const FString& Parameters)
{
    FMultiDiscreteSpace DiscreteSpace = FMultiDiscreteSpace();
    DiscreteSpace.Add(1);
    DiscreteSpace.Add(2);

    TestEqualExactFloat(TEXT("DiscreteSpace.GetFlattenedSize() == 3"), DiscreteSpace.GetFlattenedSize(), 3);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiDiscreteSpaceIsEmptyTrueTest, "Schola.Spaces.MultiDiscreteSpace.Is Empty True Test ", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiDiscreteSpaceIsEmptyTrueTest::RunTest(const FString& Parameters)
{
    FMultiDiscreteSpace DiscreteSpace = FMultiDiscreteSpace();

    TestTrue(TEXT("DiscreteSpace.IsEmpty() == true"), DiscreteSpace.IsEmpty());

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiDiscreteSpaceIsEmptyFalseTest, "Schola.Spaces.MultiDiscreteSpace.Is Empty False Test ", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiDiscreteSpaceIsEmptyFalseTest::RunTest(const FString& Parameters)
{
    FMultiDiscreteSpace DiscreteSpace = FMultiDiscreteSpace();
    DiscreteSpace.Add(1);

    TestTrue(TEXT("DiscreteSpace.IsEmpty() == false"), !DiscreteSpace.IsEmpty());

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiDiscreteSpaceGetNumDimensionsTest, "Schola.Spaces.MultiDiscreteSpace.Get Num Dimensions Test ", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiDiscreteSpaceGetNumDimensionsTest::RunTest(const FString& Parameters)
{
    FMultiDiscreteSpace DiscreteSpace = FMultiDiscreteSpace();
    DiscreteSpace.Add(1);
    DiscreteSpace.Add(2);

    TestEqualExactFloat(TEXT("DiscreteSpace.GetNumDimensions() == 2"), DiscreteSpace.GetNumDimensions(), 2);

    return true;
}

//Validation Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiDiscreteSpaceValidateSuccessTest, "Schola.Spaces.MultiDiscreteSpace.Validate Success Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiDiscreteSpaceValidateSuccessTest::RunTest(const FString& Parameters)
{
    FMultiDiscreteSpace DiscreteSpace = FMultiDiscreteSpace({3, 5});

    FMultiDiscretePoint MultiDiscretePoint = FMultiDiscretePoint({1, 2});
    TInstancedStruct<FPoint> Point = TInstancedStruct<FPoint>::Make<FMultiDiscretePoint>(MultiDiscretePoint);

    TestTrue(TEXT("MultiDiscreteSpace.Validate(Point) == ESpaceValidationResult::Success"), DiscreteSpace.Validate(Point) == ESpaceValidationResult::Success);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiDiscreteSpaceValidateWrongDimensionsTest, "Schola.Spaces.MultiDiscreteSpace.Validate Wrong Dimensions Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiDiscreteSpaceValidateWrongDimensionsTest::RunTest(const FString& Parameters)
{
    FMultiDiscreteSpace DiscreteSpace = FMultiDiscreteSpace({3, 5});

    FMultiDiscretePoint MultiDiscretePoint = FMultiDiscretePoint({1});
    TInstancedStruct<FPoint> Point = TInstancedStruct<FPoint>::Make<FMultiDiscretePoint>(MultiDiscretePoint);

    TestEqual(TEXT("MultiDiscreteSpace.Validate(Point) == ESpaceValidationResult::WrongDimensions"), DiscreteSpace.Validate(Point), ESpaceValidationResult::WrongDimensions);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiDiscreteSpaceValidateOutOfBoundsTest, "Schola.Spaces.MultiDiscreteSpace.Validate Out Of Bounds Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiDiscreteSpaceValidateOutOfBoundsTest::RunTest(const FString& Parameters)
{
    FMultiDiscreteSpace DiscreteSpace = FMultiDiscreteSpace({3, 5});

    FMultiDiscretePoint MultiDiscretePoint = FMultiDiscretePoint({5, 4});
    TInstancedStruct<FPoint> Point = TInstancedStruct<FPoint>::Make<FMultiDiscretePoint>(MultiDiscretePoint);

    TestTrue(TEXT("MultiDiscreteSpace.Validate(Point) == ESpaceValidationResult::OutOfBounds"), DiscreteSpace.Validate(Point) == ESpaceValidationResult::OutOfBounds);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiDiscreteSpaceValidateWrongDataTypeTest, "Schola.Spaces.MultiDiscreteSpace.Validate Wrong Data Type Test ", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiDiscreteSpaceValidateWrongDataTypeTest::RunTest(const FString& Parameters)
{
    FMultiDiscreteSpace DiscreteSpace = FMultiDiscreteSpace();
    DiscreteSpace.Add(1);
    DiscreteSpace.Add(2);

    TInstancedStruct<FPoint> Point;
    Point.InitializeAs<FMultiBinaryPoint>();

    TestTrue(TEXT("DiscreteSpace.Validate(Point) == ESpaceValidationResult::WrongDataType"), DiscreteSpace.Validate(Point) == ESpaceValidationResult::WrongDataType);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiDiscreteSpaceGetMaxValueTest, "Schola.Spaces.MultiDiscreteSpace.Get Max Value Test ", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiDiscreteSpaceGetMaxValueTest::RunTest(const FString& Parameters)
{
    FMultiDiscreteSpace DiscreteSpace = FMultiDiscreteSpace();

    TArray<float> Vector = { 0.0, 1.0, 3.0 };

    TestEqualExactFloat(TEXT("DiscreteSpace.GetMaxValue(Vector) == 2"), DiscreteSpace.GetMaxValue(Vector), 2);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMultiDiscreteSpaceToStringTest, "Schola.Spaces.MultiDiscreteSpace.ToString", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMultiDiscreteSpaceToStringTest::RunTest(const FString& Parameters)
{
	FMultiDiscreteSpace DiscreteSpace(TArray<int>{2, 3, 4});
	TestEqual(TEXT("FMultiDiscreteSpace::ToString"), DiscreteSpace.ToString(), FString(TEXT("MultiDiscreteSpace(High=[2, 3, 4])")));
	return true;
}
#endif
