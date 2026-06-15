// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "Spaces/DictSpace.h"
#include "Spaces/DiscreteSpace.h"
#include "Spaces/BoxSpace.h"
#include "Spaces/Space.h"
#include "Points/BoxPoint.h"
#include "Points/DictPoint.h"
#include "Points/DiscretePoint.h"

#if WITH_DEV_AUTOMATION_TESTS
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDictSpaceDefaultTest, "Schola.Spaces.DictSpace.Default Creation Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FDictSpaceDefaultTest::RunTest(const FString& Parameters)
{
	FDictSpace DictSpace = FDictSpace();

	TestEqual(TEXT("DictSpace.Spaces.Num() == 0"), DictSpace.Spaces.Num(), 0);
	TestEqual(TEXT("DictSpace.Num() == 0"), DictSpace.Num(), 0);
	TestEqual(TEXT("DictSpace.GetFlattenedSize() == 0"), DictSpace.GetFlattenedSize(), 0);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDictSpaceNumAndGetFlattenedSizeTest, "Schola.Spaces.DictSpace.Num and GetFlattenedSize Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FDictSpaceNumAndGetFlattenedSizeTest::RunTest(const FString& Parameters)
{
	FDictSpace DictSpace = FDictSpace();

	FBoxSpace BoxSpace1;
	BoxSpace1.Add(-1.0, 1.0);

	FBoxSpace BoxSpace2;
	BoxSpace2.Add(-2.0, 2.0);
	BoxSpace2.Add(-3.0, 3.0);
	

	DictSpace.Spaces.Add("box1", TInstancedStruct<FSpace>::Make<FBoxSpace>(BoxSpace1));
	DictSpace.Spaces.Add("box2", TInstancedStruct<FSpace>::Make<FBoxSpace>(BoxSpace2));

	TestEqual(TEXT("DictSpace.Num() == 2"), DictSpace.Num(), 2);
	TestEqual(TEXT("DictSpace.GetFlattenedSize() == 3"), DictSpace.GetFlattenedSize(), 3);

	return true;
}

// Validate Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDictSpaceValidateSuccessTest, "Schola.Spaces.DictSpace.Validate Success Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FDictSpaceValidateSuccessTest::RunTest(const FString& Parameters)
{
	FBoxSpace BoxSpace1;
	BoxSpace1.Add(-1.0, 1.0);
	FBoxSpace BoxSpace2;
	BoxSpace2.Add(-2.0, 2.0);

	FDictSpace DictSpace;
	DictSpace.Spaces.Add("box1", TInstancedStruct<FSpace>::Make<FBoxSpace>(BoxSpace1));
	DictSpace.Spaces.Add("box2", TInstancedStruct<FSpace>::Make<FBoxSpace>(BoxSpace2));

	FBoxPoint BoxPoint1;
	BoxPoint1.Add(0.0f);
	FBoxPoint BoxPoint2;
	BoxPoint2.Add(0.0f);

	FDictPoint DictPoint;
	DictPoint.Points.Add("box1", TInstancedStruct<FPoint>::Make<FBoxPoint>(BoxPoint1));
	DictPoint.Points.Add("box2", TInstancedStruct<FPoint>::Make<FBoxPoint>(BoxPoint2));
	TInstancedStruct<FPoint> Point = TInstancedStruct<FPoint>::Make<FDictPoint>(DictPoint);

	TestEqual(TEXT("DictSpace.Validate(Point) == ESpaceValidationResult::Success"), DictSpace.Validate(Point), ESpaceValidationResult::Success);

	return true;
}
// Check that we correctly identify when the point has the wrong data type
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDictSpaceValidateWrongDataTypeTest, "Schola.Spaces.DictSpace.Validate Wrong Data Type Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FDictSpaceValidateWrongDataTypeTest::RunTest(const FString& Parameters)
{
	FBoxSpace BoxSpace;
	BoxSpace.Add(-1.0, 1.0);
	FDictSpace DictSpace;
	DictSpace.Spaces.Add("box1", TInstancedStruct<FSpace>::Make<FBoxSpace>(BoxSpace));

	TInstancedStruct<FPoint> Point;
	Point.InitializeAs<FBoxPoint>();

	TestEqual(TEXT("DictSpace.Validate(Point) == ESpaceValidationResult::WrongDataType"), DictSpace.Validate(Point), ESpaceValidationResult::WrongDataType);

	return true;
}
// Check that we correctly identify when a sub-point has the wrong data type
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDictSpaceValidateWrongSubDataTypeTest, "Schola.Spaces.DictSpace.Validate Wrong Sub Data Type Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FDictSpaceValidateWrongSubDataTypeTest::RunTest(const FString& Parameters)
{
	FBoxSpace BoxSpace;
	BoxSpace.Add(-1.0, 1.0);

	FDictSpace DictSpace;
	DictSpace.Spaces.Add("box1", TInstancedStruct<FSpace>::Make<FBoxSpace>(BoxSpace));
	
	FDiscretePoint DiscretePoint = FDiscretePoint(0);

	FDictPoint DictPoint;
	DictPoint.Points.Add("box1", TInstancedStruct<FPoint>::Make<FDiscretePoint>(DiscretePoint));
	TInstancedStruct<FPoint> Point = TInstancedStruct<FPoint>::Make<FDictPoint>(DictPoint);

	TestTrue(TEXT("DictSpace.Validate(Point) == ESpaceValidationResult::WrongDataType"), DictSpace.Validate(Point) == ESpaceValidationResult::WrongDataType);

	return true;
}
// Check that we correctly identify when a sub-point has the wrong dimensions
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDictSpaceValidateWrongSubDimensionsTest, "Schola.Spaces.DictSpace.Validate Wrong Sub Dimensions Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FDictSpaceValidateWrongSubDimensionsTest::RunTest(const FString& Parameters)
{
	FBoxSpace BoxSpace;
	BoxSpace.Add(-1.0, 1.0);
	BoxSpace.Add(-1.0, 1.0);

	FDictSpace DictSpace;
	DictSpace.Spaces.Add("box1", TInstancedStruct<FSpace>::Make<FBoxSpace>(BoxSpace));

	FBoxPoint BoxPoint = FBoxPoint({2.0f});

	FDictPoint DictPoint;
	DictPoint.Points.Add("box1", TInstancedStruct<FPoint>::Make<FBoxPoint>(BoxPoint));
	TInstancedStruct<FPoint> Point = TInstancedStruct<FPoint>::Make<FDictPoint>(DictPoint);

	TestTrue(TEXT("DictSpace.Validate(Point) == ESpaceValidationResult::WrongDimensions"), DictSpace.Validate(Point) == ESpaceValidationResult::WrongDimensions);

	return true;
}

// Check if we correctly identify when the keys of the sub-point do not match the keys of the space
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDictSpaceValidateWrongDimensionsKeysMismatchTest, "Schola.Spaces.DictSpace.Validate Wrong Dimensions When Keys Do Not Match Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FDictSpaceValidateWrongDimensionsKeysMismatchTest::RunTest(const FString& Parameters)
{
	FBoxSpace BoxSpace1;
	BoxSpace1.Add(-1.0, 1.0);
	FBoxSpace BoxSpace2;
	BoxSpace2.Add(-2.0, 2.0);

	FDictSpace DictSpace;
	DictSpace.Spaces.Add("box1", TInstancedStruct<FSpace>::Make<FBoxSpace>(BoxSpace1));
	DictSpace.Spaces.Add("box2", TInstancedStruct<FSpace>::Make<FBoxSpace>(BoxSpace2));

	FBoxPoint BoxPoint1 = FBoxPoint({0.0f});
	FBoxPoint BoxPoint2 = FBoxPoint({0.0f});
	FDictPoint DictPoint;
	DictPoint.Points.Add("box1", TInstancedStruct<FPoint>::Make<FBoxPoint>(BoxPoint1));
	DictPoint.Points.Add("other_key", TInstancedStruct<FPoint>::Make<FBoxPoint>(BoxPoint2));
	TInstancedStruct<FPoint> Point = TInstancedStruct<FPoint>::Make<FDictPoint>(DictPoint);

	TestEqual(TEXT("DictSpace.Validate(Point) == ESpaceValidationResult::WrongDimensions when point keys do not match space keys"), DictSpace.Validate(Point), ESpaceValidationResult::WrongDimensions);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDictSpaceToStringEmptyTest, "Schola.Spaces.DictSpace.ToString.Empty", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FDictSpaceToStringEmptyTest::RunTest(const FString& Parameters)
{
	const FDictSpace DictSpace;
	TestEqual(TEXT("FDictSpace::ToString empty"), DictSpace.ToString(), FString(TEXT("DictSpace(\n{})")));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDictSpaceToStringSingleEntryTest, "Schola.Spaces.DictSpace.ToString.SingleEntry", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FDictSpaceToStringSingleEntryTest::RunTest(const FString& Parameters)
{
	FDictSpace DictSpace;
	FDiscreteSpace Discrete;
	Discrete.High = 5;
	DictSpace.Spaces.Add(TEXT("slot"), TInstancedStruct<FSpace>::Make<FDiscreteSpace>(Discrete));

	TestEqual(
		TEXT("FDictSpace::ToString single named subspace"),
		DictSpace.ToString(),
		FString(TEXT("DictSpace(\n{\tslot: DiscreteSpace(High=5),\n})")));
	return true;
}

#endif