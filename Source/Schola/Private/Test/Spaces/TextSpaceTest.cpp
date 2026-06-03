// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Spaces/TextSpace.h"
#include "Points/TextPoint.h"
#include "Points/DiscretePoint.h"

#if WITH_DEV_AUTOMATION_TESTS

// Constructor Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextSpaceDefaultTest, "Schola.Spaces.TextSpace.Constructor.Default", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextSpaceDefaultTest::RunTest(const FString& Parameters)
{
	FTextSpace TextSpace = FTextSpace();

	TestEqual(TEXT("TextSpace.MaxLength == 0"), TextSpace.MaxLength, 0);
	TestFalse(TEXT("TextSpace.bHasMinLength == false"), TextSpace.bHasMinLength);
	TestEqual(TEXT("TextSpace.MinLength == 0"), TextSpace.MinLength, 0);
	TestTrue(TEXT("TextSpace.Charset is empty"), TextSpace.Charset.IsEmpty());

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextSpaceMaxLengthConstructorTest, "Schola.Spaces.TextSpace.Constructor.MaxLength", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextSpaceMaxLengthConstructorTest::RunTest(const FString& Parameters)
{
	FTextSpace TextSpace = FTextSpace(16);

	TestEqual(TEXT("TextSpace.MaxLength == 16"), TextSpace.MaxLength, 16);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextSpaceCopyTest, "Schola.Spaces.TextSpace.Copy", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextSpaceCopyTest::RunTest(const FString& Parameters)
{
	FTextSpace OriginalSpace = FTextSpace();
	OriginalSpace.MaxLength = 32;
	OriginalSpace.bHasMinLength = true;
	OriginalSpace.MinLength = 4;
	OriginalSpace.Charset = TEXT("abc");

	FTextSpace CopiedSpace = FTextSpace();
	CopiedSpace.Copy(OriginalSpace);

	TestEqual(TEXT("CopiedSpace.MaxLength == 32"), CopiedSpace.MaxLength, 32);
	TestTrue(TEXT("CopiedSpace.bHasMinLength == true"), CopiedSpace.bHasMinLength);
	TestEqual(TEXT("CopiedSpace.MinLength == 4"), CopiedSpace.MinLength, 4);
	TestEqual(TEXT("CopiedSpace.Charset == abc"), CopiedSpace.Charset, FString(TEXT("abc")));

	return true;
}

// Method Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextSpaceGetNumDimensionsTest, "Schola.Spaces.TextSpace.Get Num Dimensions Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextSpaceGetNumDimensionsTest::RunTest(const FString& Parameters)
{
	FTextSpace TextSpace = FTextSpace(16);

	TestEqual(TEXT("TextSpace.GetNumDimensions() == 1"), TextSpace.GetNumDimensions(), 1);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextSpaceFlattenedSizeTest, "Schola.Spaces.TextSpace.FlattenedSize Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextSpaceFlattenedSizeTest::RunTest(const FString& Parameters)
{
	FTextSpace TextSpace = FTextSpace(16);

	TestEqual(TEXT("TextSpace.GetFlattenedSize() == 16"), TextSpace.GetFlattenedSize(), 16);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextSpaceIsEmptyTrueTest, "Schola.Spaces.TextSpace.Is Empty True Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextSpaceIsEmptyTrueTest::RunTest(const FString& Parameters)
{
	FTextSpace TextSpace = FTextSpace();

	TestTrue(TEXT("TextSpace.IsEmpty() == true"), TextSpace.IsEmpty());

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextSpaceIsEmptyFalseTest, "Schola.Spaces.TextSpace.Is Empty False Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextSpaceIsEmptyFalseTest::RunTest(const FString& Parameters)
{
	FTextSpace TextSpace = FTextSpace(8);

	TestFalse(TEXT("TextSpace.IsEmpty() == false"), TextSpace.IsEmpty());

	return true;
}

// Validation Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextSpaceValidateSuccessTest, "Schola.Spaces.TextSpace.Validate Success Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextSpaceValidateSuccessTest::RunTest(const FString& Parameters)
{
	FTextSpace TextSpace = FTextSpace(16);

	TInstancedStruct<FPoint> Point = TInstancedStruct<FPoint>::Make<FTextPoint>(FTextPoint(TEXT("hello")));

	TestEqual(TEXT("TextSpace.Validate(Point) == Success"), TextSpace.Validate(Point), ESpaceValidationResult::Success);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextSpaceValidateTooLongTest, "Schola.Spaces.TextSpace.Validate Too Long Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextSpaceValidateTooLongTest::RunTest(const FString& Parameters)
{
	FTextSpace TextSpace = FTextSpace(3);

	TInstancedStruct<FPoint> Point = TInstancedStruct<FPoint>::Make<FTextPoint>(FTextPoint(TEXT("hello")));

	TestEqual(TEXT("TextSpace.Validate(Point) == OutOfBounds"), TextSpace.Validate(Point), ESpaceValidationResult::OutOfBounds);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextSpaceValidateMinLengthSuccessTest, "Schola.Spaces.TextSpace.Validate Min Length Success Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextSpaceValidateMinLengthSuccessTest::RunTest(const FString& Parameters)
{
	FTextSpace TextSpace = FTextSpace(16);
	TextSpace.bHasMinLength = true;
	TextSpace.MinLength = 3;

	TInstancedStruct<FPoint> Point = TInstancedStruct<FPoint>::Make<FTextPoint>(FTextPoint(TEXT("hello")));

	TestEqual(TEXT("TextSpace.Validate(Point) == Success"), TextSpace.Validate(Point), ESpaceValidationResult::Success);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextSpaceValidateTooShortTest, "Schola.Spaces.TextSpace.Validate Too Short Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextSpaceValidateTooShortTest::RunTest(const FString& Parameters)
{
	FTextSpace TextSpace = FTextSpace(16);
	TextSpace.bHasMinLength = true;
	TextSpace.MinLength = 10;

	TInstancedStruct<FPoint> Point = TInstancedStruct<FPoint>::Make<FTextPoint>(FTextPoint(TEXT("hello")));

	TestEqual(TEXT("TextSpace.Validate(Point) == OutOfBounds"), TextSpace.Validate(Point), ESpaceValidationResult::OutOfBounds);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextSpaceValidateCharsetSuccessTest, "Schola.Spaces.TextSpace.Validate Charset Success Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextSpaceValidateCharsetSuccessTest::RunTest(const FString& Parameters)
{
	FTextSpace TextSpace = FTextSpace(16);
	TextSpace.Charset = TEXT("abc");

	TInstancedStruct<FPoint> Point = TInstancedStruct<FPoint>::Make<FTextPoint>(FTextPoint(TEXT("cab")));

	TestEqual(TEXT("TextSpace.Validate(Point) == Success"), TextSpace.Validate(Point), ESpaceValidationResult::Success);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextSpaceValidateCharsetViolationTest, "Schola.Spaces.TextSpace.Validate Charset Violation Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextSpaceValidateCharsetViolationTest::RunTest(const FString& Parameters)
{
	FTextSpace TextSpace = FTextSpace(16);
	TextSpace.Charset = TEXT("abc");

	TInstancedStruct<FPoint> Point = TInstancedStruct<FPoint>::Make<FTextPoint>(FTextPoint(TEXT("abz")));

	TestEqual(TEXT("TextSpace.Validate(Point) == OutOfBounds"), TextSpace.Validate(Point), ESpaceValidationResult::OutOfBounds);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextSpaceValidateWrongDataTypeTest, "Schola.Spaces.TextSpace.Validate Wrong Data Type Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextSpaceValidateWrongDataTypeTest::RunTest(const FString& Parameters)
{
	FTextSpace TextSpace = FTextSpace(16);

	TInstancedStruct<FPoint> Point;
	Point.InitializeAs<FDiscretePoint>();

	TestEqual(TEXT("TextSpace.Validate(Point) == WrongDataType"), TextSpace.Validate(Point), ESpaceValidationResult::WrongDataType);

	return true;
}

#endif
