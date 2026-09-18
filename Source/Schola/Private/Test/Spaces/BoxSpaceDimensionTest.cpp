// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Spaces/BoxSpaceDimension.h"
#if WITH_AUTOMATION_TESTS
#define TestEqualExactFloat(TestMessage, Actual, Expected) TestEqual(TestMessage, (float)Actual, (float)Expected, 0.0001f)

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceDimensionDefaultTest, "Schola.Spaces.BoxSpace.BoxSpaceDimension.Default Creation Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceDimensionDefaultTest::RunTest(const FString& Parameters)
{
    FBoxSpaceDimension BoxSpaceDimension = FBoxSpaceDimension();

    TestEqualExactFloat(TEXT("BoxSpaceDimension.Low == -1.0"), BoxSpaceDimension.Low, -1.0f);
	TestEqualExactFloat(TEXT("BoxSpaceDimension.High == 1.0"), BoxSpaceDimension.High, 1.0f);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceDimensionBoundsTest, "Schola.Spaces.BoxSpace.BoxSpaceDimension.Bounds Creation Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceDimensionBoundsTest::RunTest(const FString& Parameters)
{
    FBoxSpaceDimension BoxSpaceDimension = FBoxSpaceDimension(-2.0, 3.0);

    TestEqualExactFloat(TEXT("BoxSpaceDimension.Low == -2.0"), BoxSpaceDimension.Low, -2.0f);
	TestEqualExactFloat(TEXT("BoxSpaceDimension.High == 3.0"), BoxSpaceDimension.High, 3.0f);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceDimensionZeroOneUnitTest, "Schola.Spaces.BoxSpace.BoxSpaceDimension.Zero One Unit Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceDimensionZeroOneUnitTest::RunTest(const FString& Parameters)
{
    FBoxSpaceDimension BoxSpaceDimension = FBoxSpaceDimension::ZeroOneUnitDimension();

    TestEqualExactFloat(TEXT("BoxSpaceDimension.Low == 0.0"), BoxSpaceDimension.Low, 0.0);
    TestEqualExactFloat(TEXT("BoxSpaceDimension.High == 1.0"), BoxSpaceDimension.High, 1.0);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceDimensionCenteredUnitTest, "Schola.Spaces.BoxSpace.BoxSpaceDimension.Centered Unit Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceDimensionCenteredUnitTest::RunTest(const FString& Parameters)
{
    FBoxSpaceDimension BoxSpaceDimension = FBoxSpaceDimension::CenteredUnitDimension();

    TestEqualExactFloat(TEXT("BoxSpaceDimension.Low == -0.5"), BoxSpaceDimension.Low, -0.5);
    TestEqualExactFloat(TEXT("BoxSpaceDimension.High == 0.5"), BoxSpaceDimension.High, 0.5);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceDimensionDenormalizeTest, "Schola.Spaces.BoxSpace.BoxSpaceDimension.Denormalize Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceDimensionDenormalizeTest::RunTest(const FString& Parameters)
{
    FBoxSpaceDimension BoxSpaceDimension = FBoxSpaceDimension(-3.0, 3.0);

    TestEqualExactFloat(TEXT("BoxSpaceDimension.RescaleValue(0.0) == -3.0"), BoxSpaceDimension.RescaleValue(0.0), -3.0);
    TestEqualExactFloat(TEXT("BoxSpaceDimension.RescaleValue(0.5) == 0.0"), BoxSpaceDimension.RescaleValue(0.5), 0.0);
    TestEqualExactFloat(TEXT("BoxSpaceDimension.RescaleValue(1) == 3.0"), BoxSpaceDimension.RescaleValue(1), 3.0);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceDimensionRescaleTest, "Schola.Spaces.BoxSpace.BoxSpaceDimension.Rescale Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceDimensionRescaleTest::RunTest(const FString& Parameters)
{
    FBoxSpaceDimension BoxSpaceDimension = FBoxSpaceDimension(0, 10.0);

    TestEqualExactFloat(TEXT("BoxSpaceDimension.RescaleValue(0.0, 10.0, 0.0) == 0.0"), BoxSpaceDimension.RescaleValue(0.0, 10.0, 0.0), 0.0);
    TestEqualExactFloat(TEXT("BoxSpaceDimension.RescaleValue(5.0, 10.0, 0.0) == 5.0"), BoxSpaceDimension.RescaleValue(5.0, 10.0, 0.0), 5.0);

    TestEqualExactFloat(TEXT("BoxSpaceDimension.RescaleValue(7.5, 10.0, 5.0) == 5.0"), BoxSpaceDimension.RescaleValue(7.5, 10.0, 5.0), 5.0);
    TestEqualExactFloat(TEXT("BoxSpaceDimension.RescaleValue(3.0, 5.0, 0.0) == 6.0"), BoxSpaceDimension.RescaleValue(3.0, 5.0, 0.0), 6.0);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceDimensionNormalizeTest, "Schola.Spaces.BoxSpace.BoxSpaceDimension.Normalize Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceDimensionNormalizeTest::RunTest(const FString& Parameters)
{
    FBoxSpaceDimension BoxSpaceDimension = FBoxSpaceDimension(-3.0, 3.0);

    TestEqualExactFloat(TEXT("BoxSpaceDimension.NormalizeValue(-3.0) == 0.0"), BoxSpaceDimension.NormalizeValue(-3.0), 0.0);
    TestEqualExactFloat(TEXT("BoxSpaceDimension.NormalizeValue(0.0) == 0.5"), BoxSpaceDimension.NormalizeValue(0.0), 0.5);
    TestEqualExactFloat(TEXT("BoxSpaceDimension.NormalizeValue(3.0) == 1.0"), BoxSpaceDimension.NormalizeValue(3.0), 1.0);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceDimensionDefaultBoundedFlagsTest, "Schola.Spaces.BoxSpace.BoxSpaceDimension.Default Bounded Flags Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceDimensionDefaultBoundedFlagsTest::RunTest(const FString& Parameters)
{
    FBoxSpaceDimension BoxSpaceDimension = FBoxSpaceDimension();
    TestTrue(TEXT("Default dimension is low bounded"), BoxSpaceDimension.bHasLow);
    TestTrue(TEXT("Default dimension is high bounded"), BoxSpaceDimension.bHasHigh);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceDimensionValidateDimensionTest, "Schola.Spaces.BoxSpace.BoxSpaceDimension.Validate Dimension Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceDimensionValidateDimensionTest::RunTest(const FString& Parameters)
{
    FBoxSpaceDimension Bounded = FBoxSpaceDimension(-1.0f, 1.0f);
    TestTrue(TEXT("Bounded contains 0"), Bounded.ValidateDimension(0.0f) == ESpaceValidationResult::Success);
    TestTrue(TEXT("Bounded rejects below low"), Bounded.ValidateDimension(-2.0f) == ESpaceValidationResult::OutOfBounds);
    TestTrue(TEXT("Bounded rejects above high"), Bounded.ValidateDimension(2.0f) == ESpaceValidationResult::OutOfBounds);

    FBoxSpaceDimension LowerOnly = FBoxSpaceDimension::LowerBounded(0.0f);
    TestTrue(TEXT("LowerBounded accepts large positive"), LowerOnly.ValidateDimension(1000.0f) == ESpaceValidationResult::Success);
    TestTrue(TEXT("LowerBounded rejects below low"), LowerOnly.ValidateDimension(-1.0f) == ESpaceValidationResult::OutOfBounds);

    FBoxSpaceDimension UpperOnly = FBoxSpaceDimension::UpperBounded(0.0f);
    TestTrue(TEXT("UpperBounded accepts large negative"), UpperOnly.ValidateDimension(-1000.0f) == ESpaceValidationResult::Success);
    TestTrue(TEXT("UpperBounded rejects above high"), UpperOnly.ValidateDimension(1.0f) == ESpaceValidationResult::OutOfBounds);

    FBoxSpaceDimension Unbounded = FBoxSpaceDimension::Unbounded();
    TestTrue(TEXT("Unbounded accepts large magnitude"), Unbounded.ValidateDimension(-1.0e20f) == ESpaceValidationResult::Success);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceDimensionUnboundedNormalizeTest, "Schola.Spaces.BoxSpace.BoxSpaceDimension.Unbounded Normalize Identity Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceDimensionUnboundedNormalizeTest::RunTest(const FString& Parameters)
{
    FBoxSpaceDimension Unbounded = FBoxSpaceDimension::Unbounded();
    TestEqualExactFloat(TEXT("Unbounded NormalizeValue is identity"), Unbounded.NormalizeValue(7.5f), 7.5f);
    TestEqualExactFloat(TEXT("Unbounded RescaleValue is identity"), Unbounded.RescaleValue(0.25f), 0.25f);
    TestEqualExactFloat(TEXT("Unbounded two-arg RescaleValue is identity"), Unbounded.RescaleValue(3.0f, 10.0f, 0.0f), 3.0f);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceDimensionToStringTest, "Schola.Spaces.BoxSpace.BoxSpaceDimension.ToString Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceDimensionToStringTest::RunTest(const FString& Parameters)
{
    TestEqual(TEXT("Bounded ToString"), FBoxSpaceDimension(-1.0f, 1.0f).ToString(), FString(TEXT("[-1, 1]")));
    TestEqual(TEXT("Unbounded ToString"), FBoxSpaceDimension::Unbounded().ToString(), FString(TEXT("[-inf, inf]")));
    TestEqual(TEXT("LowerBounded ToString"), FBoxSpaceDimension::LowerBounded(0.0f).ToString(), FString(TEXT("[0, inf]")));
    TestEqual(TEXT("UpperBounded ToString"), FBoxSpaceDimension::UpperBounded(0.0f).ToString(), FString(TEXT("[-inf, 0]")));
    return true;
}
#endif