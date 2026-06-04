// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Spaces/Blueprint/TextSpaceBlueprintLibrary.h"
#include "Spaces/TextSpace.h"

#if WITH_DEV_AUTOMATION_TESTS

// MakeTextSpace Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextSpaceBlueprintLibrary_MakeTextSpace_FullTest, "Schola.Spaces.Blueprint.TextSpaceBlueprintLibrary.MakeTextSpace.Full", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextSpaceBlueprintLibrary_MakeTextSpace_FullTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FTextSpace> Result = UTextSpaceBlueprintLibrary::MakeTextSpace(32, 4, TEXT("abc"));

    TestTrue(TEXT("Result is valid"), Result.IsValid());

    const FTextSpace& TextSpace = Result.Get<FTextSpace>();
    TestEqual(TEXT("TextSpace.MaxLength == 32"), TextSpace.MaxLength, 32);
    TestEqual(TEXT("TextSpace.MinLength == 4"), TextSpace.MinLength, 4);
    TestEqual(TEXT("TextSpace.Charset == abc"), TextSpace.Charset, FString(TEXT("abc")));

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextSpaceBlueprintLibrary_MakeTextSpace_MinimalTest, "Schola.Spaces.Blueprint.TextSpaceBlueprintLibrary.MakeTextSpace.Minimal", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextSpaceBlueprintLibrary_MakeTextSpace_MinimalTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FTextSpace> Result = UTextSpaceBlueprintLibrary::MakeTextSpace(10, 1, FString());

    TestTrue(TEXT("Result is valid"), Result.IsValid());

    const FTextSpace& TextSpace = Result.Get<FTextSpace>();
    TestEqual(TEXT("TextSpace.MaxLength == 10"), TextSpace.MaxLength, 10);
    TestEqual(TEXT("TextSpace.MinLength == 1"), TextSpace.MinLength, 1);
    TestTrue(TEXT("TextSpace.Charset is empty"), TextSpace.Charset.IsEmpty());

    return true;
}

// MakeTextSpaceFromPreset Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextSpaceBlueprintLibrary_MakeTextSpaceFromPreset_AlphanumericTest, "Schola.Spaces.Blueprint.TextSpaceBlueprintLibrary.MakeTextSpaceFromPreset.Alphanumeric", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextSpaceBlueprintLibrary_MakeTextSpaceFromPreset_AlphanumericTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FTextSpace> Result = UTextSpaceBlueprintLibrary::MakeTextSpaceFromPreset(20, 2, ETextCharsetPreset::Alphanumeric);

    TestTrue(TEXT("Result is valid"), Result.IsValid());

    const FTextSpace& TextSpace = Result.Get<FTextSpace>();
    TestEqual(TEXT("TextSpace.MaxLength == 20"), TextSpace.MaxLength, 20);
    TestEqual(TEXT("TextSpace.MinLength == 2"), TextSpace.MinLength, 2);
    TestEqual(TEXT("TextSpace.Charset == Alphanumeric preset"), TextSpace.Charset, ScholaTextCharsets::Alphanumeric);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextSpaceBlueprintLibrary_MakeTextSpaceFromPreset_AnyTest, "Schola.Spaces.Blueprint.TextSpaceBlueprintLibrary.MakeTextSpaceFromPreset.Any", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextSpaceBlueprintLibrary_MakeTextSpaceFromPreset_AnyTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FTextSpace> Result = UTextSpaceBlueprintLibrary::MakeTextSpaceFromPreset(8, 1, ETextCharsetPreset::Any);

    TestTrue(TEXT("Result is valid"), Result.IsValid());

    const FTextSpace& TextSpace = Result.Get<FTextSpace>();
    TestEqual(TEXT("TextSpace.MaxLength == 8"), TextSpace.MaxLength, 8);
    TestTrue(TEXT("TextSpace.Charset is empty (no restriction)"), TextSpace.Charset.IsEmpty());

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextSpaceBlueprintLibrary_MakeTextSpace_MinGreaterThanMaxIsClampedTest, "Schola.Spaces.Blueprint.TextSpaceBlueprintLibrary.MakeTextSpace.MinGreaterThanMaxClamped", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextSpaceBlueprintLibrary_MakeTextSpace_MinGreaterThanMaxIsClampedTest::RunTest(const FString& Parameters)
{
    // MinLength (10) exceeds MaxLength (5); it should be clamped down to MaxLength so the space stays coherent.
    TInstancedStruct<FTextSpace> Result = UTextSpaceBlueprintLibrary::MakeTextSpace(5, 10, FString());

    TestTrue(TEXT("Result is valid"), Result.IsValid());

    const FTextSpace& TextSpace = Result.Get<FTextSpace>();
    TestEqual(TEXT("TextSpace.MaxLength == 5"), TextSpace.MaxLength, 5);
    TestEqual(TEXT("TextSpace.MinLength clamped to 5"), TextSpace.MinLength, 5);

    return true;
}

// BreakTextSpace Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextSpaceBlueprintLibrary_BreakTextSpace_BasicTest, "Schola.Spaces.Blueprint.TextSpaceBlueprintLibrary.BreakTextSpace.Basic", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextSpaceBlueprintLibrary_BreakTextSpace_BasicTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FTextSpace> Space;
    Space.InitializeAs<FTextSpace>();
    FTextSpace& TextSpace = Space.GetMutable<FTextSpace>();
    TextSpace.MaxLength = 16;
    TextSpace.MinLength = 2;
    TextSpace.Charset = TEXT("xyz");

    int32 MaxLength = 0;
    int32 MinLength = 0;
    FString Charset;
    UTextSpaceBlueprintLibrary::BreakTextSpace(Space, MaxLength, MinLength, Charset);

    TestEqual(TEXT("MaxLength == 16"), MaxLength, 16);
    TestEqual(TEXT("MinLength == 2"), MinLength, 2);
    TestEqual(TEXT("Charset == xyz"), Charset, FString(TEXT("xyz")));

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextSpaceBlueprintLibrary_BreakTextSpace_RoundTripTest, "Schola.Spaces.Blueprint.TextSpaceBlueprintLibrary.BreakTextSpace.RoundTrip", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextSpaceBlueprintLibrary_BreakTextSpace_RoundTripTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FTextSpace> Space = UTextSpaceBlueprintLibrary::MakeTextSpace(64, 8, TEXT("abcdef"));

    int32 MaxLength = 0;
    int32 MinLength = 0;
    FString Charset;
    UTextSpaceBlueprintLibrary::BreakTextSpace(Space, MaxLength, MinLength, Charset);

    TestEqual(TEXT("Round trip MaxLength"), MaxLength, 64);
    TestEqual(TEXT("Round trip MinLength"), MinLength, 8);
    TestEqual(TEXT("Round trip Charset"), Charset, FString(TEXT("abcdef")));

    return true;
}

#endif
