// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Spaces/Blueprint/TextSpaceBlueprintLibrary.h"
#include "Spaces/TextSpace.h"

#if WITH_DEV_AUTOMATION_TESTS

// MakeTextSpace Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextSpaceBlueprintLibrary_MakeTextSpace_FullTest, "Schola.Spaces.Blueprint.TextSpaceBlueprintLibrary.MakeTextSpace.Full", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextSpaceBlueprintLibrary_MakeTextSpace_FullTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FTextSpace> Result = UTextSpaceBlueprintLibrary::MakeTextSpace(32, true, 4, TEXT("abc"));

    TestTrue(TEXT("Result is valid"), Result.IsValid());

    const FTextSpace& TextSpace = Result.Get<FTextSpace>();
    TestEqual(TEXT("TextSpace.MaxLength == 32"), TextSpace.MaxLength, 32);
    TestTrue(TEXT("TextSpace.bHasMinLength == true"), TextSpace.bHasMinLength);
    TestEqual(TEXT("TextSpace.MinLength == 4"), TextSpace.MinLength, 4);
    TestEqual(TEXT("TextSpace.Charset == abc"), TextSpace.Charset, FString(TEXT("abc")));

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextSpaceBlueprintLibrary_MakeTextSpace_MinimalTest, "Schola.Spaces.Blueprint.TextSpaceBlueprintLibrary.MakeTextSpace.Minimal", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextSpaceBlueprintLibrary_MakeTextSpace_MinimalTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FTextSpace> Result = UTextSpaceBlueprintLibrary::MakeTextSpace(10, false, 0, FString());

    TestTrue(TEXT("Result is valid"), Result.IsValid());

    const FTextSpace& TextSpace = Result.Get<FTextSpace>();
    TestEqual(TEXT("TextSpace.MaxLength == 10"), TextSpace.MaxLength, 10);
    TestFalse(TEXT("TextSpace.bHasMinLength == false"), TextSpace.bHasMinLength);
    TestTrue(TEXT("TextSpace.Charset is empty"), TextSpace.Charset.IsEmpty());

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
    TextSpace.bHasMinLength = true;
    TextSpace.MinLength = 2;
    TextSpace.Charset = TEXT("xyz");

    int32 MaxLength = 0;
    bool bHasMinLength = false;
    int32 MinLength = 0;
    FString Charset;
    UTextSpaceBlueprintLibrary::BreakTextSpace(Space, MaxLength, bHasMinLength, MinLength, Charset);

    TestEqual(TEXT("MaxLength == 16"), MaxLength, 16);
    TestTrue(TEXT("bHasMinLength == true"), bHasMinLength);
    TestEqual(TEXT("MinLength == 2"), MinLength, 2);
    TestEqual(TEXT("Charset == xyz"), Charset, FString(TEXT("xyz")));

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextSpaceBlueprintLibrary_BreakTextSpace_RoundTripTest, "Schola.Spaces.Blueprint.TextSpaceBlueprintLibrary.BreakTextSpace.RoundTrip", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextSpaceBlueprintLibrary_BreakTextSpace_RoundTripTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FTextSpace> Space = UTextSpaceBlueprintLibrary::MakeTextSpace(64, true, 8, TEXT("abcdef"));

    int32 MaxLength = 0;
    bool bHasMinLength = false;
    int32 MinLength = 0;
    FString Charset;
    UTextSpaceBlueprintLibrary::BreakTextSpace(Space, MaxLength, bHasMinLength, MinLength, Charset);

    TestEqual(TEXT("Round trip MaxLength"), MaxLength, 64);
    TestTrue(TEXT("Round trip bHasMinLength"), bHasMinLength);
    TestEqual(TEXT("Round trip MinLength"), MinLength, 8);
    TestEqual(TEXT("Round trip Charset"), Charset, FString(TEXT("abcdef")));

    return true;
}

#endif
