// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Points/Blueprint/TextPointBlueprintLibrary.h"
#include "Points/TextPoint.h"

#if WITH_DEV_AUTOMATION_TESTS

// StringToTextPoint Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextPointBlueprintLibrary_StringToTextPoint_BasicTest, "Schola.Points.Blueprint.TextPointBlueprintLibrary.StringToTextPoint.Basic", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextPointBlueprintLibrary_StringToTextPoint_BasicTest::RunTest(const FString& Parameters)
{
    FString Value = TEXT("hello world");

    TInstancedStruct<FTextPoint> Result = UTextPointBlueprintLibrary::StringToTextPoint(Value);

    TestTrue(TEXT("Result is valid"), Result.IsValid());

    const FTextPoint& TextPoint = Result.Get<FTextPoint>();
    TestEqual(TEXT("TextPoint.Value"), TextPoint.Value, Value);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextPointBlueprintLibrary_StringToTextPoint_EmptyTest, "Schola.Points.Blueprint.TextPointBlueprintLibrary.StringToTextPoint.Empty", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextPointBlueprintLibrary_StringToTextPoint_EmptyTest::RunTest(const FString& Parameters)
{
    FString Value;

    TInstancedStruct<FTextPoint> Result = UTextPointBlueprintLibrary::StringToTextPoint(Value);

    TestTrue(TEXT("Result is valid"), Result.IsValid());

    const FTextPoint& TextPoint = Result.Get<FTextPoint>();
    TestTrue(TEXT("TextPoint.Value is empty"), TextPoint.Value.IsEmpty());

    return true;
}

// TextPointToString Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextPointBlueprintLibrary_TextPointToString_BasicTest, "Schola.Points.Blueprint.TextPointBlueprintLibrary.TextPointToString.Basic", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextPointBlueprintLibrary_TextPointToString_BasicTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FTextPoint> Point;
    Point.InitializeAs<FTextPoint>();
    Point.GetMutable<FTextPoint>().Value = TEXT("sample text");

    FString Result = UTextPointBlueprintLibrary::TextPointToString(Point);

    TestEqual(TEXT("Result"), Result, FString(TEXT("sample text")));

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTextPointBlueprintLibrary_TextPointToString_RoundTripTest, "Schola.Points.Blueprint.TextPointBlueprintLibrary.TextPointToString.RoundTrip", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FTextPointBlueprintLibrary_TextPointToString_RoundTripTest::RunTest(const FString& Parameters)
{
    FString OriginalValue = TEXT("round trip value");

    TInstancedStruct<FTextPoint> Point = UTextPointBlueprintLibrary::StringToTextPoint(OriginalValue);
    FString Result = UTextPointBlueprintLibrary::TextPointToString(Point);

    TestEqual(TEXT("Round trip string"), Result, OriginalValue);

    return true;
}

#endif
