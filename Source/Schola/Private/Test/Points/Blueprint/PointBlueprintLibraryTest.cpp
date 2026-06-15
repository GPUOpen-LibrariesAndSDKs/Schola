// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Points/Blueprint/PointBlueprintLibrary.h"
#include "Points/BoxPoint.h"
#include "Points/DiscretePoint.h"
#include "Points/MultiBinaryPoint.h"
#include "Points/MultiDiscretePoint.h"
#include "Points/DictPoint.h"
#include "Points/TextPoint.h"
#include "Common/InstancedStructUtils.h"
#include "Math/Vector.h"

#if WITH_DEV_AUTOMATION_TESTS

// Point_Type Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FPointBlueprintLibrary_Type_BoxTest, "Schola.Points.Blueprint.PointBlueprintLibrary.Type.Box", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPointBlueprintLibrary_Type_BoxTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FPoint> Point;
    Point.InitializeAs<FBoxPoint>();
    FBoxPoint& BoxPoint = Point.GetMutable<FBoxPoint>();
    BoxPoint.Add(1.0f);

    EPointType Result = UPointBlueprintLibrary::Point_Type(ToUntypedInstancedStruct(Point));

    TestEqual(TEXT("Point_Type(BoxPoint) == EPointType::Box"), Result, EPointType::Box);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FPointBlueprintLibrary_Type_DiscreteTest, "Schola.Points.Blueprint.PointBlueprintLibrary.Type.Discrete", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPointBlueprintLibrary_Type_DiscreteTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FPoint> Point;
    Point.InitializeAs<FDiscretePoint>(5);


    EPointType Result = UPointBlueprintLibrary::Point_Type(ToUntypedInstancedStruct(Point));

    TestEqual(TEXT("Point_Type(DiscretePoint) == EPointType::Discrete"), Result, EPointType::Discrete);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FPointBlueprintLibrary_Type_MultiBinaryTest, "Schola.Points.Blueprint.PointBlueprintLibrary.Type.MultiBinary", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPointBlueprintLibrary_Type_MultiBinaryTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FPoint> Point;
    Point.InitializeAs<FMultiBinaryPoint>();
    FMultiBinaryPoint& MultiBinaryPoint = Point.GetMutable<FMultiBinaryPoint>();
    MultiBinaryPoint.Values = {true, false, true};

    EPointType Result = UPointBlueprintLibrary::Point_Type(ToUntypedInstancedStruct(Point));

    TestEqual(TEXT("Point_Type(MultiBinaryPoint) == EPointType::MultiBinary"), Result, EPointType::MultiBinary);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FPointBlueprintLibrary_Type_MultiDiscreteTest, "Schola.Points.Blueprint.PointBlueprintLibrary.Type.MultiDiscrete", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPointBlueprintLibrary_Type_MultiDiscreteTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FPoint> Point;
    Point.InitializeAs<FMultiDiscretePoint>();
    FMultiDiscretePoint& MultiDiscretePoint = Point.GetMutable<FMultiDiscretePoint>();
    MultiDiscretePoint.Values = {1, 2, 3};

    EPointType Result = UPointBlueprintLibrary::Point_Type(ToUntypedInstancedStruct(Point));

    TestEqual(TEXT("Point_Type(MultiDiscretePoint) == EPointType::MultiDiscrete"), Result, EPointType::MultiDiscrete);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FPointBlueprintLibrary_Type_DictTest, "Schola.Points.Blueprint.PointBlueprintLibrary.Type.Dict", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPointBlueprintLibrary_Type_DictTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FPoint> Point;
    Point.InitializeAs<FDictPoint>();

    EPointType Result = UPointBlueprintLibrary::Point_Type(ToUntypedInstancedStruct(Point));

    TestEqual(TEXT("Point_Type(DictPoint) == EPointType::Dict"), Result, EPointType::Dict);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FPointBlueprintLibrary_Type_TextTest, "Schola.Points.Blueprint.PointBlueprintLibrary.Type.Text", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPointBlueprintLibrary_Type_TextTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FPoint> Point;
    Point.InitializeAs<FTextPoint>(FString(TEXT("hello")));

    EPointType Result = UPointBlueprintLibrary::Point_Type(ToUntypedInstancedStruct(Point));

    TestEqual(TEXT("Point_Type(TextPoint) == EPointType::Text"), Result, EPointType::Text);

    return true;
}

// Point_IsOfType Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FPointBlueprintLibrary_IsOfType_BoxTrueTest, "Schola.Points.Blueprint.PointBlueprintLibrary.IsOfType.BoxTrue", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPointBlueprintLibrary_IsOfType_BoxTrueTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FPoint> Point;
    Point.InitializeAs<FBoxPoint>();

    bool Result = UPointBlueprintLibrary::Point_IsOfType(ToUntypedInstancedStruct(Point), EPointType::Box);

    TestTrue(TEXT("Point_IsOfType(BoxPoint, Box) == true"), Result);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FPointBlueprintLibrary_IsOfType_BoxFalseTest, "Schola.Points.Blueprint.PointBlueprintLibrary.IsOfType.BoxFalse", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPointBlueprintLibrary_IsOfType_BoxFalseTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FPoint> Point;
    Point.InitializeAs<FBoxPoint>();

    bool Result = UPointBlueprintLibrary::Point_IsOfType(ToUntypedInstancedStruct(Point), EPointType::Discrete);

    TestFalse(TEXT("Point_IsOfType(BoxPoint, Discrete) == false"), Result);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FPointBlueprintLibrary_IsOfType_DiscreteTrueTest, "Schola.Points.Blueprint.PointBlueprintLibrary.IsOfType.DiscreteTrue", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPointBlueprintLibrary_IsOfType_DiscreteTrueTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FPoint> Point;
    Point.InitializeAs<FDiscretePoint>();

    bool Result = UPointBlueprintLibrary::Point_IsOfType(ToUntypedInstancedStruct(Point), EPointType::Discrete);

    TestTrue(TEXT("Point_IsOfType(DiscretePoint, Discrete) == true"), Result);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FPointBlueprintLibrary_IsOfType_MultiBinaryTrueTest, "Schola.Points.Blueprint.PointBlueprintLibrary.IsOfType.MultiBinaryTrue", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPointBlueprintLibrary_IsOfType_MultiBinaryTrueTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FPoint> Point;
    Point.InitializeAs<FMultiBinaryPoint>();

    bool Result = UPointBlueprintLibrary::Point_IsOfType(ToUntypedInstancedStruct(Point), EPointType::MultiBinary);

    TestTrue(TEXT("Point_IsOfType(MultiBinaryPoint, Binary) == true"), Result);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FPointBlueprintLibrary_IsOfType_MultiDiscreteTrueTest, "Schola.Points.Blueprint.PointBlueprintLibrary.IsOfType.MultiDiscreteTrue", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPointBlueprintLibrary_IsOfType_MultiDiscreteTrueTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FPoint> Point;
    Point.InitializeAs<FMultiDiscretePoint>();

    bool Result = UPointBlueprintLibrary::Point_IsOfType(ToUntypedInstancedStruct(Point), EPointType::MultiDiscrete);

    TestTrue(TEXT("Point_IsOfType(MultiDiscretePoint, MultiDiscrete) == true"), Result);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FPointBlueprintLibrary_IsOfType_DictTrueTest, "Schola.Points.Blueprint.PointBlueprintLibrary.IsOfType.DictTrue", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPointBlueprintLibrary_IsOfType_DictTrueTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FPoint> Point;
    Point.InitializeAs<FDictPoint>();

    bool Result = UPointBlueprintLibrary::Point_IsOfType(ToUntypedInstancedStruct(Point), EPointType::Dict);

    TestTrue(TEXT("Point_IsOfType(DictPoint, Dict) == true"), Result);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FPointBlueprintLibrary_IsOfType_TextTrueTest, "Schola.Points.Blueprint.PointBlueprintLibrary.IsOfType.TextTrue", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPointBlueprintLibrary_IsOfType_TextTrueTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FPoint> Point;
    Point.InitializeAs<FTextPoint>(FString(TEXT("hello")));

    bool Result = UPointBlueprintLibrary::Point_IsOfType(ToUntypedInstancedStruct(Point), EPointType::Text);

    TestTrue(TEXT("Point_IsOfType(TextPoint, Text) == true"), Result);

    return true;
}

// PointToString Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FPointBlueprintLibrary_PointToString_InvalidTest, "Schola.Points.Blueprint.PointBlueprintLibrary.PointToString.Invalid", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPointBlueprintLibrary_PointToString_InvalidTest::RunTest(const FString& Parameters)
{
    FInstancedStruct Invalid;
    TestFalse(TEXT("Uninitialized instanced struct is invalid"), Invalid.IsValid());

    FString Result = UPointBlueprintLibrary::PointToString(Invalid);
    TestTrue(TEXT("PointToString(invalid) returns empty"), Result.IsEmpty());

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FPointBlueprintLibrary_PointToString_NonPointStructTest, "Schola.Points.Blueprint.PointBlueprintLibrary.PointToString.NonPointStruct", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPointBlueprintLibrary_PointToString_NonPointStructTest::RunTest(const FString& Parameters)
{
    FInstancedStruct NotAPoint;
    NotAPoint.InitializeAs<FVector>(FVector(1.0f, 2.0f, 3.0f));

    FString Result = UPointBlueprintLibrary::PointToString(NotAPoint);
    TestTrue(TEXT("PointToString(non-FPoint struct) returns empty"), Result.IsEmpty());

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FPointBlueprintLibrary_PointToString_DiscreteMatchesNativeTest, "Schola.Points.Blueprint.PointBlueprintLibrary.PointToString.DiscreteMatchesNative", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPointBlueprintLibrary_PointToString_DiscreteMatchesNativeTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FPoint> Point;
    Point.InitializeAs<FDiscretePoint>(42);

    const FString Expected = Point.Get<FDiscretePoint>().ToString();
    const FString Result = UPointBlueprintLibrary::PointToString(ToUntypedInstancedStruct(Point));

    TestEqual(TEXT("PointToString(Discrete) matches native ToString"), Result, Expected);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FPointBlueprintLibrary_PointToString_TextMatchesNativeTest, "Schola.Points.Blueprint.PointBlueprintLibrary.PointToString.TextMatchesNative", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPointBlueprintLibrary_PointToString_TextMatchesNativeTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FPoint> Point;
    Point.InitializeAs<FTextPoint>(FString(TEXT("hello")));

    const FString Expected = Point.Get<FTextPoint>().ToString();
    const FString Result = UPointBlueprintLibrary::PointToString(ToUntypedInstancedStruct(Point));

    TestEqual(TEXT("PointToString(Text) matches native ToString"), Result, Expected);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FPointBlueprintLibrary_PointToString_BoxMatchesNativeTest, "Schola.Points.Blueprint.PointBlueprintLibrary.PointToString.BoxMatchesNative", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPointBlueprintLibrary_PointToString_BoxMatchesNativeTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FPoint> Point;
    Point.InitializeAs<FBoxPoint>();
    Point.GetMutable<FBoxPoint>().Add(3.5f);

    const FString Expected = Point.Get<FBoxPoint>().ToString();
    const FString Result = UPointBlueprintLibrary::PointToString(ToUntypedInstancedStruct(Point));

    TestEqual(TEXT("PointToString(Box) matches native ToString"), Result, Expected);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FPointBlueprintLibrary_PointToString_MultiBinaryMatchesNativeTest, "Schola.Points.Blueprint.PointBlueprintLibrary.PointToString.MultiBinaryMatchesNative", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPointBlueprintLibrary_PointToString_MultiBinaryMatchesNativeTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FPoint> Point;
    Point.InitializeAs<FMultiBinaryPoint>();
    Point.GetMutable<FMultiBinaryPoint>().Values = {true, false, true};

    const FString Expected = Point.Get<FMultiBinaryPoint>().ToString();
    const FString Result = UPointBlueprintLibrary::PointToString(ToUntypedInstancedStruct(Point));

    TestEqual(TEXT("PointToString(MultiBinary) matches native ToString"), Result, Expected);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FPointBlueprintLibrary_PointToString_MultiDiscreteMatchesNativeTest, "Schola.Points.Blueprint.PointBlueprintLibrary.PointToString.MultiDiscreteMatchesNative", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPointBlueprintLibrary_PointToString_MultiDiscreteMatchesNativeTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FPoint> Point;
    Point.InitializeAs<FMultiDiscretePoint>();
    Point.GetMutable<FMultiDiscretePoint>().Values = {1, 2, 3};

    const FString Expected = Point.Get<FMultiDiscretePoint>().ToString();
    const FString Result = UPointBlueprintLibrary::PointToString(ToUntypedInstancedStruct(Point));

    TestEqual(TEXT("PointToString(MultiDiscrete) matches native ToString"), Result, Expected);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FPointBlueprintLibrary_PointToString_DictMatchesNativeTest, "Schola.Points.Blueprint.PointBlueprintLibrary.PointToString.DictMatchesNative", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPointBlueprintLibrary_PointToString_DictMatchesNativeTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FPoint> Point;
    Point.InitializeAs<FDictPoint>();

    const FString Expected = Point.Get<FDictPoint>().ToString();
    const FString Result = UPointBlueprintLibrary::PointToString(ToUntypedInstancedStruct(Point));

    TestEqual(TEXT("PointToString(Dict) matches native ToString"), Result, Expected);

    return true;
}

#endif




