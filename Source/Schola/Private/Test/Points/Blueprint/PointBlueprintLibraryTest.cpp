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

#endif




