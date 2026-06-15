// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Points/DictPoint.h"
#include "Points/BoxPoint.h"
#include "Points/DiscretePoint.h"
#if WITH_AUTOMATION_TESTS
#define TestEqualExactFloat(TestMessage, Actual, Expected) TestEqual(TestMessage, (float)Actual, (float)Expected, 0.0001f)

// Constructor Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDictPointDefaultConstructorTest, "Schola.Points.DictPoint.Default Constructor Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FDictPointDefaultConstructorTest::RunTest(const FString& Parameters)
{
    FDictPoint DictPoint = FDictPoint();

    TestEqual(TEXT("DictPoint.Points.Num() == 0"), DictPoint.Points.Num(), 0);
    
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDictPointTMapConstructorTest, "Schola.Points.DictPoint.TMap Constructor Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FDictPointTMapConstructorTest::RunTest(const FString& Parameters)
{
    TMap<FString, TInstancedStruct<FPoint>> PointsMap;
    
    TInstancedStruct<FPoint> BoxPoint1;
    BoxPoint1.InitializeAs<FBoxPoint>(FBoxPoint({ 1.0f, 2.0f }));
    PointsMap.Add(TEXT("box1"), BoxPoint1);
    
    TInstancedStruct<FPoint> DiscretePoint1;
    DiscretePoint1.InitializeAs<FDiscretePoint>(FDiscretePoint(5));
    PointsMap.Add(TEXT("discrete1"), DiscretePoint1);
    
    FDictPoint DictPoint = FDictPoint(PointsMap);

    TestEqual(TEXT("DictPoint.Points.Num() == 2"), DictPoint.Points.Num(), 2);
    TestTrue(TEXT("DictPoint contains 'box1'"), DictPoint.Points.Contains(TEXT("box1")));
    TestTrue(TEXT("DictPoint contains 'discrete1'"), DictPoint.Points.Contains(TEXT("discrete1")));
    
    const FBoxPoint& RetrievedBoxPoint = DictPoint.Points[TEXT("box1")].Get<FBoxPoint>();
    TestEqualExactFloat(TEXT("RetrievedBoxPoint[0] == 1.0f"), RetrievedBoxPoint[0], 1.0f);
    TestEqualExactFloat(TEXT("RetrievedBoxPoint[1] == 2.0f"), RetrievedBoxPoint[1], 2.0f);
    
    const FDiscretePoint& RetrievedDiscretePoint = DictPoint.Points[TEXT("discrete1")].Get<FDiscretePoint>();
    TestEqual(TEXT("RetrievedDiscretePoint.Value == 5"), RetrievedDiscretePoint.Value, 5);
    
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDictPointInitializerListConstructorTest, "Schola.Points.DictPoint.InitializerList Constructor Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FDictPointInitializerListConstructorTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FPoint> Point1;
    Point1.InitializeAs<FBoxPoint>(FBoxPoint({ 3.0f, 4.0f, 5.0f }));
    
    TInstancedStruct<FPoint> Point2;
    Point2.InitializeAs<FDiscretePoint>(FDiscretePoint(10));
    
    FDictPoint DictPoint = FDictPoint({ 
        TPair<FString, TInstancedStruct<FPoint>>(TEXT("myBox"), Point1),
        TPair<FString, TInstancedStruct<FPoint>>(TEXT("myDiscrete"), Point2)
    });

    TestEqual(TEXT("DictPoint.Points.Num() == 2"), DictPoint.Points.Num(), 2);
    TestTrue(TEXT("DictPoint contains 'myBox'"), DictPoint.Points.Contains(TEXT("myBox")));
    TestTrue(TEXT("DictPoint contains 'myDiscrete'"), DictPoint.Points.Contains(TEXT("myDiscrete")));
    
    const FBoxPoint& RetrievedBoxPoint = DictPoint.Points[TEXT("myBox")].Get<FBoxPoint>();
    TestEqualExactFloat(TEXT("RetrievedBoxPoint[0] == 3.0f"), RetrievedBoxPoint[0], 3.0f);
    TestEqualExactFloat(TEXT("RetrievedBoxPoint[1] == 4.0f"), RetrievedBoxPoint[1], 4.0f);
    TestEqualExactFloat(TEXT("RetrievedBoxPoint[2] == 5.0f"), RetrievedBoxPoint[2], 5.0f);
    
    const FDiscretePoint& RetrievedDiscretePoint = DictPoint.Points[TEXT("myDiscrete")].Get<FDiscretePoint>();
    TestEqual(TEXT("RetrievedDiscretePoint.Value == 10"), RetrievedDiscretePoint.Value, 10);
    
    return true;
}

// Method Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDictPointAddExistingPointTest, "Schola.Points.DictPoint.Add Existing Point Test ", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FDictPointAddExistingPointTest::RunTest(const FString& Parameters)
{
    FDictPoint DictPoint = FDictPoint();

    FBoxPoint MyBoxPoint;
    MyBoxPoint.Add(1.0f);
    MyBoxPoint.Add(2.0f);

    TInstancedStruct<FPoint> InstancedBoxPoint;
    InstancedBoxPoint.InitializeAs<FBoxPoint>(MyBoxPoint);
    DictPoint.Points.Add(TEXT("TestPoint"), MoveTemp(InstancedBoxPoint));

    const TInstancedStruct<FPoint>& RetrievedPoint = DictPoint.Points[TEXT("TestPoint")];
    const FBoxPoint& RetrievedBoxPoint = RetrievedPoint.Get<FBoxPoint>();

    TestEqualExactFloat(TEXT("RetrievedBoxPoint[0] == 1.0f"), RetrievedBoxPoint[0], 1.0f);
    TestEqualExactFloat(TEXT("RetrievedBoxPoint[1] == 2.0f"), RetrievedBoxPoint[1], 2.0f);
    
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDictPointEmplacePointTest, "Schola.Points.DictPoint.Emplace Point Test ", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FDictPointEmplacePointTest::RunTest(const FString& Parameters)
{
    FDictPoint DictPoint = FDictPoint();
    
    TInstancedStruct<FPoint> InstancedBoxPoint;
    InstancedBoxPoint.InitializeAs<FBoxPoint>();
    DictPoint.Points.Emplace(TEXT("EmptyPoint"), MoveTemp(InstancedBoxPoint));

    const TInstancedStruct<FPoint>& RetrievedPoint = DictPoint.Points[TEXT("EmptyPoint")];
    const FBoxPoint& RetrievedBoxPoint = RetrievedPoint.Get<FBoxPoint>();

    TestEqual(TEXT("RetrievedBoxPoint.Values.Num() == 0"), RetrievedBoxPoint.Values.Num(), 0);
    
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDictPointAddMultiplePointsTest, "Schola.Points.DictPoint.Add Multiple Points Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FDictPointAddMultiplePointsTest::RunTest(const FString& Parameters)
{
    FDictPoint DictPoint = FDictPoint();

    TInstancedStruct<FPoint> Point1;
    Point1.InitializeAs<FBoxPoint>(FBoxPoint({ 1.0f, 2.0f }));
    DictPoint.Points.Add(TEXT("Point1"), MoveTemp(Point1));

    TInstancedStruct<FPoint> Point2;
    Point2.InitializeAs<FBoxPoint>(FBoxPoint({ 3.0f, 4.0f }));
    DictPoint.Points.Add(TEXT("Point2"), MoveTemp(Point2));

    TestEqual(TEXT("DictPoint should contain 2 points"), DictPoint.Points.Num(), 2);

    const FBoxPoint& RetrievedPoint1 = DictPoint.Points[TEXT("Point1")].Get<FBoxPoint>();
    TestEqualExactFloat(TEXT("RetrievedPoint1[0] == 1.0f"), RetrievedPoint1[0], 1.0f);
    TestEqualExactFloat(TEXT("RetrievedPoint1[1] == 2.0f"), RetrievedPoint1[1], 2.0f);

    const FBoxPoint& RetrievedPoint2 = DictPoint.Points[TEXT("Point2")].Get<FBoxPoint>();
    TestEqualExactFloat(TEXT("RetrievedPoint2[0] == 3.0f"), RetrievedPoint2[0], 3.0f);
    TestEqualExactFloat(TEXT("RetrievedPoint2[1] == 4.0f"), RetrievedPoint2[1], 4.0f);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDictPointOverwriteExistingPointTest, "Schola.Points.DictPoint.Overwrite Existing Point Test", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FDictPointOverwriteExistingPointTest::RunTest(const FString& Parameters)
{
    FDictPoint DictPoint = FDictPoint();

    TInstancedStruct<FPoint> OriginalPoint;
    OriginalPoint.InitializeAs<FBoxPoint>(FBoxPoint({ 10.0f }));
    DictPoint.Points.Add(TEXT("OverwritingPoint"), MoveTemp(OriginalPoint));

    TestEqual(TEXT("DictPoint should contain 1 point initially"), DictPoint.Points.Num(), 1);
    const FBoxPoint& RetrievedOriginalPoint = DictPoint.Points[TEXT("OverwritingPoint")].Get<FBoxPoint>();
    TestEqualExactFloat(TEXT("Original point value should be 10.0f"), RetrievedOriginalPoint[0], 10.0f);

    TInstancedStruct<FPoint> OverwritingPoint;
    OverwritingPoint.InitializeAs<FBoxPoint>(FBoxPoint({ 20.0f }));
    DictPoint.Points.Emplace(TEXT("OverwritingPoint"), MoveTemp(OverwritingPoint));

    TestEqual(TEXT("DictPoint should still contain 1 point after overwrite"), DictPoint.Points.Num(), 1);
    const FBoxPoint& RetrievedOverwrittenPoint = DictPoint.Points[TEXT("OverwritingPoint")].Get<FBoxPoint>();
    TestEqualExactFloat(TEXT("Overwritten point value should be 20.0f"), RetrievedOverwrittenPoint[0], 20.0f);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDictPointToStringEmptyTest, "Schola.Points.DictPoint.ToString.Empty", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FDictPointToStringEmptyTest::RunTest(const FString& Parameters)
{
	const FDictPoint DictPoint;
	TestEqual(TEXT("FDictPoint::ToString empty map"), DictPoint.ToString(), FString(TEXT("{\n}")));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FDictPointToStringSingleEntryTest, "Schola.Points.DictPoint.ToString.SingleEntry", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FDictPointToStringSingleEntryTest::RunTest(const FString& Parameters)
{
	FDictPoint DictPoint;
	TInstancedStruct<FPoint> Inner;
	Inner.InitializeAs<FDiscretePoint>(7);
	DictPoint.Points.Add(TEXT("alpha"), MoveTemp(Inner));

	const FString Expected = FString(TEXT("{\n    alpha: 7,\n}"));
	TestEqual(TEXT("FDictPoint::ToString single key"), DictPoint.ToString(), Expected);
	return true;
}
#endif