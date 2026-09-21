// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Spaces/Blueprint/BoxSpaceBlueprintLibrary.h"
#include "Spaces/BoxSpace.h"

#if WITH_DEV_AUTOMATION_TESTS

#define TestEqualExactFloat(TestMessage, Actual, Expected) TestEqual(TestMessage, (float)Actual, (float)Expected, 0.0001f)

// ArraysToBoxSpace Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceBlueprintLibrary_ArraysToBoxSpace_BasicTest, "Schola.Spaces.Blueprint.BoxSpaceBlueprintLibrary.ArraysToBoxSpace.Basic", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceBlueprintLibrary_ArraysToBoxSpace_BasicTest::RunTest(const FString& Parameters)
{
    TArray<float> Low = {-1.0f, -2.0f, -3.0f};
    TArray<float> High = {1.0f, 2.0f, 3.0f};
    TArray<int32> Shape = {3};

    TInstancedStruct<FBoxSpace> Result = UBoxSpaceBlueprintLibrary::ArraysToBoxSpace(Low, High, Shape);

    TestTrue(TEXT("Result is valid"), Result.IsValid());
    
    const FBoxSpace& BoxSpace = Result.Get<FBoxSpace>();
    TestEqual(TEXT("BoxSpace.Shape"), BoxSpace.Shape, Shape);
    const TArray<FBoxSpaceDimension> ExpectedDimensions = {
        FBoxSpaceDimension(Low[0], High[0]),
        FBoxSpaceDimension(Low[1], High[1]),
        FBoxSpaceDimension(Low[2], High[2]),
    };
    TestEqual(TEXT("BoxSpace.Dimensions"), BoxSpace.Dimensions, ExpectedDimensions);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceBlueprintLibrary_ArraysToBoxSpace_WithShapeTest, "Schola.Spaces.Blueprint.BoxSpaceBlueprintLibrary.ArraysToBoxSpace.WithShape", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceBlueprintLibrary_ArraysToBoxSpace_WithShapeTest::RunTest(const FString& Parameters)
{
    TArray<float> Low = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    TArray<float> High = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    TArray<int32> Shape = {2, 3};

    TInstancedStruct<FBoxSpace> Result = UBoxSpaceBlueprintLibrary::ArraysToBoxSpace(Low, High, Shape);

    TestTrue(TEXT("Result is valid"), Result.IsValid());
    
    const FBoxSpace& BoxSpace = Result.Get<FBoxSpace>();
    TestEqual(TEXT("BoxSpace.Dimensions.Num() == 6"), BoxSpace.Dimensions.Num(), 6);
    TestEqual(TEXT("BoxSpace.Shape"), BoxSpace.Shape, Shape);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceBlueprintLibrary_ArraysToBoxSpace_EmptyTest, "Schola.Spaces.Blueprint.BoxSpaceBlueprintLibrary.ArraysToBoxSpace.Empty", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceBlueprintLibrary_ArraysToBoxSpace_EmptyTest::RunTest(const FString& Parameters)
{
    TArray<float> Low;
    TArray<float> High;
    TArray<int32> Shape;

    TInstancedStruct<FBoxSpace> Result = UBoxSpaceBlueprintLibrary::ArraysToBoxSpace(Low, High, Shape);

    TestTrue(TEXT("Result is valid"), Result.IsValid());
    
    const FBoxSpace& BoxSpace = Result.Get<FBoxSpace>();
    TestEqual(TEXT("BoxSpace.Dimensions.Num() == 0"), BoxSpace.Dimensions.Num(), 0);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceBlueprintLibrary_ArraysToBoxSpace_NegativeRangeTest, "Schola.Spaces.Blueprint.BoxSpaceBlueprintLibrary.ArraysToBoxSpace.NegativeRange", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceBlueprintLibrary_ArraysToBoxSpace_NegativeRangeTest::RunTest(const FString& Parameters)
{
    TArray<float> Low = {-10.0f, -20.0f};
    TArray<float> High = {-5.0f, -10.0f};
    TArray<int32> Shape = {2};

    TInstancedStruct<FBoxSpace> Result = UBoxSpaceBlueprintLibrary::ArraysToBoxSpace(Low, High, Shape);

    TestTrue(TEXT("Result is valid"), Result.IsValid());
    
    const FBoxSpace& BoxSpace = Result.Get<FBoxSpace>();
    const TArray<FBoxSpaceDimension> ExpectedDimensions = {
        FBoxSpaceDimension(Low[0], High[0]),
        FBoxSpaceDimension(Low[1], High[1]),
    };
    TestEqual(TEXT("BoxSpace.Dimensions"), BoxSpace.Dimensions, ExpectedDimensions);

    return true;
}

// VectorToBoxSpace Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceBlueprintLibrary_VectorToBoxSpace_BasicTest, "Schola.Spaces.Blueprint.BoxSpaceBlueprintLibrary.VectorToBoxSpace.Basic", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceBlueprintLibrary_VectorToBoxSpace_BasicTest::RunTest(const FString& Parameters)
{
    FVector Low(-100.0f, -200.0f, -300.0f);
    FVector High(100.0f, 200.0f, 300.0f);

    TInstancedStruct<FBoxSpace> Result = UBoxSpaceBlueprintLibrary::VectorToBoxSpace(Low, High);

    TestTrue(TEXT("Result is valid"), Result.IsValid());
    
    const FBoxSpace& BoxSpace = Result.Get<FBoxSpace>();
    TestEqual(TEXT("BoxSpace.Shape"), BoxSpace.Shape, TArray<int>({3}));
    const TArray<FBoxSpaceDimension> ExpectedDimensions = {
        FBoxSpaceDimension(Low.X, High.X),
        FBoxSpaceDimension(Low.Y, High.Y),
        FBoxSpaceDimension(Low.Z, High.Z),
    };
    TestEqual(TEXT("BoxSpace.Dimensions"), BoxSpace.Dimensions, ExpectedDimensions);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceBlueprintLibrary_VectorToBoxSpace_ZeroToOneTest, "Schola.Spaces.Blueprint.BoxSpaceBlueprintLibrary.VectorToBoxSpace.ZeroToOne", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceBlueprintLibrary_VectorToBoxSpace_ZeroToOneTest::RunTest(const FString& Parameters)
{
    FVector Low = FVector::ZeroVector;
    FVector High = FVector::OneVector;

    TInstancedStruct<FBoxSpace> Result = UBoxSpaceBlueprintLibrary::VectorToBoxSpace(Low, High);

    TestTrue(TEXT("Result is valid"), Result.IsValid());
    
    const FBoxSpace& BoxSpace = Result.Get<FBoxSpace>();
    const TArray<FBoxSpaceDimension> ExpectedDimensions = {
        FBoxSpaceDimension(0.f, 1.f),
        FBoxSpaceDimension(0.f, 1.f),
        FBoxSpaceDimension(0.f, 1.f),
    };
    TestEqual(TEXT("BoxSpace.Dimensions"), BoxSpace.Dimensions, ExpectedDimensions);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceBlueprintLibrary_VectorToBoxSpace_AsymmetricTest, "Schola.Spaces.Blueprint.BoxSpaceBlueprintLibrary.VectorToBoxSpace.Asymmetric", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceBlueprintLibrary_VectorToBoxSpace_AsymmetricTest::RunTest(const FString& Parameters)
{
    FVector Low(-10.5f, 0.0f, -50.0f);
    FVector High(20.5f, 100.0f, 50.0f);

    TInstancedStruct<FBoxSpace> Result = UBoxSpaceBlueprintLibrary::VectorToBoxSpace(Low, High);

    TestTrue(TEXT("Result is valid"), Result.IsValid());
    
    const FBoxSpace& BoxSpace = Result.Get<FBoxSpace>();
    const TArray<FBoxSpaceDimension> ExpectedDimensions = {
        FBoxSpaceDimension(Low.X, High.X),
        FBoxSpaceDimension(Low.Y, High.Y),
        FBoxSpaceDimension(Low.Z, High.Z),
    };
    TestEqual(TEXT("BoxSpace.Dimensions"), BoxSpace.Dimensions, ExpectedDimensions);

    return true;
}

// RotatorSpace Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceBlueprintLibrary_RotatorSpace_BasicTest, "Schola.Spaces.Blueprint.BoxSpaceBlueprintLibrary.RotatorSpace.Basic", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceBlueprintLibrary_RotatorSpace_BasicTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FBoxSpace> Result = UBoxSpaceBlueprintLibrary::RotatorSpace();

    TestTrue(TEXT("Result is valid"), Result.IsValid());
    
    const FBoxSpace& BoxSpace = Result.Get<FBoxSpace>();
    TestEqual(TEXT("BoxSpace.Shape"), BoxSpace.Shape, TArray<int>({3}));
    const TArray<FBoxSpaceDimension> ExpectedDimensions = {
        FBoxSpaceDimension(-180.f, 180.f),
        FBoxSpaceDimension(-180.f, 180.f),
        FBoxSpaceDimension(-180.f, 180.f),
    };
    TestEqual(TEXT("BoxSpace.Dimensions"), BoxSpace.Dimensions, ExpectedDimensions);

    return true;
}

// TransformToBoxSpace Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceBlueprintLibrary_TransformToBoxSpace_BasicTest, "Schola.Spaces.Blueprint.BoxSpaceBlueprintLibrary.TransformToBoxSpace.Basic", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceBlueprintLibrary_TransformToBoxSpace_BasicTest::RunTest(const FString& Parameters)
{
    FVector LocationLow(-100.0f, -100.0f, -100.0f);
    FVector LocationHigh(100.0f, 100.0f, 100.0f);
    FVector ScaleLow(0.1f, 0.1f, 0.1f);
    FVector ScaleHigh(10.0f, 10.0f, 10.0f);

    TInstancedStruct<FBoxSpace> Result = UBoxSpaceBlueprintLibrary::TransformToBoxSpace(
        LocationLow, LocationHigh,
        ScaleLow, ScaleHigh
    );

    TestTrue(TEXT("Result is valid"), Result.IsValid());
    
    const FBoxSpace& BoxSpace = Result.Get<FBoxSpace>();
    TestEqual(TEXT("BoxSpace.Shape"), BoxSpace.Shape, TArray<int>({9}));
    const TArray<FBoxSpaceDimension> ExpectedDimensions = {
        FBoxSpaceDimension(LocationLow.X, LocationHigh.X),
        FBoxSpaceDimension(LocationLow.Y, LocationHigh.Y),
        FBoxSpaceDimension(LocationLow.Z, LocationHigh.Z),
        FBoxSpaceDimension(-180.f, 180.f),
        FBoxSpaceDimension(-180.f, 180.f),
        FBoxSpaceDimension(-180.f, 180.f),
        FBoxSpaceDimension(ScaleLow.X, ScaleHigh.X),
        FBoxSpaceDimension(ScaleLow.Y, ScaleHigh.Y),
        FBoxSpaceDimension(ScaleLow.Z, ScaleHigh.Z),
    };
    TestEqual(TEXT("BoxSpace.Dimensions"), BoxSpace.Dimensions, ExpectedDimensions);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceBlueprintLibrary_TransformToBoxSpace_IdentityRangeTest, "Schola.Spaces.Blueprint.BoxSpaceBlueprintLibrary.TransformToBoxSpace.IdentityRange", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceBlueprintLibrary_TransformToBoxSpace_IdentityRangeTest::RunTest(const FString& Parameters)
{
    FVector LocationLow = FVector::ZeroVector;
    FVector LocationHigh = FVector::ZeroVector;
    FVector ScaleLow = FVector::OneVector;
    FVector ScaleHigh = FVector::OneVector;

    TInstancedStruct<FBoxSpace> Result = UBoxSpaceBlueprintLibrary::TransformToBoxSpace(
        LocationLow, LocationHigh,
        ScaleLow, ScaleHigh
    );

    TestTrue(TEXT("Result is valid"), Result.IsValid());
    
    const FBoxSpace& BoxSpace = Result.Get<FBoxSpace>();
    const TArray<FBoxSpaceDimension> ExpectedDimensions = {
        FBoxSpaceDimension(0.f, 0.f),
        FBoxSpaceDimension(0.f, 0.f),
        FBoxSpaceDimension(0.f, 0.f),
        FBoxSpaceDimension(-180.f, 180.f),
        FBoxSpaceDimension(-180.f, 180.f),
        FBoxSpaceDimension(-180.f, 180.f),
        FBoxSpaceDimension(1.f, 1.f),
        FBoxSpaceDimension(1.f, 1.f),
        FBoxSpaceDimension(1.f, 1.f),
    };
    TestEqual(TEXT("BoxSpace.Dimensions"), BoxSpace.Dimensions, ExpectedDimensions);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceBlueprintLibrary_TransformToBoxSpace_AsymmetricTest, "Schola.Spaces.Blueprint.BoxSpaceBlueprintLibrary.TransformToBoxSpace.Asymmetric", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceBlueprintLibrary_TransformToBoxSpace_AsymmetricTest::RunTest(const FString& Parameters)
{
    FVector LocationLow(-50.0f, 0.0f, -100.0f);
    FVector LocationHigh(150.0f, 200.0f, 100.0f);
    FVector ScaleLow(0.5f, 0.5f, 0.5f);
    FVector ScaleHigh(2.0f, 3.0f, 4.0f);

    TInstancedStruct<FBoxSpace> Result = UBoxSpaceBlueprintLibrary::TransformToBoxSpace(
        LocationLow, LocationHigh,
        ScaleLow, ScaleHigh
    );

    TestTrue(TEXT("Result is valid"), Result.IsValid());
    
    const FBoxSpace& BoxSpace = Result.Get<FBoxSpace>();
    const TArray<FBoxSpaceDimension> ExpectedDimensions = {
        FBoxSpaceDimension(LocationLow.X, LocationHigh.X),
        FBoxSpaceDimension(LocationLow.Y, LocationHigh.Y),
        FBoxSpaceDimension(LocationLow.Z, LocationHigh.Z),
        FBoxSpaceDimension(-180.f, 180.f),
        FBoxSpaceDimension(-180.f, 180.f),
        FBoxSpaceDimension(-180.f, 180.f),
        FBoxSpaceDimension(ScaleLow.X, ScaleHigh.X),
        FBoxSpaceDimension(ScaleLow.Y, ScaleHigh.Y),
        FBoxSpaceDimension(ScaleLow.Z, ScaleHigh.Z),
    };
    TestEqual(TEXT("BoxSpace.Dimensions"), BoxSpace.Dimensions, ExpectedDimensions);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceBlueprintLibrary_TransformToBoxSpace_NegativeScaleTest, "Schola.Spaces.Blueprint.BoxSpaceBlueprintLibrary.TransformToBoxSpace.NegativeScale", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceBlueprintLibrary_TransformToBoxSpace_NegativeScaleTest::RunTest(const FString& Parameters)
{
    FVector LocationLow = FVector::ZeroVector;
    FVector LocationHigh = FVector::ZeroVector;
    FVector ScaleLow(-2.0f, -2.0f, -2.0f);
    FVector ScaleHigh(2.0f, 2.0f, 2.0f);

    TInstancedStruct<FBoxSpace> Result = UBoxSpaceBlueprintLibrary::TransformToBoxSpace(
        LocationLow, LocationHigh,
        ScaleLow, ScaleHigh
    );

    TestTrue(TEXT("Result is valid"), Result.IsValid());
    
    const FBoxSpace& BoxSpace = Result.Get<FBoxSpace>();
    const TArray<FBoxSpaceDimension> ExpectedDimensions = {
        FBoxSpaceDimension(0.f, 0.f),
        FBoxSpaceDimension(0.f, 0.f),
        FBoxSpaceDimension(0.f, 0.f),
        FBoxSpaceDimension(-180.f, 180.f),
        FBoxSpaceDimension(-180.f, 180.f),
        FBoxSpaceDimension(-180.f, 180.f),
        FBoxSpaceDimension(ScaleLow.X, ScaleHigh.X),
        FBoxSpaceDimension(ScaleLow.Y, ScaleHigh.Y),
        FBoxSpaceDimension(ScaleLow.Z, ScaleHigh.Z),
    };
    TestEqual(TEXT("BoxSpace.Dimensions"), BoxSpace.Dimensions, ExpectedDimensions);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceBlueprintLibrary_MakeUnboundedBoxSpaceTest, "Schola.Spaces.Blueprint.BoxSpaceBlueprintLibrary.MakeUnboundedBoxSpace", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceBlueprintLibrary_MakeUnboundedBoxSpaceTest::RunTest(const FString& Parameters)
{
    TArray<int32> Shape = {2, 2};
    TInstancedStruct<FBoxSpace> Result = UBoxSpaceBlueprintLibrary::MakeUnboundedBoxSpace(Shape);

    TestTrue(TEXT("Result is valid"), Result.IsValid());
    const FBoxSpace& BoxSpace = Result.Get<FBoxSpace>();
    TestEqual(TEXT("BoxSpace.Shape"), BoxSpace.Shape, Shape);
    TestEqual(TEXT("BoxSpace.Dimensions.Num() == 4"), BoxSpace.Dimensions.Num(), 4);
    for (const FBoxSpaceDimension& Dimension : BoxSpace.Dimensions)
    {
        TestTrue(TEXT("Dimension is unbounded"), Dimension.IsUnbounded());
    }

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBoxSpaceBlueprintLibrary_MakeHalfBoundedBoxSpaceDimensionTest, "Schola.Spaces.Blueprint.BoxSpaceBlueprintLibrary.MakeHalfBoundedBoxSpaceDimension", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FBoxSpaceBlueprintLibrary_MakeHalfBoundedBoxSpaceDimensionTest::RunTest(const FString& Parameters)
{
    TestEqual(
        TEXT("MakeLowerBoundedBoxSpaceDimension"),
        UBoxSpaceBlueprintLibrary::MakeLowerBoundedBoxSpaceDimension(0.0f),
        FBoxSpaceDimension::LowerBounded(0.0f));
    TestEqual(
        TEXT("MakeUpperBoundedBoxSpaceDimension"),
        UBoxSpaceBlueprintLibrary::MakeUpperBoundedBoxSpaceDimension(2.0f),
        FBoxSpaceDimension::UpperBounded(2.0f));

    return true;
}


#endif




