// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Spaces/Blueprint/SpaceBlueprintLibrary.h"
#include "Spaces/BoxSpace.h"
#include "Spaces/DiscreteSpace.h"
#include "Spaces/MultiBinarySpace.h"
#include "Spaces/MultiDiscreteSpace.h"
#include "Spaces/DictSpace.h"
#include "Spaces/TextSpace.h"
#include "Common/InstancedStructUtils.h"
#include "Math/Vector.h"

#if WITH_DEV_AUTOMATION_TESTS

// Space_Type Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSpaceBlueprintLibrary_Type_BoxTest, "Schola.Spaces.Blueprint.SpaceBlueprintLibrary.Type.Box", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSpaceBlueprintLibrary_Type_BoxTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FSpace> Space;
    Space.InitializeAs<FBoxSpace>();
    FBoxSpace& BoxSpace = Space.GetMutable<FBoxSpace>();
    BoxSpace.Add(-1.0f, 1.0f);

    ESpaceType Result = USpaceBlueprintLibrary::Space_Type(ToUntypedInstancedStruct(Space));

    TestEqual(TEXT("Space_Type(BoxSpace) == ESpaceType::Box"), Result, ESpaceType::Box);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSpaceBlueprintLibrary_Type_DiscreteTest, "Schola.Spaces.Blueprint.SpaceBlueprintLibrary.Type.Discrete", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSpaceBlueprintLibrary_Type_DiscreteTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FSpace> Space;
    Space.InitializeAs<FDiscreteSpace>();
    FDiscreteSpace& DiscreteSpace = Space.GetMutable<FDiscreteSpace>();
    DiscreteSpace.High = 5;

    ESpaceType Result = USpaceBlueprintLibrary::Space_Type(ToUntypedInstancedStruct(Space));

    TestEqual(TEXT("Space_Type(DiscreteSpace) == ESpaceType::Discrete"), Result, ESpaceType::Discrete);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSpaceBlueprintLibrary_Type_MultiBinaryTest, "Schola.Spaces.Blueprint.SpaceBlueprintLibrary.Type.MultiBinary", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSpaceBlueprintLibrary_Type_MultiBinaryTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FSpace> Space;
    Space.InitializeAs<FMultiBinarySpace>();
    FMultiBinarySpace& MultiBinarySpace = Space.GetMutable<FMultiBinarySpace>();
    MultiBinarySpace.Shape = 8;

    ESpaceType Result = USpaceBlueprintLibrary::Space_Type(ToUntypedInstancedStruct(Space));

    TestEqual(TEXT("Space_Type(MultiBinarySpace) == ESpaceType::Binary"), Result, ESpaceType::MultiBinary);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSpaceBlueprintLibrary_Type_MultiDiscreteTest, "Schola.Spaces.Blueprint.SpaceBlueprintLibrary.Type.MultiDiscrete", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSpaceBlueprintLibrary_Type_MultiDiscreteTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FSpace> Space;
    Space.InitializeAs<FMultiDiscreteSpace>(TArray<int32>{2, 3, 4});
    FMultiDiscreteSpace& MultiDiscreteSpace = Space.GetMutable<FMultiDiscreteSpace>();

    ESpaceType Result = USpaceBlueprintLibrary::Space_Type(ToUntypedInstancedStruct(Space));

    TestEqual(TEXT("Space_Type(MultiDiscreteSpace) == ESpaceType::MultiDiscrete"), Result, ESpaceType::MultiDiscrete);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSpaceBlueprintLibrary_Type_DictTest, "Schola.Spaces.Blueprint.SpaceBlueprintLibrary.Type.Dict", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSpaceBlueprintLibrary_Type_DictTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FSpace> Space;
    Space.InitializeAs<FDictSpace>();

    ESpaceType Result = USpaceBlueprintLibrary::Space_Type(ToUntypedInstancedStruct(Space));

    TestEqual(TEXT("Space_Type(DictSpace) == ESpaceType::Dict"), Result, ESpaceType::Dict);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSpaceBlueprintLibrary_Type_TextTest, "Schola.Spaces.Blueprint.SpaceBlueprintLibrary.Type.Text", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSpaceBlueprintLibrary_Type_TextTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FSpace> Space;
    Space.InitializeAs<FTextSpace>(16);

    ESpaceType Result = USpaceBlueprintLibrary::Space_Type(ToUntypedInstancedStruct(Space));

    TestEqual(TEXT("Space_Type(TextSpace) == ESpaceType::Text"), Result, ESpaceType::Text);

    return true;
}

// Space_IsOfType Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSpaceBlueprintLibrary_IsOfType_BoxTrueTest, "Schola.Spaces.Blueprint.SpaceBlueprintLibrary.IsOfType.BoxTrue", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSpaceBlueprintLibrary_IsOfType_BoxTrueTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FSpace> Space;
    Space.InitializeAs<FBoxSpace>();

    bool Result = USpaceBlueprintLibrary::Space_IsOfType(ToUntypedInstancedStruct(Space), ESpaceType::Box);

    TestTrue(TEXT("Space_IsOfType(BoxSpace, Box) == true"), Result);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSpaceBlueprintLibrary_IsOfType_BoxFalseTest, "Schola.Spaces.Blueprint.SpaceBlueprintLibrary.IsOfType.BoxFalse", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSpaceBlueprintLibrary_IsOfType_BoxFalseTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FSpace> Space;
    Space.InitializeAs<FBoxSpace>();

    bool Result = USpaceBlueprintLibrary::Space_IsOfType(ToUntypedInstancedStruct(Space), ESpaceType::Discrete);

    TestFalse(TEXT("Space_IsOfType(BoxSpace, Discrete) == false"), Result);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSpaceBlueprintLibrary_IsOfType_DiscreteTrueTest, "Schola.Spaces.Blueprint.SpaceBlueprintLibrary.IsOfType.DiscreteTrue", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSpaceBlueprintLibrary_IsOfType_DiscreteTrueTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FSpace> Space;
    Space.InitializeAs<FDiscreteSpace>();

    bool Result = USpaceBlueprintLibrary::Space_IsOfType(ToUntypedInstancedStruct(Space), ESpaceType::Discrete);

    TestTrue(TEXT("Space_IsOfType(DiscreteSpace, Discrete) == true"), Result);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSpaceBlueprintLibrary_IsOfType_MultiBinaryTrueTest, "Schola.Spaces.Blueprint.SpaceBlueprintLibrary.IsOfType.MultiBinaryTrue", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSpaceBlueprintLibrary_IsOfType_MultiBinaryTrueTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FSpace> Space;
    Space.InitializeAs<FMultiBinarySpace>();

    bool Result = USpaceBlueprintLibrary::Space_IsOfType(ToUntypedInstancedStruct(Space), ESpaceType::MultiBinary);

    TestTrue(TEXT("Space_IsOfType(MultiBinarySpace, MultiBinary) == true"), Result);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSpaceBlueprintLibrary_IsOfType_MultiDiscreteTrueTest, "Schola.Spaces.Blueprint.SpaceBlueprintLibrary.IsOfType.MultiDiscreteTrue", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSpaceBlueprintLibrary_IsOfType_MultiDiscreteTrueTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FSpace> Space;
    Space.InitializeAs<FMultiDiscreteSpace>();

    bool Result = USpaceBlueprintLibrary::Space_IsOfType(ToUntypedInstancedStruct(Space), ESpaceType::MultiDiscrete);

    TestTrue(TEXT("Space_IsOfType(MultiDiscreteSpace, MultiDiscrete) == true"), Result);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSpaceBlueprintLibrary_IsOfType_DictTrueTest, "Schola.Spaces.Blueprint.SpaceBlueprintLibrary.IsOfType.DictTrue", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSpaceBlueprintLibrary_IsOfType_DictTrueTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FSpace> Space;
    Space.InitializeAs<FDictSpace>();

    bool Result = USpaceBlueprintLibrary::Space_IsOfType(ToUntypedInstancedStruct(Space), ESpaceType::Dict);

    TestTrue(TEXT("Space_IsOfType(DictSpace, Dict) == true"), Result);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSpaceBlueprintLibrary_IsOfType_TextTrueTest, "Schola.Spaces.Blueprint.SpaceBlueprintLibrary.IsOfType.TextTrue", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSpaceBlueprintLibrary_IsOfType_TextTrueTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FSpace> Space;
    Space.InitializeAs<FTextSpace>(16);

    bool Result = USpaceBlueprintLibrary::Space_IsOfType(ToUntypedInstancedStruct(Space), ESpaceType::Text);

    TestTrue(TEXT("Space_IsOfType(TextSpace, Text) == true"), Result);

    return true;
}

// SpaceToString Tests

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSpaceBlueprintLibrary_SpaceToString_InvalidTest, "Schola.Spaces.Blueprint.SpaceBlueprintLibrary.SpaceToString.Invalid", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSpaceBlueprintLibrary_SpaceToString_InvalidTest::RunTest(const FString& Parameters)
{
    FInstancedStruct Invalid;
    TestFalse(TEXT("Uninitialized instanced struct is invalid"), Invalid.IsValid());

    FString Result = USpaceBlueprintLibrary::SpaceToString(Invalid);
    TestTrue(TEXT("SpaceToString(invalid) returns empty"), Result.IsEmpty());

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSpaceBlueprintLibrary_SpaceToString_NonSpaceStructTest, "Schola.Spaces.Blueprint.SpaceBlueprintLibrary.SpaceToString.NonSpaceStruct", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSpaceBlueprintLibrary_SpaceToString_NonSpaceStructTest::RunTest(const FString& Parameters)
{
    FInstancedStruct NotASpace;
    NotASpace.InitializeAs<FVector>(FVector(1.0f, 2.0f, 3.0f));

    FString Result = USpaceBlueprintLibrary::SpaceToString(NotASpace);
    TestTrue(TEXT("SpaceToString(non-FSpace struct) returns empty"), Result.IsEmpty());

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSpaceBlueprintLibrary_SpaceToString_DiscreteMatchesNativeTest, "Schola.Spaces.Blueprint.SpaceBlueprintLibrary.SpaceToString.DiscreteMatchesNative", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSpaceBlueprintLibrary_SpaceToString_DiscreteMatchesNativeTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FSpace> Space;
    Space.InitializeAs<FDiscreteSpace>();
    Space.GetMutable<FDiscreteSpace>().High = 10;

    const FString Expected = Space.Get<FDiscreteSpace>().ToString();
    const FString Result = USpaceBlueprintLibrary::SpaceToString(ToUntypedInstancedStruct(Space));

    TestEqual(TEXT("SpaceToString(Discrete) matches native ToString"), Result, Expected);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSpaceBlueprintLibrary_SpaceToString_TextMatchesNativeTest, "Schola.Spaces.Blueprint.SpaceBlueprintLibrary.SpaceToString.TextMatchesNative", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSpaceBlueprintLibrary_SpaceToString_TextMatchesNativeTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FSpace> Space;
    Space.InitializeAs<FTextSpace>(16);

    const FString Expected = Space.Get<FTextSpace>().ToString();
    const FString Result = USpaceBlueprintLibrary::SpaceToString(ToUntypedInstancedStruct(Space));

    TestEqual(TEXT("SpaceToString(Text) matches native ToString"), Result, Expected);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSpaceBlueprintLibrary_SpaceToString_BoxMatchesNativeTest, "Schola.Spaces.Blueprint.SpaceBlueprintLibrary.SpaceToString.BoxMatchesNative", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSpaceBlueprintLibrary_SpaceToString_BoxMatchesNativeTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FSpace> Space;
    Space.InitializeAs<FBoxSpace>();
    Space.GetMutable<FBoxSpace>().Add(-1.0f, 1.0f);

    const FString Expected = Space.Get<FBoxSpace>().ToString();
    const FString Result = USpaceBlueprintLibrary::SpaceToString(ToUntypedInstancedStruct(Space));

    TestEqual(TEXT("SpaceToString(Box) matches native ToString"), Result, Expected);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSpaceBlueprintLibrary_SpaceToString_MultiBinaryMatchesNativeTest, "Schola.Spaces.Blueprint.SpaceBlueprintLibrary.SpaceToString.MultiBinaryMatchesNative", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSpaceBlueprintLibrary_SpaceToString_MultiBinaryMatchesNativeTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FSpace> Space;
    Space.InitializeAs<FMultiBinarySpace>();
    Space.GetMutable<FMultiBinarySpace>().Shape = 4;

    const FString Expected = Space.Get<FMultiBinarySpace>().ToString();
    const FString Result = USpaceBlueprintLibrary::SpaceToString(ToUntypedInstancedStruct(Space));

    TestEqual(TEXT("SpaceToString(MultiBinary) matches native ToString"), Result, Expected);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSpaceBlueprintLibrary_SpaceToString_MultiDiscreteMatchesNativeTest, "Schola.Spaces.Blueprint.SpaceBlueprintLibrary.SpaceToString.MultiDiscreteMatchesNative", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSpaceBlueprintLibrary_SpaceToString_MultiDiscreteMatchesNativeTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FSpace> Space;
    Space.InitializeAs<FMultiDiscreteSpace>(TArray<int32>{2, 3, 4});

    const FString Expected = Space.Get<FMultiDiscreteSpace>().ToString();
    const FString Result = USpaceBlueprintLibrary::SpaceToString(ToUntypedInstancedStruct(Space));

    TestEqual(TEXT("SpaceToString(MultiDiscrete) matches native ToString"), Result, Expected);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSpaceBlueprintLibrary_SpaceToString_DictMatchesNativeTest, "Schola.Spaces.Blueprint.SpaceBlueprintLibrary.SpaceToString.DictMatchesNative", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FSpaceBlueprintLibrary_SpaceToString_DictMatchesNativeTest::RunTest(const FString& Parameters)
{
    TInstancedStruct<FSpace> Space;
    Space.InitializeAs<FDictSpace>();

    const FString Expected = Space.Get<FDictSpace>().ToString();
    const FString Result = USpaceBlueprintLibrary::SpaceToString(ToUntypedInstancedStruct(Space));

    TestEqual(TEXT("SpaceToString(Dict) matches native ToString"), Result, Expected);

    return true;
}

#endif




