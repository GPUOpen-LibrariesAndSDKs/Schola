// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Sensors/FakeCameraSensor.h"

#if WITH_DEV_AUTOMATION_TESTS

static UFakeCameraSensor* CreateFakeCameraSensor(int32 NumChannels, int32 Width, int32 Height)
{
	UFakeCameraSensor* Sensor = NewObject<UFakeCameraSensor>();
	Sensor->NumChannels = NumChannels;
	Sensor->Width = Width;
	Sensor->Height = Height;
	return Sensor;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FFakeCameraSensorObservationSpace_Default_Test,
	"Schola.Sensors.FakeCameraSensor.ObservationSpace.Default",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FFakeCameraSensorObservationSpace_Default_Test::RunTest(const FString& Parameters)
{
	UFakeCameraSensor* Sensor = NewObject<UFakeCameraSensor>();

	FInstancedStruct ObservationSpace;
	Sensor->GetObservationSpace_Implementation(ObservationSpace);

	TestTrue(TEXT("ObservationSpace should be a BoxSpace"), ObservationSpace.GetScriptStruct() == FBoxSpace::StaticStruct());

	const FBoxSpace& Space = ObservationSpace.Get<FBoxSpace>();
	TestEqual(TEXT("Shape should have 3 dimensions"), Space.Shape.Num(), 3);
	TestEqual(TEXT("Shape[0] should be 3 (channels)"), Space.Shape[0], 3);
	TestEqual(TEXT("Shape[1] should be 128 (height)"), Space.Shape[1], 128);
	TestEqual(TEXT("Shape[2] should be 128 (width)"), Space.Shape[2], 128);
	TestEqual(TEXT("Total dimensions should be 49152"), Space.Dimensions.Num(), 49152);

	for (int i = 0; i < Space.Dimensions.Num(); i++)
	{
		TestEqual(TEXT("Dimension low should be 0.0"), Space.Dimensions[i].Low, 0.0f);
		TestEqual(TEXT("Dimension high should be 1.0"), Space.Dimensions[i].High, 1.0f);
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FFakeCameraSensorObservationSpace_CustomDims_Test,
	"Schola.Sensors.FakeCameraSensor.ObservationSpace.CustomDims",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FFakeCameraSensorObservationSpace_CustomDims_Test::RunTest(const FString& Parameters)
{
	UFakeCameraSensor* Sensor = CreateFakeCameraSensor(4, 320, 240);

	FInstancedStruct ObservationSpace;
	Sensor->GetObservationSpace_Implementation(ObservationSpace);

	const FBoxSpace& Space = ObservationSpace.Get<FBoxSpace>();
	TestEqual(TEXT("Shape[0] should be 4 (channels)"), Space.Shape[0], 4);
	TestEqual(TEXT("Shape[1] should be 240 (height)"), Space.Shape[1], 240);
	TestEqual(TEXT("Shape[2] should be 320 (width)"), Space.Shape[2], 320);
	TestEqual(TEXT("Total dimensions should be 307200"), Space.Dimensions.Num(), 4 * 240 * 320);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FFakeCameraSensorCollectObservations_DefaultZeros_Test,
	"Schola.Sensors.FakeCameraSensor.CollectObservations.DefaultZeros",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FFakeCameraSensorCollectObservations_DefaultZeros_Test::RunTest(const FString& Parameters)
{
	UFakeCameraSensor* Sensor = CreateFakeCameraSensor(2, 4, 3);

	FInstancedStruct Observations;
	Sensor->CollectObservations_Implementation(Observations);

	TestTrue(TEXT("Observation should be a BoxPoint"), Observations.GetScriptStruct() == FBoxPoint::StaticStruct());

	const FBoxPoint& Point = Observations.Get<FBoxPoint>();
	TestEqual(TEXT("Shape should have 3 dimensions"), Point.Shape.Num(), 3);
	TestEqual(TEXT("Shape[0] should be 2 (channels)"), Point.Shape[0], 2);
	TestEqual(TEXT("Shape[1] should be 3 (height)"), Point.Shape[1], 3);
	TestEqual(TEXT("Shape[2] should be 4 (width)"), Point.Shape[2], 4);
	TestEqual(TEXT("Values length should match channels * height * width"), Point.Values.Num(), 24);

	for (int32 Index = 0; Index < Point.Values.Num(); ++Index)
	{
		TestEqual(FString::Printf(TEXT("Default value at %d should be 0"), Index), Point.Values[Index], 0.0f);
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FFakeCameraSensorCollectObservations_CustomImage_Test,
	"Schola.Sensors.FakeCameraSensor.CollectObservations.CustomImage",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FFakeCameraSensorCollectObservations_CustomImage_Test::RunTest(const FString& Parameters)
{
	UFakeCameraSensor* Sensor = CreateFakeCameraSensor(2, 2, 2);
	Sensor->CustomImage = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f };

	FInstancedStruct Observations;
	Sensor->CollectObservations_Implementation(Observations);

	const FBoxPoint& Point = Observations.Get<FBoxPoint>();
	TestEqual(TEXT("Values length should match custom image"), Point.Values.Num(), 8);
	TestEqual(TEXT("Shape[0] should be 2 (channels)"), Point.Shape[0], 2);
	TestEqual(TEXT("Shape[1] should be 2 (height)"), Point.Shape[1], 2);
	TestEqual(TEXT("Shape[2] should be 2 (width)"), Point.Shape[2], 2);
	TestEqual(TEXT("Returned values should match CustomImage"), Point.Values, Sensor->CustomImage);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FFakeCameraSensorCollectObservations_CustomImageShorter_Test,
	"Schola.Sensors.FakeCameraSensor.CollectObservations.CustomImageShorter",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FFakeCameraSensorCollectObservations_CustomImageShorter_Test::RunTest(const FString& Parameters)
{
	AddExpectedError(TEXT("CustomImage size"), EAutomationExpectedErrorFlags::Contains, 1);

	UFakeCameraSensor* Sensor = CreateFakeCameraSensor(1, 2, 2);
	Sensor->CustomImage = { 0.25f, 0.5f };

	FInstancedStruct Observations;
	Sensor->CollectObservations_Implementation(Observations);

	const FBoxPoint& Point = Observations.Get<FBoxPoint>();
	TestEqual(TEXT("Values length should match expected size"), Point.Values.Num(), 4);
	TestEqual(TEXT("Copied prefix [0]"), Point.Values[0], 0.25f);
	TestEqual(TEXT("Copied prefix [1]"), Point.Values[1], 0.5f);
	TestEqual(TEXT("Padded [2] should be 0"), Point.Values[2], 0.0f);
	TestEqual(TEXT("Padded [3] should be 0"), Point.Values[3], 0.0f);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FFakeCameraSensorCollectObservations_CustomImageLonger_Test,
	"Schola.Sensors.FakeCameraSensor.CollectObservations.CustomImageLonger",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FFakeCameraSensorCollectObservations_CustomImageLonger_Test::RunTest(const FString& Parameters)
{
	AddExpectedError(TEXT("CustomImage size"), EAutomationExpectedErrorFlags::Contains, 1);

	UFakeCameraSensor* Sensor = CreateFakeCameraSensor(1, 2, 1);
	Sensor->CustomImage = { 0.1f, 0.2f, 0.3f, 0.4f };

	FInstancedStruct Observations;
	Sensor->CollectObservations_Implementation(Observations);

	const FBoxPoint& Point = Observations.Get<FBoxPoint>();
	TestEqual(TEXT("Values length should match expected size"), Point.Values.Num(), 2);
	TestEqual(TEXT("Truncated [0]"), Point.Values[0], 0.1f);
	TestEqual(TEXT("Truncated [1]"), Point.Values[1], 0.2f);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FFakeCameraSensorGenerateId_Test,
	"Schola.Sensors.FakeCameraSensor.GenerateId",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FFakeCameraSensorGenerateId_Test::RunTest(const FString& Parameters)
{
	UFakeCameraSensor* Sensor = CreateFakeCameraSensor(3, 64, 32);
	TestEqual(TEXT("Id should encode channels, width, and height"), Sensor->GenerateId(), FString(TEXT("FakeCamera_C3_W64_H32")));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FFakeCameraSensorGetObservationSize_Test,
	"Schola.Sensors.FakeCameraSensor.GetObservationSize",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FFakeCameraSensorGetObservationSize_Test::RunTest(const FString& Parameters)
{
	UFakeCameraSensor* Sensor = CreateFakeCameraSensor(3, 10, 8);
	TestEqual(TEXT("Observation size should be C*H*W"), Sensor->GetObservationSize(), 240);
	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
