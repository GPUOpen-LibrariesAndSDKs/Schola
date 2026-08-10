// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "Test/Sensors/CameraSensorTestHelpers.h"

#include "SensorInterface.h"
#include "Sensors/CameraSensor.h"
#include "Sensors/CameraSensorUtils.h"
#include "Points/BoxPoint.h"
#include "Spaces/BoxSpace.h"

#if WITH_DEV_AUTOMATION_TESTS

using namespace ScholaCameraSensorTest;

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FCameraSensorIntegration_IScholaSensorCollectFlow_Test,
	"Schola.Sensors.CameraSensor.Integration.IScholaSensor.CollectFlow",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter | EAutomationTestFlags::NonNullRHI)

bool FCameraSensorIntegration_IScholaSensorCollectFlow_Test::RunTest(const FString& Parameters)
{
	FScholaCameraSensorTestWorld TestWorld;
	if (!TestWorld.Setup(this))
	{
		return false;
	}

	UCameraSensor* Sensor = NewObject<UCameraSensor>(GetTransientPackage());
	Sensor->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
	Sensor->EnabledChannels = 15;
	Sensor->TextureTarget = CreateInitializedRenderTarget(Sensor, 8, 8);

	if (!FillRenderTargetSolidColor(TestWorld.GetWorld(), Sensor->TextureTarget, FLinearColor(0.1f, 0.2f, 0.3f, 1.0f)))
	{
		return false;
	}

	TInstancedStruct<FPoint> Observations;
	IScholaSensor::Execute_CollectObservations(Sensor, Observations);

	TestTrue(TEXT("Interface dispatch returns BoxPoint"), Observations.GetScriptStruct() == FBoxPoint::StaticStruct());
	TestTrue(TEXT("Interface dispatch populates values"), Observations.Get<FBoxPoint>().Values.Num() > 0);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FCameraSensorIntegration_ObsSpaceVsCollect_Test,
	"Schola.Sensors.CameraSensor.Integration.ObsSpaceVsCollect",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter | EAutomationTestFlags::NonNullRHI)

bool FCameraSensorIntegration_ObsSpaceVsCollect_Test::RunTest(const FString& Parameters)
{
	FScholaCameraSensorTestWorld TestWorld;
	if (!TestWorld.Setup(this))
	{
		return false;
	}

	UCameraSensor* Sensor = NewObject<UCameraSensor>(GetTransientPackage());
	Sensor->CaptureSource = ESceneCaptureSource::SCS_SceneColorSceneDepth;
	Sensor->EnabledChannels = 15;
	Sensor->TextureTarget = CreateInitializedRenderTarget(Sensor, 16, 16);
	FillRenderTargetSolidColor(TestWorld.GetWorld(), Sensor->TextureTarget, FLinearColor(0.5f, 0.5f, 0.5f, 0.75f));

	TInstancedStruct<FSpace> ObservationSpace;
	IScholaSensor::Execute_GetObservationSpace(Sensor, ObservationSpace);
	const FBoxSpace& Space = ObservationSpace.Get<FBoxSpace>();

	TInstancedStruct<FPoint> Observations;
	IScholaSensor::Execute_CollectObservations(Sensor, Observations);
	const FBoxPoint& BoxPoint = Observations.Get<FBoxPoint>();

	TestEqual(TEXT("Shape matches between space and collect"), BoxPoint.Shape, Space.Shape);
	TestEqual(TEXT("Value count matches space dimensions"), BoxPoint.Values.Num(), Space.Dimensions.Num());

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FCameraSensorIntegration_EnabledChannelsRespected_Test,
	"Schola.Sensors.CameraSensor.Integration.EnabledChannelsRespected",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter | EAutomationTestFlags::NonNullRHI)

bool FCameraSensorIntegration_EnabledChannelsRespected_Test::RunTest(const FString& Parameters)
{
	FScholaCameraSensorTestWorld TestWorld;
	if (!TestWorld.Setup(this))
	{
		return false;
	}

	UCameraSensor* Sensor = NewObject<UCameraSensor>(GetTransientPackage());
	Sensor->CaptureSource = ESceneCaptureSource::SCS_SceneColorSceneDepth;
	Sensor->EnabledChannels = static_cast<uint8>(EChannels::R) | static_cast<uint8>(EChannels::B);
	Sensor->TextureTarget = CreateInitializedRenderTarget(Sensor, 4, 4);
	FillRenderTargetSolidColor(TestWorld.GetWorld(), Sensor->TextureTarget, FLinearColor(1.0f, 0.0f, 1.0f, 1.0f));

	FInstancedStruct Observations;
	Sensor->CollectObservations_Implementation(Observations);
	const FBoxPoint& BoxPoint = Observations.Get<FBoxPoint>();

	TestEqual(TEXT("Only R and B enabled"), BoxPoint.Shape[0], 2);
	TestEqual(TEXT("Value count for two channels"), BoxPoint.Values.Num(), 2 * 4 * 4);

	const float ExpectedR = 1.0f;
	const float ExpectedB = 1.0f;
	TestTrue(TEXT("R channel preserved"), FMath::IsNearlyEqual(BoxPoint.Values[0], ExpectedR, 0.02f));
	TestTrue(TEXT("B channel preserved"), FMath::IsNearlyEqual(BoxPoint.Values[4 * 4 + 0], ExpectedB, 0.02f));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FCameraSensorIntegration_InvalidChannelsStripped_Test,
	"Schola.Sensors.CameraSensor.Integration.InvalidChannelsStripped",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter | EAutomationTestFlags::NonNullRHI)

bool FCameraSensorIntegration_InvalidChannelsStripped_Test::RunTest(const FString& Parameters)
{
	UCameraSensor* Sensor = NewObject<UCameraSensor>(GetTransientPackage());
	Sensor->CaptureSource = ESceneCaptureSource::SCS_SceneDepth;
	Sensor->EnabledChannels = 15;
	Sensor->TextureTarget = NewObject<UTextureRenderTarget2D>(Sensor);
	Sensor->TextureTarget->RenderTargetFormat = ETextureRenderTargetFormat::RTF_RGBA8;
	Sensor->TextureTarget->SizeX = 8;
	Sensor->TextureTarget->SizeY = 8;

	TestEqual(TEXT("SceneDepth exposes one channel"), Sensor->GetNumChannels(), 1);

	TArray<FColor> Bitmap;
	Bitmap.Init(FColor(200, 100, 50, 255), 8 * 8);

	FBoxPoint BoxPoint;
	FCameraSensorUtils::ConvertBitmapToBoxPoint(
		Bitmap,
		8,
		8,
		Sensor->EnabledChannels & ~Sensor->GetInvalidChannels(),
		BoxPoint);

	TestEqual(TEXT("Only R collected for depth"), BoxPoint.Shape[0], 1);
	TestEqual(TEXT("R value normalized"), BoxPoint.Values[0], 200.0f / 255.0f);

	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
