// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "Test/Sensors/CameraSensorTestHelpers.h"

#include "Sensors/CameraSensor.h"
#include "Engine/DirectionalLight.h"
#include "Engine/StaticMesh.h"
#include "Points/BoxPoint.h"

#if WITH_DEV_AUTOMATION_TESTS

using namespace ScholaCameraSensorTest;

/**
 * Scene-validation strategies used in this file:
 *
 * - Baseline + single cube: capture an empty scene, spawn one coloured cube, capture again,
 *   and assert the expected channel rose vs baseline in a fixed region. Used by AfterInitSensor
 *   and ObsSpaceMatchesCollect.
 *
 * - Sequential two-colour: baseline capture, then spawn cube A and assert, destroy A, spawn
 *   cube B at a different location/colour and assert. Proves capture tracks scene changes and
 *   that colour localises to the expected image region. Used only by SpatialLocalization.
 *
 * - Side-by-side cubes: baseline capture, then spawn multiple cubes in one pass before the
 *   second capture. Used by DepthMode (two grey cubes along +X for centre depth geometry).
 *
 * ResolutionChange does not perform spatial/channel validation; it only checks value counts.
 */

static void SpawnDirectionalLight(UWorld* World)
{
	if (!World)
	{
		return;
	}

	FActorSpawnParameters SpawnParams;
	SpawnParams.ObjectFlags = RF_Transient;
	World->SpawnActor<ADirectionalLight>(FVector(0.0f, 0.0f, 500.0f), FRotator(-45.0f, 45.0f, 0.0f), SpawnParams);
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FCameraSensorCapture_AfterInitSensor_Test,
	"Schola.Sensors.CameraSensor.Capture.AfterInitSensor",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter | EAutomationTestFlags::NonNullRHI)
bool FCameraSensorCapture_AfterInitSensor_Test::RunTest(const FString& Parameters)
{
	FScholaCameraSensorTestWorld TestWorld;
	if (!TestWorld.Setup(this))
	{
		return false;
	}

	UWorld* World = TestWorld.GetWorld();
	SpawnDirectionalLight(World);

	UCameraSensor* Sensor = SpawnCameraSensor(
		World,
		FVector(-400.0f, 0.0f, 0.0f),
		FRotator::ZeroRotator,
		64,
		64,
		ESceneCaptureSource::SCS_FinalColorLDR,
		15,
		false,
		false);
	TestNotNull(TEXT("Camera sensor spawned"), Sensor);
	if (!Sensor)
	{
		return false;
	}

	TestTrue(TEXT("No render target before InitSensor"), Sensor->TextureTarget == nullptr);
	Sensor->InitSensor_Implementation();

	TestNotNull(TEXT("InitSensor creates render target"), Sensor->TextureTarget.Get());
	TestEqual(TEXT("InitSensor default width"), static_cast<int32>(Sensor->TextureTarget->GetSurfaceWidth()), 128);
	TestEqual(TEXT("InitSensor default height"), static_cast<int32>(Sensor->TextureTarget->GetSurfaceHeight()), 128);
	TestTrue(TEXT("InitSensor sets GPU shared flag"), Sensor->TextureTarget->bGPUSharedFlag);

	// Baseline + single cube: empty scene first, then one green cube at the frame centre.
	const int32 Width = 128;
	const int32 Height = 128;
	const FIntRect CenterRegion(Width * 7 / 16, Height * 7 / 16, Width * 9 / 16, Height * 9 / 16);
	constexpr int32 GreenChannel = 1;

	FInstancedStruct BaselineObservations;
	CaptureAndCollect(Sensor, TestWorld, BaselineObservations);
	TestTrue(TEXT("Baseline collect succeeds after InitSensor"), BaselineObservations.GetScriptStruct() == FBoxPoint::StaticStruct());
	const FBoxPoint& BaselineBoxPoint = BaselineObservations.Get<FBoxPoint>();
	AssertBoxPointShape(*this, BaselineBoxPoint, 3, Height, Width, TEXT("AfterInitSensor baseline"));

	UStaticMeshComponent* GreenCube = SpawnColoredCube(
		World,
		FVector(0.0f, 0.0f, 0.0f),
		FLinearColor(0.0f, 1.0f, 0.0f, 1.0f));
	TestNotNull(TEXT("Green cube spawned"), GreenCube);
	SettleScene(TestWorld);

	FInstancedStruct Observations;
	CaptureAndCollect(Sensor, TestWorld, Observations);
	TestTrue(TEXT("CollectObservations succeeds after InitSensor"), Observations.GetScriptStruct() == FBoxPoint::StaticStruct());
	TestTrue(TEXT("Collected values populated"), Observations.Get<FBoxPoint>().Values.Num() > 0);

	const FBoxPoint& BoxPoint = Observations.Get<FBoxPoint>();
	AssertBoxPointShape(*this, BoxPoint, 3, Height, Width, TEXT("AfterInitSensor"));

	AssertBoxPointRegionChannelDeltaFromBaseline(
		*this,
		BoxPoint,
		BaselineBoxPoint,
		GreenChannel,
		CenterRegion,
		CaptureSpatialMinChannelDelta,
		TEXT("AfterInitSensor green centre vs baseline"));
	AssertBoxPointRegionDominantChannel(
		*this,
		BoxPoint,
		GreenChannel,
		CenterRegion,
		TEXT("AfterInitSensor green dominates centre"));

	DestroyColoredCube(GreenCube, TestWorld);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FCameraSensorCapture_ObsSpaceMatchesCollect_Test,
	"Schola.Sensors.CameraSensor.Capture.ObsSpaceMatchesCollect",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter | EAutomationTestFlags::NonNullRHI)
bool FCameraSensorCapture_ObsSpaceMatchesCollect_Test::RunTest(const FString& Parameters)
{
	FScholaCameraSensorTestWorld TestWorld;
	if (!TestWorld.Setup(this))
	{
		return false;
	}

	UWorld* World = TestWorld.GetWorld();
	SpawnDirectionalLight(World);

	UCameraSensor* Sensor = SpawnCameraSensor(World, FVector(-400.0f, 0.0f, 0.0f), FRotator::ZeroRotator, 32, 32);

	// Baseline + single cube: empty scene first, then one orange cube at the frame centre.
	const int32 Width = 32;
	const int32 Height = 32;
	const FIntRect CenterRegion(Width * 3 / 8, Height * 3 / 8, Width * 5 / 8, Height * 5 / 8);
	constexpr int32 RedChannel = 0;

	FInstancedStruct BaselineObservations;
	CaptureAndCollect(Sensor, TestWorld, BaselineObservations);
	const FBoxPoint& BaselineBoxPoint = BaselineObservations.Get<FBoxPoint>();

	UStaticMeshComponent* OrangeCube = SpawnColoredCube(
		World,
		FVector(0.0f, 0.0f, 0.0f),
		FLinearColor(1.0f, 0.5f, 0.0f, 1.0f));
	TestNotNull(TEXT("Orange cube spawned"), OrangeCube);
	SettleScene(TestWorld);

	FInstancedStruct ObservationSpace;
	Sensor->GetObservationSpace_Implementation(ObservationSpace);
	const FBoxSpace& Space = ObservationSpace.Get<FBoxSpace>();

	FInstancedStruct Observations;
	CaptureAndCollect(Sensor, TestWorld, Observations);
	const FBoxPoint& BoxPoint = Observations.Get<FBoxPoint>();

	TestEqual(TEXT("Collected shape matches observation space"), BoxPoint.Shape, Space.Shape);
	TestEqual(TEXT("Collected value count matches space dimensions"), BoxPoint.Values.Num(), Space.Dimensions.Num());

	AssertBoxPointRegionChannelDeltaFromBaseline(
		*this,
		BoxPoint,
		BaselineBoxPoint,
		RedChannel,
		CenterRegion,
		CaptureSpatialMinChannelDelta,
		TEXT("ObsSpaceMatchesCollect red centre vs baseline"));
	AssertBoxPointRegionDominantChannel(
		*this,
		BoxPoint,
		RedChannel,
		CenterRegion,
		TEXT("ObsSpaceMatchesCollect red dominates centre"));

	DestroyColoredCube(OrangeCube, TestWorld);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FCameraSensorCapture_DepthMode_Test,
	"Schola.Sensors.CameraSensor.Capture.DepthMode",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter | EAutomationTestFlags::NonNullRHI)
bool FCameraSensorCapture_DepthMode_Test::RunTest(const FString& Parameters)
{
	FScholaCameraSensorTestWorld TestWorld;
	if (!TestWorld.Setup(this))
	{
		return false;
	}

	UWorld* World = TestWorld.GetWorld();
	SpawnDirectionalLight(World);

	UCameraSensor* Sensor = SpawnCameraSensor(
		World,
		FVector(-600.0f, 0.0f, 0.0f),
		FRotator::ZeroRotator,
		64,
		64,
		ESceneCaptureSource::SCS_SceneDepth,
		15);
	TestNotNull(TEXT("Camera sensor spawned"), Sensor);
	if (!Sensor)
	{
		return false;
	}

	// Side-by-side cubes: baseline with no geometry, then two cubes spawned together along +X.
	const int32 Width = 64;
	const int32 Height = 64;
	const FIntRect CenterRegion(Width * 3 / 8, Height * 3 / 8, Width * 5 / 8, Height * 5 / 8);

	FInstancedStruct BaselineObservations;
	CaptureAndCollect(Sensor, TestWorld, BaselineObservations);
	const FBoxPoint& BaselineBoxPoint = BaselineObservations.Get<FBoxPoint>();

	SpawnColoredCube(World, FVector(-50.0f, 0.0f, 0.0f), FLinearColor(0.5f, 0.5f, 0.5f, 1.0f));
	SpawnColoredCube(World, FVector(50.0f, 0.0f, 0.0f), FLinearColor(0.5f, 0.5f, 0.5f, 1.0f));
	SettleScene(TestWorld);

	FInstancedStruct Observations;
	CaptureAndCollect(Sensor, TestWorld, Observations);
	const FBoxPoint& BoxPoint = Observations.Get<FBoxPoint>();

	AssertBoxPointShape(*this, BoxPoint, 1, Height, Width, TEXT("DepthMode"));

	const float CenterDepth = GetBoxPointChannelValue(BoxPoint, 0, Height / 2, Width / 2);
	TestTrue(TEXT("Center depth value should be populated"), CenterDepth > 0.0f && CenterDepth <= 1.0f);

	const float CenterDepthMean = ComputeBoxPointRegionMean(BoxPoint, 0, CenterRegion);
	const float BaselineCenterDepthMean = ComputeBoxPointRegionMean(BaselineBoxPoint, 0, CenterRegion);
	TestTrue(
		FString::Printf(
			TEXT("Centre depth (%.4f) should differ from baseline centre depth (%.4f)"),
			CenterDepthMean,
			BaselineCenterDepthMean),
		!FMath::IsNearlyEqual(CenterDepthMean, BaselineCenterDepthMean, 0.02f));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FCameraSensorCapture_ResolutionChange_Test,
	"Schola.Sensors.CameraSensor.Capture.ResolutionChange",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter | EAutomationTestFlags::NonNullRHI)
bool FCameraSensorCapture_ResolutionChange_Test::RunTest(const FString& Parameters)
{
	FScholaCameraSensorTestWorld TestWorld;
	if (!TestWorld.Setup(this))
	{
		return false;
	}

	UWorld* World = TestWorld.GetWorld();
	SpawnDirectionalLight(World);
	// No baseline or spatial validation; a cube is present only so captures are non-empty.
	SpawnColoredCube(World, FVector(0.0f, 0.0f, 0.0f), FLinearColor(1.0f, 0.0f, 0.0f, 1.0f));

	auto RunCaptureAtResolution = [&](int32 Resolution) -> int32
	{
		UCameraSensor* Sensor = SpawnCameraSensor(
			World,
			FVector(-400.0f, 0.0f, 0.0f),
			FRotator::ZeroRotator,
			Resolution,
			Resolution);
		for (int32 i = 0; i < 3; ++i)
		{
			TestWorld.Tick(0.016f);
		}

		FInstancedStruct Observations;
		CaptureAndCollect(Sensor, TestWorld, Observations);
		return Observations.Get<FBoxPoint>().Values.Num();
	};

	const int32 Values32 = RunCaptureAtResolution(32);
	const int32 Values64 = RunCaptureAtResolution(64);

	TestEqual(TEXT("32x32 value count"), Values32, 3 * 32 * 32);
	TestEqual(TEXT("64x64 value count"), Values64, 3 * 64 * 64);
	TestTrue(TEXT("Higher resolution produces more values"), Values64 > Values32);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FCameraSensorCapture_SpatialLocalization_Test,
	"Schola.Sensors.CameraSensor.Capture.SpatialLocalization",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter | EAutomationTestFlags::NonNullRHI)
bool FCameraSensorCapture_SpatialLocalization_Test::RunTest(const FString& Parameters)
{
	FScholaCameraSensorTestWorld TestWorld;
	if (!TestWorld.Setup(this))
	{
		return false;
	}

	UWorld* World = TestWorld.GetWorld();
	SpawnDirectionalLight(World);

	// Camera faces +X; its right vector is +Y, which maps to the right (higher column) side of the image.
	UCameraSensor* Sensor = SpawnCameraSensor(World, FVector(-400.0f, 0.0f, 0.0f), FRotator::ZeroRotator, 64, 64);
	TestNotNull(TEXT("Camera sensor spawned"), Sensor);
	if (!Sensor)
	{
		return false;
	}

	// Sequential two-colour: baseline, red cube at centre (assert + destroy), green cube at +Y (assert).
	const int32 Width = 64;
	const int32 Height = 64;
	const FIntRect CenterRegion(Width * 3 / 8, Height * 3 / 8, Width * 5 / 8, Height * 5 / 8);
	const FIntRect RightHalf(Width / 2, 0, Width, Height);
	const FIntRect LeftHalf(0, 0, Width / 2, Height);
	constexpr int32 RedChannel = 0;
	constexpr int32 GreenChannel = 1;

	FInstancedStruct BaselineObservations;
	CaptureAndCollect(Sensor, TestWorld, BaselineObservations);
	const FBoxPoint& BaselineBoxPoint = BaselineObservations.Get<FBoxPoint>();
	AssertBoxPointShape(*this, BaselineBoxPoint, 3, Height, Width, TEXT("SpatialLocalization baseline"));

	UStaticMeshComponent* RedCube = SpawnColoredCube(
		World,
		FVector(0.0f, 0.0f, 0.0f),
		FLinearColor(1.0f, 0.0f, 0.0f, 1.0f));
	TestNotNull(TEXT("Red cube spawned"), RedCube);
	SettleScene(TestWorld);

	FInstancedStruct RedObservations;
	CaptureAndCollect(Sensor, TestWorld, RedObservations);
	const FBoxPoint& RedBoxPoint = RedObservations.Get<FBoxPoint>();

	AssertBoxPointRegionChannelDeltaFromBaseline(
		*this,
		RedBoxPoint,
		BaselineBoxPoint,
		RedChannel,
		CenterRegion,
		CaptureSpatialMinChannelDelta,
		TEXT("SpatialLocalization red centre vs baseline"));
	AssertBoxPointRegionDominantChannel(
		*this,
		RedBoxPoint,
		RedChannel,
		CenterRegion,
		TEXT("SpatialLocalization red dominates centre"));

	DestroyColoredCube(RedCube, TestWorld);

	UStaticMeshComponent* GreenCube = SpawnColoredCube(
		World,
		FVector(0.0f, 150.0f, 0.0f),
		FLinearColor(0.0f, 1.0f, 0.0f, 1.0f));
	TestNotNull(TEXT("Green cube spawned"), GreenCube);
	SettleScene(TestWorld);

	FInstancedStruct GreenObservations;
	CaptureAndCollect(Sensor, TestWorld, GreenObservations);
	const FBoxPoint& GreenBoxPoint = GreenObservations.Get<FBoxPoint>();

	AssertBoxPointRegionChannelDeltaFromBaseline(
		*this,
		GreenBoxPoint,
		BaselineBoxPoint,
		GreenChannel,
		RightHalf,
		CaptureSpatialMinChannelDelta,
		TEXT("SpatialLocalization green right half vs baseline"));
	AssertBoxPointRegionDominantChannel(
		*this,
		GreenBoxPoint,
		GreenChannel,
		RightHalf,
		TEXT("SpatialLocalization green dominates right half"));

	const float RightGreenDelta =
		ComputeBoxPointRegionMean(GreenBoxPoint, GreenChannel, RightHalf)
		- ComputeBoxPointRegionMean(BaselineBoxPoint, GreenChannel, RightHalf);
	const float LeftGreenDelta =
		ComputeBoxPointRegionMean(GreenBoxPoint, GreenChannel, LeftHalf)
		- ComputeBoxPointRegionMean(BaselineBoxPoint, GreenChannel, LeftHalf);
	TestTrue(
		FString::Printf(
			TEXT("SpatialLocalization right green delta %.3f should exceed left green delta %.3f by >= %.3f"),
			RightGreenDelta,
			LeftGreenDelta,
			CaptureSpatialMinChannelDelta),
		(RightGreenDelta - LeftGreenDelta) >= CaptureSpatialMinChannelDelta);

	DestroyColoredCube(GreenCube, TestWorld);

	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
