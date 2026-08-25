// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "Test/Sensors/CameraSensorTestHelpers.h"

#include "Sensors/CameraSensor.h"
#include "Engine/DirectionalLight.h"
#include "Engine/StaticMesh.h"
#include "Points/BoxPoint.h"

#if WITH_DEV_AUTOMATION_TESTS

using namespace ScholaCameraSensorTest;

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

	SpawnColoredCube(World, FVector(0.0f, 0.0f, 0.0f), FLinearColor(0.0f, 1.0f, 0.0f, 1.0f));
	for (int32 i = 0; i < 5; ++i)
	{
		TestWorld.Tick(0.016f);
	}

	FInstancedStruct Observations;
	CaptureAndCollect(Sensor, TestWorld, Observations);
	TestTrue(TEXT("CollectObservations succeeds after InitSensor"), Observations.GetScriptStruct() == FBoxPoint::StaticStruct());
	TestTrue(TEXT("Collected values populated"), Observations.Get<FBoxPoint>().Values.Num() > 0);

	// Spatial checks: InitSensor allocates a 128x128 target and FinalColorLDR yields RGB (3 channels).
	const FBoxPoint& BoxPoint = Observations.Get<FBoxPoint>();
	const int32 Width = 128;
	const int32 Height = 128;
	AssertBoxPointShape(*this, BoxPoint, 3, Height, Width, TEXT("AfterInitSensor"));

	// The green cube sits directly ahead of the camera, so it projects to the centre of the frame.
	const FIntRect CenterRegion(Width * 7 / 16, Height * 7 / 16, Width * 9 / 16, Height * 9 / 16);
	const FIntRect CornerRegion(0, 0, Width / 8, Height / 8);
	constexpr int32 RedChannel = 0;
	constexpr int32 GreenChannel = 1;
	constexpr int32 BlueChannel = 2;

	// Centre (cube) should be noticeably greener than the empty background corner.
	AssertBoxPointRegionBrighter(*this, BoxPoint, GreenChannel, CenterRegion, CornerRegion, 0.05f, TEXT("AfterInitSensor green centre vs corner"));

	// Green should dominate red/blue where the green cube is rendered.
	const float CenterR = ComputeBoxPointRegionMean(BoxPoint, RedChannel, CenterRegion);
	const float CenterG = ComputeBoxPointRegionMean(BoxPoint, GreenChannel, CenterRegion);
	const float CenterB = ComputeBoxPointRegionMean(BoxPoint, BlueChannel, CenterRegion);
	TestTrue(FString::Printf(TEXT("Centre green %.3f exceeds red %.3f"), CenterG, CenterR), CenterG > CenterR);
	TestTrue(FString::Printf(TEXT("Centre green %.3f exceeds blue %.3f"), CenterG, CenterB), CenterG > CenterB);

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
	SpawnColoredCube(World, FVector(0.0f, 0.0f, 0.0f), FLinearColor(1.0f, 0.5f, 0.0f, 1.0f));

	for (int32 i = 0; i < 5; ++i)
	{
		TestWorld.Tick(0.016f);
	}

	FInstancedStruct ObservationSpace;
	Sensor->GetObservationSpace_Implementation(ObservationSpace);
	const FBoxSpace& Space = ObservationSpace.Get<FBoxSpace>();

	FInstancedStruct Observations;
	CaptureAndCollect(Sensor, TestWorld, Observations);
	const FBoxPoint& BoxPoint = Observations.Get<FBoxPoint>();

	TestEqual(TEXT("Collected shape matches observation space"), BoxPoint.Shape, Space.Shape);
	TestEqual(TEXT("Collected value count matches space dimensions"), BoxPoint.Values.Num(), Space.Dimensions.Num());

	// Spatial check: the orange cube (strong red) sits centre-frame, so the centre should be
	// clearly redder than the empty background corner.
	const int32 Width = 32;
	const int32 Height = 32;
	const FIntRect CenterRegion(Width * 3 / 8, Height * 3 / 8, Width * 5 / 8, Height * 5 / 8);
	const FIntRect CornerRegion(0, 0, Width / 8, Height / 8);
	AssertBoxPointRegionBrighter(*this, BoxPoint, 0, CenterRegion, CornerRegion, 0.05f, TEXT("ObsSpaceMatchesCollect orange centre vs corner"));

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

	SpawnColoredCube(World, FVector(-100.0f, 0.0f, 0.0f), FLinearColor(0.5f, 0.5f, 0.5f, 1.0f), FVector(150.0f));
	SpawnColoredCube(World, FVector(200.0f, 0.0f, 0.0f), FLinearColor(0.5f, 0.5f, 0.5f, 1.0f), FVector(150.0f));

	for (int32 i = 0; i < 5; ++i)
	{
		TestWorld.Tick(0.016f);
	}

	FInstancedStruct Observations;
	CaptureAndCollect(Sensor, TestWorld, Observations);
	const FBoxPoint& BoxPoint = Observations.Get<FBoxPoint>();

	const int32 Width = 64;
	const int32 Height = 64;
	AssertBoxPointShape(*this, BoxPoint, 1, Height, Width, TEXT("DepthMode"));

	const float CenterDepth = GetBoxPointChannelValue(BoxPoint, 0, Height / 2, Width / 2);
	TestTrue(TEXT("Center depth value should be populated"), CenterDepth > 0.0f && CenterDepth <= 1.0f);

	// Spatial check: the cubes occupy the centre of the frame while the corners see empty background,
	// so the centre depth should be measurably different from the background corners.
	const FIntRect CenterRegion(Width * 3 / 8, Height * 3 / 8, Width * 5 / 8, Height * 5 / 8);
	const FIntRect CornerRegion(0, 0, Width / 8, Height / 8);
	const float CenterDepthMean = ComputeBoxPointRegionMean(BoxPoint, 0, CenterRegion);
	const float CornerDepthMean = ComputeBoxPointRegionMean(BoxPoint, 0, CornerRegion);
	TestTrue(
		FString::Printf(
			TEXT("Centre depth (%.4f) should differ from background corner depth (%.4f)"),
			CenterDepthMean,
			CornerDepthMean),
		!FMath::IsNearlyEqual(CenterDepthMean, CornerDepthMean, 0.02f));

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

	// Offset the green cube to +Y so it should render in the right half of the frame.
	SpawnColoredCube(World, FVector(0.0f, 150.0f, 0.0f), FLinearColor(0.0f, 1.0f, 0.0f, 1.0f));

	for (int32 i = 0; i < 5; ++i)
	{
		TestWorld.Tick(0.016f);
	}

	FInstancedStruct Observations;
	CaptureAndCollect(Sensor, TestWorld, Observations);
	const FBoxPoint& BoxPoint = Observations.Get<FBoxPoint>();

	const int32 Width = 64;
	const int32 Height = 64;
	AssertBoxPointShape(*this, BoxPoint, 3, Height, Width, TEXT("SpatialLocalization"));

	constexpr int32 GreenChannel = 1;
	const FIntRect RightHalf(Width / 2, 0, Width, Height);
	const FIntRect LeftHalf(0, 0, Width / 2, Height);

	// The +Y cube should localise to the right half of the image, leaving the left half dark.
	AssertBoxPointRegionBrighter(
		*this,
		BoxPoint,
		GreenChannel,
		RightHalf,
		LeftHalf,
		0.05f,
		TEXT("SpatialLocalization +Y cube on right"));

	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
