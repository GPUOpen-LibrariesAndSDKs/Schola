// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Misc/AutomationTest.h"
#include "Tests/AutomationCommon.h"
#include "Tests/AutomationEditorCommon.h"

#include "Engine/TextureRenderTarget2D.h"
#include "Points/BoxPoint.h"
#include "Sensors/CameraSensor.h"

class UStaticMesh;
class UStaticMeshComponent;

/**
 * RAII wrapper around FTestWorldWrapper for camera sensor tests: creates a dedicated game world,
 * runs BeginPlay, ticks a few frames, then EndPlay + Destroy on scope exit.
 */
struct FScholaCameraSensorTestWorld
{
	enum class EPhase : uint8
	{
		None,
		Created,
		Playing,
	};

	FTestWorldWrapper Wrapper;
	FAutomationTestBase* Test = nullptr;
	EPhase Phase = EPhase::None;

	bool Setup(FAutomationTestBase* InTest);

	void Tick(float DeltaSeconds = 0.016f);

	UWorld* GetWorld() const { return Wrapper.GetTestWorld(); }

	~FScholaCameraSensorTestWorld();

	FScholaCameraSensorTestWorld() = default;
	FScholaCameraSensorTestWorld(const FScholaCameraSensorTestWorld&) = delete;
	FScholaCameraSensorTestWorld& operator=(const FScholaCameraSensorTestWorld&) = delete;
};

namespace ScholaCameraSensorTest
{
	/** Row-major FColor bitmap with R=(w,h) and G=(h,w) encodings for layout checks. */
	void BuildKnownColorBitmap(int32 Width, int32 Height, TArray<FColor>& OutBitmap);

	UTextureRenderTarget2D* CreateInitializedRenderTarget(
		UObject* Outer,
		int32 Width,
		int32 Height,
		ETextureRenderTargetFormat Format = ETextureRenderTargetFormat::RTF_RGBA8);

	bool FillRenderTargetSolidColor(UWorld* World, UTextureRenderTarget2D* RenderTarget, const FLinearColor& Color);

	bool FillRenderTargetFromBitmap(
		UWorld* World,
		UTextureRenderTarget2D* RenderTarget,
		const TArray<FColor>& Bitmap,
		int32 Width,
		int32 Height);

	/** Flush pending render commands after GPU writes or scene capture. */
	void FlushRendering();

	float GetBoxPointChannelValue(
		const FBoxPoint& BoxPoint,
		int32 ChannelIndex,
		int32 PixelY,
		int32 PixelX,
		int32 Width,
		int32 Height);

	bool AssertBoxPointShape(
		FAutomationTestBase& Test,
		const FBoxPoint& BoxPoint,
		int32 ExpectedChannels,
		int32 ExpectedHeight,
		int32 ExpectedWidth,
		const TCHAR* Context);

	bool AssertBoxPointChannelNear(
		FAutomationTestBase& Test,
		const FBoxPoint& BoxPoint,
		int32 ChannelIndex,
		int32 PixelY,
		int32 PixelX,
		int32 Width,
		int32 Height,
		float ExpectedValue,
		float Tolerance,
		const TCHAR* Context);

	bool AssertBoxPointRegionMeanAbove(
		FAutomationTestBase& Test,
		const FBoxPoint& BoxPoint,
		int32 ChannelIndex,
		int32 StartY,
		int32 EndY,
		int32 StartX,
		int32 EndX,
		int32 Width,
		int32 Height,
		float MinMean,
		const TCHAR* Context);

	bool AssertBoxPointRegionMeanBelow(
		FAutomationTestBase& Test,
		const FBoxPoint& BoxPoint,
		int32 ChannelIndex,
		int32 StartY,
		int32 EndY,
		int32 StartX,
		int32 EndX,
		int32 Width,
		int32 Height,
		float MaxMean,
		const TCHAR* Context);

	/**
	 * @brief Mean of a single channel over the half-open pixel rect [Min, Max).
	 *
	 * Region is expressed in image pixel coordinates (X = column, Y = row).
	 */
	float ComputeBoxPointRegionMean(
		const FBoxPoint& BoxPoint,
		int32 ChannelIndex,
		const FIntRect& Region,
		int32 Width,
		int32 Height);

	/**
	 * @brief Assert that a channel is brighter in one image region than another.
	 *
	 * Useful for spatial-localization checks (e.g. a cube rendered in the centre of the frame
	 * should be brighter than the background corners). Compares mean(BrightRegion) against
	 * mean(DimRegion) and requires the difference to be at least MinDelta.
	 *
	 * @return True if mean(BrightRegion) - mean(DimRegion) >= MinDelta.
	 */
	bool AssertBoxPointRegionBrighter(
		FAutomationTestBase& Test,
		const FBoxPoint& BoxPoint,
		int32 ChannelIndex,
		const FIntRect& BrightRegion,
		const FIntRect& DimRegion,
		int32 Width,
		int32 Height,
		float MinDelta,
		const TCHAR* Context);

	UCameraSensor* SpawnCameraSensor(
		UWorld* World,
		const FVector& Location,
		const FRotator& Rotation,
		int32 RenderTargetWidth = 64,
		int32 RenderTargetHeight = 64,
		ESceneCaptureSource CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR,
		uint8 EnabledChannels = 15,
		bool bCallInitSensor = true,
		bool bCreateRenderTarget = true);

	UStaticMeshComponent* SpawnColoredCube(
		UWorld* World,
		const FVector& Location,
		const FLinearColor& Color,
		const FVector& Scale = FVector(100.0f),
		UStaticMesh* CubeMesh = nullptr);

	void CaptureAndCollect(UCameraSensor* Sensor, FScholaCameraSensorTestWorld& TestWorld, FInstancedStruct& OutObservations);

	uint8 GetEnabledValidChannels(const UCameraSensor* Sensor);
}
