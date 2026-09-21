// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Components/SceneComponent.h"
#include "SensorInterface.h"
#include "FakeCameraSensor.generated.h"

/**
 * @brief Test/dummy camera sensor that emits configurable image observations without scene capture.
 *
 * Observation layout matches UCameraSensor: a BoxPoint with shape [NumChannels, Height, Width]
 * in row-major (CHW) layout and values in [0, 1]. If CustomImage is empty, CollectObservations
 * returns an all-zero image of that size. If CustomImage is populated, those values are returned
 * (padded with zeros or truncated to match NumChannels * Height * Width).
 */
UCLASS(Blueprintable, meta = (BlueprintSpawnableComponent))
class SCHOLAINTERACTORS_API UFakeCameraSensor : public USceneComponent, public IScholaSensor
{
	GENERATED_BODY()

public:
	/** Number of image channels in the observation (e.g. 1 for grayscale, 3 for RGB, 4 for RGBA). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Sensor Properties", meta = (ClampMin = "1", ClampMax = "4"))
	int32 NumChannels = 3;

	/** Image width in pixels. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Sensor Properties", meta = (ClampMin = "1"))
	int32 Width = 128;

	/** Image height in pixels. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Sensor Properties", meta = (ClampMin = "1"))
	int32 Height = 128;

	/**
	 * Optional custom image buffer returned by CollectObservations.
	 *
	 * Expected length is NumChannels * Height * Width in CHW order.
	 * Leave empty to emit an all-zero image of that size.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Sensor Properties")
	TArray<float> CustomImage;

	/**
	 * @brief Flattened observation size (NumChannels * Height * Width), using clamped positive dimensions.
	 */
	int32 GetObservationSize() const;

	/**
	 * @brief Generate a unique ID string for this sensor.
	 *
	 * @return FString describing the sensor configuration (e.g., "FakeCamera_C3_W128_H128")
	 */
	FString GenerateId() const;

	/**
	 * @brief Collect image observations from CustomImage, or zeros if it has not been supplied.
	 *
	 * The resulting BoxPoint has shape [NumChannels, Height, Width] in row-major (CHW) layout.
	 *
	 * @param[out] OutObservations A BoxPoint that will be populated with image values
	 */
	void CollectObservations_Implementation(FInstancedStruct& OutObservations) override;

	/**
	 * @brief Get the observation space for this sensor.
	 *
	 * Returns a BoxSpace describing the image dimensions and channels.
	 * Each dimension is bounded [0.0, 1.0] for normalized pixel values.
	 * Shape is [NumChannels, Height, Width] in row-major (CHW) layout.
	 *
	 * @param[out] OutObservationSpace The observation space definition to be populated
	 */
	void GetObservationSpace_Implementation(FInstancedStruct& OutObservationSpace) const override;
};
