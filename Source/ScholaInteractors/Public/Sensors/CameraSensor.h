// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "TextureResource.h"
#include "Engine/Texture2D.h"  
#include "RenderUtils.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/SceneCapture2D.h"
#include "Components/SceneCaptureComponent2D.h"
#include "SensorInterface.h"
#include "CameraSensor.generated.h"


/**
 * @brief Bitflag enum for RGBA color channels.
 * 
 * Used to selectively enable/disable individual color channels in camera observations.
 */
UENUM(BlueprintType, meta = (Bitflags, UseEnumValuesAsMaskValuesInEditor = "true"))
enum class EChannels : uint8
{
	NONE = 0 UMETA(Hidden), /** No channels enabled */
	R = 1 << 0,              /** Red channel */
	G = 1 << 1,              /** Green channel */
	B = 1 << 2,              /** Blue channel */
	A = 1 << 3,              /** Alpha channel */
};
ENUM_CLASS_FLAGS(EChannels)


/**
 * @brief Camera sensor that collects 2D image observations from the environment.
 * 
 * This sensor uses a SceneCaptureComponent2D and a RenderTarget to capture images from the environment.
 * The captured images are converted to normalized floating-point arrays suitable for neural network input.
 * 
 * @details 
 * To use this sensor:
 * 1. Create a UTextureRenderTarget2D asset in the Unreal Editor
 * 2. Assign it to the TextureTarget property (inherited from USceneCaptureComponent2D)
 * 3. Configure which color channels to include via EnabledChannels
 * 
 * Change the CaptureSource setting to choose the mode of operation:
 * - SCS_FinalColorLDR: Standard RGB rendering
 * - SCS_SceneDepth: Depth-only rendering
 * - SCS_SceneColorSceneDepth: Combined RGB and depth
 * 
 * RenderTarget setting recommendations:
 *   Advanced --> Shared = true
 * 
 * SceneCaptureComponent2D setting recommendations:
 *   bCaptureEveryFrame = true
 *   bRenderInMainRenderer = true
 *   CompositeMode = overwrite
 * 
 * @see https://dev.epicgames.com/documentation/en-us/unreal-engine/1.7---scene-capture-2d
 * @see https://dev.epicgames.com/documentation/en-us/unreal-engine/BlueprintAPI/RenderTarget
 */
UCLASS(Blueprintable, meta = (BlueprintSpawnableComponent))
class SCHOLAINTERACTORS_API UCameraSensor : public USceneCaptureComponent2D, public IScholaSensor
{
	GENERATED_BODY()

public:

	/** Bitmask of enabled color channels. Set bits determine which RGBA channels are included in observations. */
	UPROPERTY(EditAnywhere, meta = (Bitmask, BitmaskEnum = "/Script/ScholaInteractors.EChannels"), Category = "Sensor Properties")
	uint8 EnabledChannels = 15;
	
	/**
	 * @brief Generate a unique ID string for this sensor.
	 * 
	 * Includes capture source, render target format, enabled channels, and resolution.
	 * 
	 * @return FString describing the sensor configuration (e.g., "Camera_SCS_FinalColorLDR_RTF_RGBA8_RGB_W128_H128")
	 */
	FString GenerateId() const;
	
	/**
	 * @brief Determine which channels are invalid for the current capture configuration.
	 * 
	 * Some capture sources (e.g., depth) don't support all RGBA channels.
	 * This method checks the CaptureSource and RenderTargetFormat to determine restrictions.
	 * 
	 * @return uint8 Bitmask of invalid channels for the current configuration
	 */
	UFUNCTION()
	uint8 GetInvalidChannels() const;

	/**
	 * @brief Get the number of enabled and valid color channels.
	 * 
	 * Counts the channels that are both enabled and valid for the current capture configuration.
	 * 
	 * @return int Number of channels that will be included in observations
	 */
	int GetNumChannels() const;

	/**
	 * @brief Collect image observations from the render target.
	 * 
	 * Reads pixels from the render target, extracts enabled channels, and normalizes to [0,1].
	 * The resulting BoxPoint has shape [NumChannels, Height, Width] in row-major (CHW) layout.
	 * 
	 * @param[out] OutObservations A BoxPoint that will be populated with normalized pixel values
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

	/**
	 * @brief Initialize the camera sensor.
	 * 
	 * Creates a default render target if none is assigned. Configures render target settings.
	 */
	void InitSensor_Implementation() override;
};