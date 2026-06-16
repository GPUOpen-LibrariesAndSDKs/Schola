// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Kismet/BlueprintFunctionLibrary.h"
#include "StructUtils/InstancedStruct.h"
#include "Spaces/BoxSpace.h"
#include "BoxSpaceBlueprintLibrary.generated.h"

/**
 * @class UBoxSpaceBlueprintLibrary
 * @brief Blueprint oriented helper functions for creating & inspecting Box Space InstancedStructs.
 * 
 * This library provides utility functions for creating and manipulating Box (continuous) Space instances
 * from within Blueprints. These return TInstancedStruct<FBoxSpace>.
 */
UCLASS()
class SCHOLA_API UBoxSpaceBlueprintLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()
public:

    /**
     * @brief Converts arrays to a box (continuous) space with specified bounds and shape.
     * @param[in] InLow Array of lower bounds for each dimension.
     * @param[in] InHigh Array of upper bounds for each dimension.
     * @param[in] InShape The dimensional shape of the space.
     * @return A new box space instance.
     */
    UFUNCTION(BlueprintPure, Category="Schola|Space|Box", meta=(DisplayName="From Arrays (Box Space)"))
    static TInstancedStruct<FBoxSpace> ArraysToBoxSpace(UPARAM(DisplayName="Low") const TArray<float>& InLow, UPARAM(DisplayName="High") const TArray<float>& InHigh, UPARAM(DisplayName="Shape") const TArray<int32>& InShape);

    /**
     * @brief Creates a box space for FVector with specified bounds for each component.
     * @param[in] InLow The lower bounds for X, Y, Z components.
     * @param[in] InHigh The upper bounds for X, Y, Z components.
     * @return A box space with 3 dimensions (X, Y, Z) with the specified bounds.
     */
    UFUNCTION(BlueprintPure, Category="Schola|Space|Box", meta=(DisplayName="From Vector (Box Space)"))
    static TInstancedStruct<FBoxSpace> VectorToBoxSpace(UPARAM(DisplayName="Low") const FVector& InLow, UPARAM(DisplayName="High") const FVector& InHigh);

    /**
     * @brief Creates a box space for FRotator with specified bounds for each component.
     * @return A box space with 3 dimensions (Pitch, Yaw, Roll) with bounds [-180,180].
     */
    UFUNCTION(BlueprintPure, Category="Schola|Space|Box", meta=(DisplayName="Rotator Space"))
    static TInstancedStruct<FBoxSpace> RotatorSpace();

    /**
     * @brief Creates a box space for FTransform with specified bounds for Location, Rotation, and Scale.
     * @param[in] InLocationLow The lower bounds for Location (X, Y, Z).
     * @param[in] InLocationHigh The upper bounds for Location (X, Y, Z).
     * @param[in] InScaleLow The lower bounds for Scale (X, Y, Z).
     * @param[in] InScaleHigh The upper bounds for Scale (X, Y, Z).
     * @return A box space with 9 dimensions (Location XYZ, Rotation PYR, Scale XYZ) with the specified bounds.
     */
    UFUNCTION(BlueprintPure, Category="Schola|Space|Box", meta=(DisplayName="From Transform (Box Space)"))
    static TInstancedStruct<FBoxSpace> TransformToBoxSpace(
        UPARAM(DisplayName="Location Low") const FVector& InLocationLow, 
        UPARAM(DisplayName="Location High") const FVector& InLocationHigh,
        UPARAM(DisplayName="Scale Low") const FVector& InScaleLow,
        UPARAM(DisplayName="Scale High") const FVector& InScaleHigh);

};
