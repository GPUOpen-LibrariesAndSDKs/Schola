// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Kismet/BlueprintFunctionLibrary.h"
#include "StructUtils/InstancedStruct.h"
#include "Spaces/Space.h"
#include "SpaceBlueprintLibrary.generated.h"

/**
 * @class USpaceBlueprintLibrary
 * @brief Blueprint helpers for inspecting Space InstancedStructs.
 * 
 * This library provides general utility functions for inspecting different types
 * of Space instances from within Blueprints.
 * 
 * For type-specific functions (creation, conversion, etc.), see:
 * - UMultiBinarySpaceBlueprintLibrary for MultiBinary spaces
 * - UDiscreteSpaceBlueprintLibrary for Discrete spaces
 * - UMultiDiscreteSpaceBlueprintLibrary for MultiDiscrete spaces
 * - UBoxSpaceBlueprintLibrary for Box (continuous) spaces
 * - UDictSpaceBlueprintLibrary for Dictionary spaces
 * - UTextSpaceBlueprintLibrary for Text spaces
 */
UCLASS()
class SCHOLA_API USpaceBlueprintLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()
public:

    /**
     * @brief Gets the underlying kind/type of a space.
     * @param[in] InSpace The space to inspect.
     * @return The kind of the space (Binary, Discrete, Box, or Dict).
     */
    UFUNCTION(BlueprintPure, Category="Schola|Space", meta=(DisplayName="Get Type (Space)", ReturnDisplayName="Type"))
    static ESpaceType Space_Type(UPARAM(DisplayName="Space") const FInstancedStruct& InSpace);
	
    /**
     * @brief Check if a space is of a specific type.
     * @param[in] InSpace The space to inspect.
     * @param[in] InType The type to check against.
     * @return True if the space is of the specified type, false otherwise.
     */
    UFUNCTION(BlueprintPure, Category = "Schola|Space", meta = (DisplayName = "Is Of Type (Space)", ReturnDisplayName="Is Type"))
    static bool Space_IsOfType(UPARAM(DisplayName="Space") const FInstancedStruct& InSpace, UPARAM(DisplayName="Type") ESpaceType InType);


    /**
     * @brief Checks if a point is contained in a space.
     * @param[in] InSpace The space to check.
     * @param[in] InPoint The point to check.
     * @return True if the point is contained in the space, false otherwise.
     */
    UFUNCTION(BlueprintPure, Category = "Schola|Space", meta = (DisplayName = "Contains (Space, Point)", ReturnDisplayName="Contains"))
    static bool Space_Contains(UPARAM(DisplayName="Space") const FInstancedStruct& InSpace, UPARAM(DisplayName="Point") const FInstancedStruct& InPoint);

	/**
	 * @brief String representation of any instanced space whose script struct derives from FSpace.
	 * @param[in] InSpace Type-erased instanced struct holding a concrete FSpace subtype.
	 * @return Debug string from the space's ToString(), or empty if invalid / wrong type.
	 */
	UFUNCTION(BlueprintPure, Category = "Schola|Space", meta = (DisplayName = "To String (Space)", ReturnDisplayName = "String"))
	static FString SpaceToString(UPARAM(DisplayName = "Space") const FInstancedStruct& InSpace);
};
