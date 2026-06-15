// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Kismet/BlueprintFunctionLibrary.h"
#include "StructUtils/InstancedStruct.h"
#include "Points/Point.h"
#include "PointBlueprintLibrary.generated.h"

/**
 * @class UPointBlueprintLibrary
 * @brief Blueprint oriented helper functions for inspecting Point InstancedStructs.
 * 
 * This library provides general utility functions for inspecting different types
 * of Point instances from within Blueprints.
 * 
 * For type-specific functions (creation, conversion, etc.), see:
 * - UMultiBinaryPointBlueprintLibrary for MultiBinary points
 * - UDiscretePointBlueprintLibrary for Discrete points
 * - UMultiDiscretePointBlueprintLibrary for MultiDiscrete points
 * - UBoxPointBlueprintLibrary for Box (continuous) points
 * - UDictPointBlueprintLibrary for Dictionary points
 * - UTextPointBlueprintLibrary for Text points
 */
UCLASS()
class SCHOLA_API UPointBlueprintLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()
public:

    /**
     * @brief Gets the underlying kind/type of a point.
     * @param[in] InPoint The point to inspect.
     * @return The kind of the point (Binary, Discrete, Box, or Dict).
     */
	UFUNCTION(BlueprintPure, Category = "Schola|Point", meta = (DisplayName = "Get Type (Point)", ReturnDisplayName="Type"))
    static EPointType Point_Type(UPARAM(DisplayName="Point") const FInstancedStruct& InPoint);

    /**
     * @brief Check if a point is of a specific type.
     * @param[in] InPoint The point to inspect.
     * @param[in] InType The type to check against.
     * @return True if the point is of the specified type, false otherwise.
     */
    UFUNCTION(BlueprintPure, Category="Schola|Point", meta=(DisplayName="Is Of Type (Point)", ReturnDisplayName="Is Type"))
    static bool Point_IsOfType(UPARAM(DisplayName="Point") const FInstancedStruct& InPoint, UPARAM(DisplayName="Type") EPointType InType);

	/**
	 * @brief String representation of any instanced point whose script struct derives from FPoint.
	 * @param[in] InPoint Type-erased instanced struct holding a concrete FPoint subtype.
	 * @return Debug string from the point's ToString(), or empty if invalid / wrong type.
	 */
	UFUNCTION(BlueprintPure, Category = "Schola|Point", meta = (DisplayName = "To String (Point)", ReturnDisplayName = "String"))
	static FString PointToString(UPARAM(DisplayName = "Point") const FInstancedStruct& InPoint);

};
