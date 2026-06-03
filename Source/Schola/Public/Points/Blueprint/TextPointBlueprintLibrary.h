// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Kismet/BlueprintFunctionLibrary.h"
#include "StructUtils/InstancedStruct.h"
#include "Points/Point.h"
#include "Points/TextPoint.h"
#include "TextPointBlueprintLibrary.generated.h"

/**
 * @class UTextPointBlueprintLibrary
 * @brief Blueprint oriented helper functions for creating & inspecting Text Point InstancedStructs.
 * 
 * This library provides utility functions for creating and manipulating Text Point instances
 * from within Blueprints. These return TInstancedStruct<FTextPoint>.
 */
UCLASS()
class SCHOLA_API UTextPointBlueprintLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()
public:

    /**
     * @brief Converts a string value to a text point.
     * @param[in] InValue The string value to wrap in a text point.
     * @return A new text point instance.
     */
    UFUNCTION(BlueprintPure, Category="Schola|Point|Text", meta=(DisplayName="From String (Text Point)"))
    static TInstancedStruct<FTextPoint> StringToTextPoint(UPARAM(DisplayName="Value") const FString& InValue);

    /**
     * @brief Converts a text point to its string value.
     * @param[in] InTextPoint The text point to convert.
     * @return The string value stored in the text point.
     */
    UFUNCTION(BlueprintPure, Category="Schola|Point|Text", meta=(BlueprintAutocast, DisplayName="To String (Text Point)", CompactNodeTitle="->"))
    static FString TextPointToString(UPARAM(DisplayName="Text Point") const TInstancedStruct<FTextPoint>& InTextPoint);

};
