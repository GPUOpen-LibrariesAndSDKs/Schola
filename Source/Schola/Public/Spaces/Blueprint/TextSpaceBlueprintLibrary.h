// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "Kismet/BlueprintFunctionLibrary.h"
#include "StructUtils/InstancedStruct.h"
#include "Spaces/Space.h"
#include "Spaces/TextSpace.h"
#include "TextSpaceBlueprintLibrary.generated.h"

/**
 * @class UTextSpaceBlueprintLibrary
 * @brief Blueprint oriented helper functions for creating & inspecting Text Space InstancedStructs.
 * 
 * This library provides utility functions for creating and manipulating Text Space instances
 * from within Blueprints. These return TInstancedStruct<FTextSpace>.
 */
UCLASS()
class SCHOLA_API UTextSpaceBlueprintLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()
public:

    /**
     * @brief Creates a text space from its length bounds and optional character set.
     * @param[in] InMaxLength The maximum allowed string length (inclusive).
     * @param[in] bInHasMinLength Whether a minimum length constraint is enforced.
     * @param[in] InMinLength The minimum allowed string length (inclusive). Only used when bInHasMinLength is true.
     * @param[in] InCharset The set of allowed characters. An empty string applies no charset restriction in C++, but maps to Gymnasium's default (alphanumeric) set when exchanged with Python.
     * @return A new text space instance.
     */
    UFUNCTION(BlueprintPure, Category="Schola|Space|Text", meta=(AutoCreateRefTerm="InCharset", DisplayName="Make Text Space"))
    static TInstancedStruct<FTextSpace> MakeTextSpace(UPARAM(DisplayName="Max Length") int32 InMaxLength, UPARAM(DisplayName="Has Min Length") bool bInHasMinLength, UPARAM(DisplayName="Min Length") int32 InMinLength, UPARAM(DisplayName="Charset") const FString& InCharset);

    /**
     * @brief Breaks a text space into its length bounds and character set.
     * @param[in] InTextSpace The text space to break apart.
     * @param[out] OutMaxLength Output parameter that receives the maximum length.
     * @param[out] bOutHasMinLength Output parameter that receives whether a minimum length is enforced.
     * @param[out] OutMinLength Output parameter that receives the minimum length.
     * @param[out] OutCharset Output parameter that receives the character set.
     */
    UFUNCTION(BlueprintPure, Category="Schola|Space|Text", meta=(DisplayName="Break Text Space"))
    static void BreakTextSpace(UPARAM(DisplayName="Text Space") const TInstancedStruct<FTextSpace>& InTextSpace, UPARAM(DisplayName="Max Length") int32& OutMaxLength, UPARAM(DisplayName="Has Min Length") bool& bOutHasMinLength, UPARAM(DisplayName="Min Length") int32& OutMinLength, UPARAM(DisplayName="Charset") FString& OutCharset);

};
