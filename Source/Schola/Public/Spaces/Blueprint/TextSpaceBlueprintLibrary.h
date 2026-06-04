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
     * @param[in] InMinLength The minimum allowed string length (inclusive). Defaults to 1, matching Gymnasium.
     * @param[in] InCharset The set of allowed characters. An empty string applies no charset restriction in C++, but maps to Gymnasium's default (alphanumeric) set when exchanged with Python. Use GetTextCharsetPreset for common sets.
     * @return A new text space instance.
     */
    UFUNCTION(BlueprintPure, Category="Schola|Space|Text", meta=(DisplayName="Make Text Space"))
    static TInstancedStruct<FTextSpace> MakeTextSpace(UPARAM(DisplayName="Max Length") int32 InMaxLength, UPARAM(DisplayName="Min Length") int32 InMinLength = 1, UPARAM(DisplayName="Charset") const FString& InCharset = TEXT(""));

    /**
     * @brief Creates a text space from its length bounds and a character-set preset.
     * @param[in] InMaxLength The maximum allowed string length (inclusive).
     * @param[in] InMinLength The minimum allowed string length (inclusive). Defaults to 1, matching Gymnasium.
     * @param[in] InCharsetPreset The character-set preset to use. ETextCharsetPreset::Any applies no restriction.
     * @return A new text space instance.
     */
    UFUNCTION(BlueprintPure, Category="Schola|Space|Text", meta=(DisplayName="Make Text Space From Preset"))
    static TInstancedStruct<FTextSpace> MakeTextSpaceFromPreset(UPARAM(DisplayName="Max Length") int32 InMaxLength, UPARAM(DisplayName="Min Length") int32 InMinLength = 1, UPARAM(DisplayName="Charset Preset") ETextCharsetPreset InCharsetPreset = ETextCharsetPreset::Any);

    /**
     * @brief Breaks a text space into its length bounds and character set.
     * @param[in] InTextSpace The text space to break apart.
     * @param[out] OutMaxLength Output parameter that receives the maximum length.
     * @param[out] OutMinLength Output parameter that receives the minimum length.
     * @param[out] OutCharset Output parameter that receives the character set.
     */
    UFUNCTION(BlueprintPure, Category="Schola|Space|Text", meta=(DisplayName="Break Text Space"))
    static void BreakTextSpace(UPARAM(DisplayName="Text Space") const TInstancedStruct<FTextSpace>& InTextSpace, UPARAM(DisplayName="Max Length") int32& OutMaxLength, UPARAM(DisplayName="Min Length") int32& OutMinLength, UPARAM(DisplayName="Charset") FString& OutCharset);

    /**
     * @brief Returns the character-set string for a given preset.
     * @param[in] InPreset The preset to resolve.
     * @return The matching character-set string. ETextCharsetPreset::Any returns an empty string (no restriction).
     */
    UFUNCTION(BlueprintPure, Category="Schola|Space|Text", meta=(DisplayName="Get Text Charset Preset"))
    static FString GetTextCharsetPreset(UPARAM(DisplayName="Preset") ETextCharsetPreset InPreset);

};
