// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "Policies/AdaptiveLearningTypes.h"
#include "Policies/PolicyInterface.h"
#include "AdaptiveNPCPolicy.generated.h"

/**
 * Local setup for the adaptive Unreal policy bridge. Call its methods on the game thread.
 * Currently accepts Box observations and finite, bounded Box actions of configurable size.
 * Init validates and copies definitions only; it does not connect to Python or mark a session Ready.
 */
UCLASS(BlueprintType, Blueprintable, EditInlineNew, DefaultToInstanced)
class SCHOLAINFERENCEUTILS_API UAdaptiveNPCPolicy : public UObject, public IPolicy
{
    GENERATED_BODY()

public:
    /** Reject invalid definitions without replacing an existing valid definition. See GetLastError. */
    UFUNCTION(BlueprintCallable, Category = "Schola|Adaptive")
    bool Init(const FInteractionDefinition& InDefinition) override;

    /** Immediate adaptive inference is unavailable; returns false without modifying Action. */
    UFUNCTION(BlueprintCallable, Category = "Schola|Adaptive")
    bool Think(const TInstancedStruct<FPoint>& Observation, TInstancedStruct<FPoint>& Action) override;

    /** Also preserves the whole output array on failure, unlike the default IPolicy batch fallback. */
    UFUNCTION(BlueprintCallable, Category = "Schola|Adaptive")
    bool BatchedThink(const TArray<TInstancedStruct<FPoint>>& Observations,
                      TArray<TInstancedStruct<FPoint>>& Actions) override;

    /** No requests are dispatched by this foundation. This is not a readiness check. */
    UFUNCTION(BlueprintPure, Category = "Schola|Adaptive")
    bool IsInferenceBusy() const override { return false; }

    UFUNCTION(BlueprintPure, Category = "Schola|Adaptive")
    bool IsInitialized() const { return bInitialized; }

    UFUNCTION(BlueprintPure, Category = "Schola|Adaptive")
    EAdaptiveSessionState GetSessionState() const { return SessionState; }

    /** Error from the latest failed operation, cleared by successful initialization. */
    UFUNCTION(BlueprintPure, Category = "Schola|Adaptive")
    FString GetLastError() const { return LastError; }

    /** Copies the accepted definition. Before initialization returns false and preserves the output. */
    UFUNCTION(BlueprintPure, Category = "Schola|Adaptive")
    bool GetInteractionDefinition(FInteractionDefinition& OutDefinition) const;

private:
    UPROPERTY()
    FInteractionDefinition InteractionDefinition;

    UPROPERTY()
    EAdaptiveSessionState SessionState = EAdaptiveSessionState::Uninitialized;

    UPROPERTY()
    FString LastError;

    UPROPERTY()
    bool bInitialized = false;
};
