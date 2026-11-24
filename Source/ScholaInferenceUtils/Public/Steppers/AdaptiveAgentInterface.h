// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "UObject/Interface.h"
#include "AdaptiveAgentInterface.generated.h"

UINTERFACE(Blueprintable, BlueprintType)
class SCHOLAINFERENCEUTILS_API UAdaptiveAgent : public UInterface
{
    GENERATED_BODY()
};

/**
 * The learning half of an online-learning NPC, implemented next to IAgent.
 * IAgent supplies observations and applies actions; this interface supplies what a learner also
 * needs: the reward earned by the last action and whether the encounter has ended.
 * UAdaptiveStepper calls these on the game thread, once per decision interval.
 */
class SCHOLAINFERENCEUTILS_API IAdaptiveAgent
{
    GENERATED_BODY()

public:
    /** Return the reward accumulated since the previous call and clear it. Must be finite. */
    UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "Schola|Adaptive")
    float ConsumeReward();

    /** Terminated: the task reached a terminal state. Truncated: a time limit or interruption ended it. */
    UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "Schola|Adaptive")
    void GetEpisodeStatus(bool& bTerminated, bool& bTruncated);

    /** Start a new encounter. Called after the ending outcome was submitted; learned state is kept. */
    UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "Schola|Adaptive")
    void ResetEpisode();
};
