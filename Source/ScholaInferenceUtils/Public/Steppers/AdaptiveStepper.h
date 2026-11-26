// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "Steppers/StepperInterface.h"
#include "Agent/AgentInterface.h"
#include "Policies/AdaptiveLearningTypes.h"
#include "AdaptiveStepper.generated.h"

class UAdaptiveNPCPolicy;

/** Counters since Init. Submit results count learner admission, not completed training. */
USTRUCT(BlueprintType)
struct SCHOLAINFERENCEUTILS_API FAdaptiveStepperStats
{
    GENERATED_BODY()

    /** Actions applied to agents. */
    UPROPERTY(BlueprintReadOnly, Category = "Schola|Adaptive")
    int64 Decisions = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Schola|Adaptive")
    int64 Accepted = 0;

    /** Rejected, Dropped, Frozen and NotReady submissions together. */
    UPROPERTY(BlueprintReadOnly, Category = "Schola|Adaptive")
    int64 NotAccepted = 0;

    /** Requests that expired, failed or were cancelled before their action was applied. */
    UPROPERTY(BlueprintReadOnly, Category = "Schola|Adaptive")
    int64 Abandoned = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Schola|Adaptive")
    int64 Episodes = 0;
};

/**
 * Drives adaptive (online-learning) policies, which other steppers cannot: UAdaptiveNPCPolicy::Think
 * deliberately returns false because an adaptive action must be attributed to the outcome it causes.
 *
 * Each agent gets its own policy binding: Init duplicates the template policy once per agent, with the
 * template's outer, and initializes it with that agent's definition. Learning is independent per agent
 * unless the template's backend pools it (for example URLToolsNPCPolicy::SharedGroup).
 *
 * Every Step, for each agent whose policy session is Ready:
 *   1. A pending request is polled. A Ready action is applied with IAgent::Act and opens an interval.
 *   2. Otherwise the open interval is closed: the next observation, IAdaptiveAgent reward and episode
 *      flags are submitted with SubmitExperience. Only then, if the episode ended, the agent and the
 *      policy reset the episode (the policy validates the experience's episode ID, so this order matters).
 *   3. A new decision is requested and polled at once, so synchronous backends act in the same Step.
 * Reward earned while a request is still pending is discarded when the action arrives: no decision
 * owns it. Agents must implement both IAgent and IAdaptiveAgent. Call everything on the game thread.
 * Destroying the stepper does not shut bindings down early; each policy releases its own on destruction.
 */
UCLASS(BlueprintType, EditInlineNew, DefaultToInstanced)
class SCHOLAINFERENCEUTILS_API UAdaptiveStepper : public UObject, public IStepper
{
    GENERATED_BODY()

public:
    /** Seconds a requested action may wait (and stay usable) before the policy expires it. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Adaptive", meta = (ClampMin = "0.001"))
    float DecisionTimeoutSeconds = 1.0f;

    /** InPolicy must be a UAdaptiveNPCPolicy; it is used as a template and is not itself bound. */
    UFUNCTION(BlueprintCallable, Category = "Schola|Stepper")
    bool Init(const TArray<TScriptInterface<IAgent>>& InAgents, const TScriptInterface<IPolicy>& InPolicy) override;

    UFUNCTION(BlueprintCallable, Category = "Schola|Stepper")
    void Step();

    /** Shut down every binding. Init is required before stepping again. */
    UFUNCTION(BlueprintCallable, Category = "Schola|Stepper")
    void Shutdown();

    /** The policy bound to agent Index, for learning controls and diagnostics; null if out of range. */
    UFUNCTION(BlueprintPure, Category = "Schola|Stepper")
    UAdaptiveNPCPolicy* GetPolicy(int32 Index) const;

    UFUNCTION(BlueprintPure, Category = "Schola|Stepper")
    int32 GetNumAgents() const { return Slots.Num(); }

    /** True when every binding has a Ready session. Native learners become Ready asynchronously. */
    UFUNCTION(BlueprintPure, Category = "Schola|Stepper")
    bool AreAllReady() const;

    UFUNCTION(BlueprintPure, Category = "Schola|Stepper")
    FAdaptiveStepperStats GetStats() const { return Stats; }

    /** Error from the latest failure; empty when the last Init succeeded and nothing failed since. */
    UFUNCTION(BlueprintPure, Category = "Schola|Stepper")
    FString GetLastError() const { return LastError; }

private:
    struct FSlot
    {
        TScriptInterface<IAgent> Agent;
        TObjectPtr<UAdaptiveNPCPolicy> Policy;
        int64 PendingTicket = INDEX_NONE;
        TInstancedStruct<FPoint> PendingObservation;
        // The applied decision whose outcome is not yet submitted.
        bool bIntervalOpen = false;
        FAdaptiveDecision Applied;
        TInstancedStruct<FPoint> AppliedObservation;
    };

    void StepSlot(FSlot& Slot);
    bool PollPending(FSlot& Slot);
    void CloseInterval(FSlot& Slot, TInstancedStruct<FPoint>& OutNextObservation);
    void Fail(const FString& Error);

    // Reflected copies keep the agents and bindings alive; Slots refers to the same objects.
    UPROPERTY(Transient)
    TArray<TScriptInterface<IAgent>> Agents;

    UPROPERTY(Transient)
    TArray<TObjectPtr<UAdaptiveNPCPolicy>> Policies;

    TArray<FSlot> Slots;
    FAdaptiveStepperStats Stats;
    FString LastError;
};
