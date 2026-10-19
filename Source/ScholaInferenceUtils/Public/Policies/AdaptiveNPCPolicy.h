// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "Policies/AdaptiveLearningTypes.h"
#include "Policies/PolicyInterface.h"
#include "AdaptiveNPCPolicy.generated.h"

/**
 * Nonblocking adaptive decision lifecycle. Call all methods on the game thread.
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

    /** Outstanding, unexpired inference only. A false value does not imply session readiness. */
    UFUNCTION(BlueprintPure, Category = "Schola|Adaptive")
    bool IsInferenceBusy() const override;

    /** Issue a ticket without waiting. Rejection preserves OutDecisionId. One pending ticket at a time. */
    UFUNCTION(BlueprintCallable, Category = "Schola|Adaptive")
    bool RequestDecision(const TInstancedStruct<FPoint>& Observation, int64& OutDecisionId, float TimeoutSeconds = 1.0f);

    /** Poll without waiting. Only Ready modifies OutDecision. Reads do not consume the result. */
    UFUNCTION(BlueprintCallable, Category = "Schola|Adaptive")
    EAdaptiveRequestStatus TryGetDecision(int64 DecisionId, FAdaptiveDecision& OutDecision);

    /** Invalidate a pending or ready action; cancellation does not promise remote computation stops. */
    UFUNCTION(BlueprintCallable, Category = "Schola|Adaptive")
    bool CancelDecision(int64 DecisionId);

    /** Advance the encounter, cancelling its actions while preserving the ready learning session. */
    UFUNCTION(BlueprintCallable, Category = "Schola|Adaptive")
    bool ResetEpisode();

    /** Invalidate the session and stop accepting work. Init is required before starting another session. */
    UFUNCTION(BlueprintCallable, Category = "Schola|Adaptive")
    void Shutdown();

    /** Diagnostic for the retained ticket; empty for an unknown ticket. */
    UFUNCTION(BlueprintPure, Category = "Schola|Adaptive")
    FString GetDecisionError(int64 DecisionId) const;

    UFUNCTION(BlueprintPure, Category = "Schola|Adaptive")
    bool IsInitialized() const { return bInitialized; }

    UFUNCTION(BlueprintPure, Category = "Schola|Adaptive")
    EAdaptiveSessionState GetSessionState() const { return SessionState; }

    /** Error from the latest failed operation; successful mutating operations clear it. */
    UFUNCTION(BlueprintPure, Category = "Schola|Adaptive")
    FString GetLastError() const { return LastError; }

    /** Copies the accepted definition. Before initialization returns false and preserves the output. */
    UFUNCTION(BlueprintPure, Category = "Schola|Adaptive")
    bool GetInteractionDefinition(FInteractionDefinition& OutDefinition) const;

protected:
    /** Native integration hooks, not Blueprint readiness switches. No network implementation yet.
     * The future client calls StartSession only after a successful handshake. Completions must be
     * marshalled to the game thread, capturing a weak policy pointer and the original request identity.
     */
    bool StartSession(const FGuid& InSessionId, int64 InAgentId, int64 InEpisodeId);
    bool CompleteDecision(const FAdaptiveDecision& InDecision);
    bool FailDecision(const FAdaptiveDecisionRequest& InRequest, const FString& Error);

    /** Must return promptly and copy input before retaining it. False means dispatch was rejected.
     * May complete synchronously. The base implementation rejects dispatch; tests override this hook.
     */
    virtual bool DispatchDecision(const FAdaptiveDecisionRequest& InRequest) { return false; }

    /** Monotonic local elapsed time. Overridden only by deterministic tests. */
    virtual double GetTimeSeconds() const;

    void BeginDestroy() override;
    void PostLoad() override;
    void PostDuplicate(bool bDuplicateForPIE) override;

private:
    void ResetRuntimeState();
    void ExpireDecision();
    void InvalidateDecision(const FString& Reason);
    bool MatchesPendingRequest(const FGuid& InSessionId, int64 InAgentId, int64 InEpisodeId, int64 InDecisionId) const;

    UPROPERTY()
    FInteractionDefinition InteractionDefinition;

    UPROPERTY(Transient)
    EAdaptiveSessionState SessionState = EAdaptiveSessionState::Uninitialized;

    UPROPERTY()
    FString LastError;

    UPROPERTY()
    bool bInitialized = false;

    UPROPERTY(Transient, DuplicateTransient)
    FAdaptiveDecisionRequest Request;

    UPROPERTY(Transient, DuplicateTransient)
    FAdaptiveDecision Decision;

    FGuid SessionId;
    int64 AgentId = INDEX_NONE;
    int64 EpisodeId = INDEX_NONE;
    // Never reset within this object's lifetime, including reinitialization/new sessions.
    int64 NextDecisionId = 0;
    EAdaptiveRequestStatus RequestStatus = EAdaptiveRequestStatus::Unknown;
    FString RequestError;
    double DeadlineSeconds = 0.0;
};
