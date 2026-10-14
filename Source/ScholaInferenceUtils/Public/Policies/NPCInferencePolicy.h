// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "Policies/PolicyInterface.h"
#include "NPCInferencePolicy.generated.h"

/** Selection for the mock inference API. Existing IPolicy implementations are unchanged. */
UENUM(BlueprintType)
enum class ENPCInferenceParadigm : uint8
{
    Adaptive,
    EventBased,
    Background,
    SharedPolicy,
    AdvancedStrategy
};

UENUM(BlueprintType)
enum class ENPCInferenceFanOut : uint8
{
    Contiguous,
    RoundRobin,
    MassEntity
};

UENUM(BlueprintType)
enum class ENPCSearchStrategy : uint8
{
    MonteCarloTreeSearch,
    Minimax
};

/** All execution is synchronous and deterministic; no training, workers or Mass runtime is used. */
USTRUCT(BlueprintType)
struct SCHOLAINFERENCEUTILS_API FNPCInferenceConfig
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference") ENPCInferenceParadigm Paradigm = ENPCInferenceParadigm::Adaptive;
    /** Any subscribed event authorizes one successful batch (or one Think call). */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference") TArray<FName> Events;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference") int32 MaxBatchSize = 32;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference") int32 WorkerCount = 1;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference") ENPCInferenceFanOut FanOut = ENPCInferenceFanOut::Contiguous;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference") FName Head = NAME_None;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference") ENPCSearchStrategy SearchStrategy = ENPCSearchStrategy::MonteCarloTreeSearch;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference") int32 SearchBudget = 64;
};

/** Mock online-learning sample. Nonnegative reward stores a preferred action for this behavior. */
USTRUCT(BlueprintType)
struct SCHOLAINFERENCEUTILS_API FPlayerBehaviorFeedback
{
    GENERATED_BODY()
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference") TInstancedStruct<FPoint> PlayerBehavior;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference") TInstancedStruct<FPoint> PreferredAction;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference") float Reward = 0.0f;
};

/** Caller-supplied simulated opponent replies; scores are from the NPC's perspective. */
USTRUCT(BlueprintType)
struct SCHOLAINFERENCEUTILS_API FStrategySearchCandidate
{
    GENERATED_BODY()
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference") TInstancedStruct<FPoint> Action;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference") TArray<float> OpponentReplyScores;
};

/** Diagnostics make mock routing and shared computation observable to API callers. */
USTRUCT(BlueprintType)
struct SCHOLAINFERENCEUTILS_API FNPCInferenceStats
{
    GENERATED_BODY()
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int32 LearningUpdates = 0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int32 TrunkEvaluations = 0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int32 HeadEvaluations = 0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int32 BatchDispatches = 0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") int32 SearchEvaluations = 0;
    /** Lane per observation in the last background request, in original input order. */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") TArray<int32> WorkerLanes;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Schola|Inference") TArray<int64> EntityIds;
};

/**
 * API-only mock for all five NPC paradigms, compatible with IPolicy and existing steppers.
 * Outputs are copied points and may not fit a game's action space. Use only for prototypes.
 * All methods must be called serially, including when using a pipelined stepper.
 * Failed calls preserve outputs and do not consume event permits.
 */
UCLASS(BlueprintType, Blueprintable, EditInlineNew, DefaultToInstanced)
class SCHOLAINFERENCEUTILS_API UNPCInferencePolicy : public UObject, public IPolicy
{
    GENERATED_BODY()
public:
    /** Validates configuration and resets all feedback, events, heads, candidates and stats. */
    UFUNCTION(BlueprintCallable, Category = "Schola|Inference")
    bool Configure(const FNPCInferenceConfig& InConfig);
    UFUNCTION(BlueprintCallable, Category = "Schola|Inference")
    bool Init(const FInteractionDefinition& InDefinition) override;
    UFUNCTION(BlueprintCallable, Category = "Schola|Inference")
    bool Think(const TInstancedStruct<FPoint>& Observation, TInstancedStruct<FPoint>& Action) override;
    UFUNCTION(BlueprintCallable, Category = "Schola|Inference")
    bool BatchedThink(const TArray<TInstancedStruct<FPoint>>& Observations,
                      TArray<TInstancedStruct<FPoint>>& Actions) override;
    UFUNCTION(BlueprintPure, Category = "Schola|Inference")
    bool IsInferenceBusy() const override { return false; }

    UFUNCTION(BlueprintCallable, Category = "Schola|Inference")
    bool SubmitFeedback(const FPlayerBehaviorFeedback& Feedback);
    /** Unsubscribed events return false; notifications coalesce until successful inference. */
    UFUNCTION(BlueprintCallable, Category = "Schola|Inference")
    bool NotifyEvent(FName Event);
    /** Heads specialize a shared copied representation with a fixed mock action. */
    UFUNCTION(BlueprintCallable, Category = "Schola|Inference")
    bool RegisterHead(FName Head, const TInstancedStruct<FPoint>& SpecializedAction);
    /** Encodes once, then evaluates all requested heads. Output order matches Heads. */
    UFUNCTION(BlueprintCallable, Category = "Schola|Inference")
    bool ThinkShared(const TInstancedStruct<FPoint>& Observation, const TArray<FName>& Heads,
                     TArray<TInstancedStruct<FPoint>>& Actions);
    UFUNCTION(BlueprintCallable, Category = "Schola|Inference")
    bool SetSearchCandidates(const TArray<FStrategySearchCandidate>& Candidates);
    /** Mock Mass adapter: stable IDs map input rows to output rows without a Mass dependency. */
    UFUNCTION(BlueprintCallable, Category = "Schola|Inference")
    bool ThinkEntities(const TArray<int64>& EntityIds,
                       const TArray<TInstancedStruct<FPoint>>& Observations,
                       TArray<TInstancedStruct<FPoint>>& Actions);
    UFUNCTION(BlueprintPure, Category = "Schola|Inference")
    FNPCInferenceConfig GetConfig() const { return Config; }
    UFUNCTION(BlueprintPure, Category = "Schola|Inference")
    FNPCInferenceStats GetStats() const { return Stats; }

private:
    bool ValidateReady(const TArray<TInstancedStruct<FPoint>>& Observations) const;
    TInstancedStruct<FPoint> InferMock(const TInstancedStruct<FPoint>& Observation);
    UPROPERTY() FNPCInferenceConfig Config;
    UPROPERTY() FNPCInferenceStats Stats;
    UPROPERTY() TArray<FPlayerBehaviorFeedback> FeedbackSamples;
    UPROPERTY() TMap<FName, TInstancedStruct<FPoint>> HeadActions;
    UPROPERTY() TArray<FStrategySearchCandidate> SearchCandidates;
    bool bEventPending = false;
    bool bInitialized = false;
};
