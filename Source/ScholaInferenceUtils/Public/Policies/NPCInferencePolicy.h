// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "Policies/PolicyInterface.h"
#include "Inference/NPCBackgroundInferenceSubsystem.h"
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
    MassEntity,
    SharedQueue,
    BatchRoundRobin,
    WorkStealing,
    LeastLoaded
};

UENUM(BlueprintType)
enum class ENPCSearchStrategy : uint8
{
    MonteCarloTreeSearch,
    Minimax
};

/** Background uses dedicated workers; the other paradigms are deterministic prototypes. */
USTRUCT(BlueprintType)
struct SCHOLAINFERENCEUTILS_API FNPCInferenceConfig
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference") ENPCInferenceParadigm Paradigm = ENPCInferenceParadigm::Adaptive;
    /** Any subscribed event authorizes one successful batch (or one Think call). */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference") TArray<FName> Events;
    /** Background: overrides BackgroundSettings.MaxBatchSize. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference") int32 MaxBatchSize = 32;
    /** Background: overrides BackgroundSettings.WorkerCount. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference") int32 WorkerCount = 1;
    /** Background: overrides BackgroundSettings.Dispatch. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference") ENPCInferenceFanOut FanOut = ENPCInferenceFanOut::Contiguous;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Instanced, Category = "Schola|Inference") TObjectPtr<UNPCBackgroundBackendProvider> BackgroundProvider;
    /** WorkerCount, MaxBatchSize, and Dispatch here are ignored: the fields above take precedence (a warning is logged on conflict). */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference") FNPCBackgroundSchedulerSettings BackgroundSettings;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Schola|Inference") bool bShareBackgroundBucket = true;
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

/** Stored payload for the private passthrough transport; reflected for GC. */
USTRUCT()
struct FNPCBackgroundPointRequest
{
    GENERATED_BODY()
    UPROPERTY() FNPCBackgroundRequest BackendRequest;
    UPROPERTY() TArray<TInstancedStruct<FPoint>> Observations;
    int32 Rows = 0;
    bool bCancelled = false;
};

/** Background inference facade; other paradigms retain their prototype behavior.
 * Background calls are thread-safe. Configure/Init/teardown run on the game thread.
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
    bool IsInferenceBusy() const override;

    UFUNCTION(BlueprintCallable, Category="Schola|Inference")
    ENPCBackgroundStatus SubmitBackground(const TArray<TInstancedStruct<FPoint>>& Observations, const TArray<int64>& Keys, FNPCBackgroundRequest& OutRequest);
    UFUNCTION(BlueprintPure, Category="Schola|Inference") ENPCBackgroundStatus GetBackgroundStatus(FNPCBackgroundRequest Request) const;
    UFUNCTION(BlueprintCallable, Category="Schola|Inference") ENPCBackgroundStatus ConsumeBackground(FNPCBackgroundRequest Request, TArray<TInstancedStruct<FPoint>>& Actions);
    UFUNCTION(BlueprintCallable, Category="Schola|Inference") bool CancelBackground(FNPCBackgroundRequest Request);
    UFUNCTION(BlueprintPure, Category="Schola|Inference") FNPCBackgroundStats GetBackgroundStats() const;
    bool GetBackgroundBinding(TSharedPtr<FNPCBackgroundScheduler, ESPMode::ThreadSafe>& Scheduler, TSharedPtr<const FNPCRowCodec, ESPMode::ThreadSafe>& Codec, uint32& Generation) const;
    TSharedPtr<FNPCCohortScheduler> GetBackgroundCohorts() const;
    virtual void BeginDestroy() override;

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
    FNPCInferenceConfig GetConfig() const;
    UFUNCTION(BlueprintPure, Category = "Schola|Inference")
    FNPCInferenceStats GetStats() const;

private:
    void ReleaseBackground();
    ENPCBackgroundStatus SubmitBackgroundLocked(const TArray<TInstancedStruct<FPoint>>& Observations, const TArray<int64>& Keys, FNPCBackgroundRequest& OutRequest);
    ENPCBackgroundStatus ConsumeBackgroundLocked(FNPCBackgroundRequest Request, TArray<TInstancedStruct<FPoint>>& Actions);
    bool ThinkBackground(const TArray<TInstancedStruct<FPoint>>& Observations, const TArray<int64>& Keys, TArray<TInstancedStruct<FPoint>>& Actions);
    TSharedPtr<FNPCCohortScheduler> BackgroundCohorts;
    mutable FCriticalSection BackgroundMutex;
    TSharedPtr<FNPCBackgroundScheduler, ESPMode::ThreadSafe> BackgroundScheduler;
    TSharedPtr<const FNPCRowCodec, ESPMode::ThreadSafe> BackgroundCodec;
    TWeakObjectPtr<UNPCBackgroundInferenceSubsystem> BackgroundSubsystem;
    FNPCBackgroundBucketHandle BackgroundBucket;
    uint32 BackgroundGeneration = 0;
    int64 NextRequestId = 1;
    int64 RecordedBackendCalls = 0;
    void RecordBackgroundCallsLocked();
    int32 LastSubmissionRows = 1;
    /** Rows of BackgroundRequests entries, including cancelled ones not yet consumed. */
    int32 HeldBackgroundRows = 0;
    void RemoveBackgroundRequestLocked(int64 Id);
    bool IsBackgroundFullLocked(int32 Rows) const;
    bool bWarnedSynchronous = false;
    UPROPERTY() TMap<int64, FNPCBackgroundPointRequest> BackgroundRequests;
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
