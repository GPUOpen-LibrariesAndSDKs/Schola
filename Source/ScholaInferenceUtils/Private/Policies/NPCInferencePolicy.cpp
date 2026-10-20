// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "Policies/NPCInferencePolicy.h"
#include "Misc/ScopeLock.h"
#include "LogScholaInferenceUtils.h"
#include "Engine/World.h"
#include "UObject/GarbageCollection.h"
#include "UObject/StrongObjectPtr.h"
#include <atomic>

namespace
{
// Protect reflected point/map mutations from GC on compatibility worker threads.
// Guards cover only encoding/publication; no guard or policy lock spans inference.
class FNPCPolicyGCGuard
{
    TOptional<FGCScopeGuard> Guard;
public:
    FNPCPolicyGCGuard() { if(!IsInGameThread())Guard.Emplace(); }
};
// Legacy config fields win; warn (once per process) when a non-default BackgroundSettings value is overridden.
void WarnOverriddenBackgroundSettings(const FNPCBackgroundSchedulerSettings& Requested, const FNPCBackgroundSchedulerSettings& Effective)
{
    static std::atomic<bool> bWarned{false};
    const FNPCBackgroundSchedulerSettings Defaults;
    TArray<FString> Fields;
    if (Requested.WorkerCount != Defaults.WorkerCount && Requested.WorkerCount != Effective.WorkerCount) Fields.Add(TEXT("WorkerCount"));
    if (Requested.MaxBatchSize != Defaults.MaxBatchSize && Requested.MaxBatchSize != Effective.MaxBatchSize) Fields.Add(TEXT("MaxBatchSize"));
    if (Requested.Dispatch != Defaults.Dispatch && Requested.Dispatch != Effective.Dispatch) Fields.Add(TEXT("Dispatch"));
    if (Fields.IsEmpty() || bWarned.exchange(true)) return;
    UE_LOG(LogScholaInferenceUtils, Warning, TEXT("BackgroundSettings.%s ignored: FNPCInferenceConfig WorkerCount/MaxBatchSize/FanOut take precedence"),
        *FString::Join(Fields, TEXT(", BackgroundSettings.")));
}
// Blueprint instanced-struct pins can carry unrelated structs; validate in every entry point.
bool IsPoint(const TInstancedStruct<FPoint>& Point)
{
    return Point.IsValid() && Point.GetScriptStruct()->IsChildOf(FPoint::StaticStruct());
}
}

bool UNPCInferencePolicy::Configure(const FNPCInferenceConfig& InConfig)
{
    if (InConfig.MaxBatchSize <= 0 || InConfig.WorkerCount <= 0 || InConfig.SearchBudget <= 0
        || !StaticEnum<ENPCInferenceParadigm>()->IsValidEnumValue(static_cast<int64>(InConfig.Paradigm))
        || !StaticEnum<ENPCInferenceFanOut>()->IsValidEnumValue(static_cast<int64>(InConfig.FanOut))
        || !StaticEnum<ENPCSearchStrategy>()->IsValidEnumValue(static_cast<int64>(InConfig.SearchStrategy)))
    {
        return false;
    }
    if (InConfig.Paradigm == ENPCInferenceParadigm::EventBased
        && (InConfig.Events.IsEmpty() || InConfig.Events.Contains(NAME_None)))
    {
        return false;
    }
    if (InConfig.Paradigm == ENPCInferenceParadigm::SharedPolicy && InConfig.Head.IsNone())
    {
        return false;
    }
    FScopeLock Lock(&BackgroundMutex);
    ReleaseBackground();
    Config = InConfig;
    Stats = FNPCInferenceStats();
    FeedbackSamples.Reset();
    HeadActions.Reset();
    SearchCandidates.Reset();
    bEventPending = false;
    bInitialized = false;
    return true;
}

bool UNPCInferencePolicy::Init(const FInteractionDefinition& InDefinition)
{
    FScopeLock Lock(&BackgroundMutex);
    ReleaseBackground();
    if (Config.Paradigm == ENPCInferenceParadigm::Background)
    {
        auto* Provider = Config.BackgroundProvider.Get();
        if (!Provider) Provider = GetMutableDefault<UNPCPassthroughBackendProvider>();
        auto Settings = Config.BackgroundSettings;
        Settings.WorkerCount = Config.WorkerCount;
        Settings.MaxBatchSize = Config.MaxBatchSize;
        switch (Config.FanOut)
        {
        case ENPCInferenceFanOut::Contiguous: Settings.Dispatch = ENPCBackgroundDispatch::Contiguous; break;
        case ENPCInferenceFanOut::RoundRobin: Settings.Dispatch = ENPCBackgroundDispatch::RowRoundRobin; break;
        case ENPCInferenceFanOut::MassEntity: Settings.Dispatch = ENPCBackgroundDispatch::KeyAffinity; break;
        case ENPCInferenceFanOut::SharedQueue: Settings.Dispatch = ENPCBackgroundDispatch::SharedQueue; break;
        case ENPCInferenceFanOut::BatchRoundRobin: Settings.Dispatch = ENPCBackgroundDispatch::BatchRoundRobin; break;
        case ENPCInferenceFanOut::WorkStealing: Settings.Dispatch = ENPCBackgroundDispatch::WorkStealing; break;
        case ENPCInferenceFanOut::LeastLoaded: Settings.Dispatch = ENPCBackgroundDispatch::LeastLoaded; break;
        }
        WarnOverriddenBackgroundSettings(Config.BackgroundSettings, Settings);
        FString Error;
        if (GetWorld() && Config.bShareBackgroundBucket && !Provider->IsA<UNPCPassthroughBackendProvider>())
        {
            auto* Subsystem = GetWorld()->GetSubsystem<UNPCBackgroundInferenceSubsystem>();
            BackgroundBucket = Subsystem->AcquireBucket(*Provider, InDefinition, Settings, Error);
            BackgroundSubsystem = Subsystem;
            BackgroundScheduler = Subsystem->ShareScheduler(BackgroundBucket);
            BackgroundCodec = Subsystem->ShareCodec(BackgroundBucket);
            // BackgroundGeneration is policy-local. Reacquiring a still-shared bucket
            // must not restore the epoch held by a pre-Configure stepper.
            BackgroundCohorts = Subsystem->ShareCohorts(BackgroundBucket);
        }
        else
        {
            BackgroundCohorts = MakeShared<FNPCCohortScheduler>();
            FNPCRowLayout Layout;
            auto Backend = Provider->CreateBackend(InDefinition, Layout, Error);
            if (Backend && !Provider->IsA<UNPCPassthroughBackendProvider>())
            {
                auto Codec = FNPCRowCodec::Create(InDefinition, Layout, Error);
                if (!Codec) return false;
                BackgroundCodec = MakeShared<FNPCRowCodec, ESPMode::ThreadSafe>(MoveTemp(Codec.GetValue()));
            }
            if (Backend) BackgroundScheduler = FNPCBackgroundScheduler::Create(Settings, Backend.ToSharedRef(), &Error);
        }
        if (!BackgroundScheduler)
        { UE_LOG(LogScholaInferenceUtils, Error, TEXT("Background initialization failed: %s"), *Error); return false; }
    }
    RecordedBackendCalls = BackgroundScheduler ? BackgroundScheduler->GetBackendCallCount() : 0;
    bInitialized = true;
    return true;
}

bool UNPCInferencePolicy::SubmitFeedback(const FPlayerBehaviorFeedback& Feedback)
{
    if (Config.Paradigm != ENPCInferenceParadigm::Adaptive || !IsPoint(Feedback.PlayerBehavior)
        || !IsPoint(Feedback.PreferredAction) || !FMath::IsFinite(Feedback.Reward))
    {
        return false;
    }
    FeedbackSamples.Add(Feedback);
    ++Stats.LearningUpdates;
    return true;
}

bool UNPCInferencePolicy::NotifyEvent(FName Event)
{
    if (Config.Paradigm != ENPCInferenceParadigm::EventBased || !Config.Events.Contains(Event))
    {
        return false;
    }
    bEventPending = true;
    return true;
}

bool UNPCInferencePolicy::RegisterHead(FName Head, const TInstancedStruct<FPoint>& SpecializedAction)
{
    if (Config.Paradigm != ENPCInferenceParadigm::SharedPolicy || Head.IsNone() || !IsPoint(SpecializedAction))
    {
        return false;
    }
    HeadActions.Add(Head, SpecializedAction);
    return true;
}

bool UNPCInferencePolicy::ThinkShared(const TInstancedStruct<FPoint>& Observation,
    const TArray<FName>& Heads, TArray<TInstancedStruct<FPoint>>& Actions)
{
    if (!bInitialized || Config.Paradigm != ENPCInferenceParadigm::SharedPolicy
        || !IsPoint(Observation) || Heads.IsEmpty())
    {
        return false;
    }
    for (FName Head : Heads)
    {
        if (!HeadActions.Contains(Head)) { return false; }
    }
    TArray<TInstancedStruct<FPoint>> Results;
    // One mock trunk representation reused by every specialized head.
    ++Stats.TrunkEvaluations;
    for (FName Head : Heads)
    {
        Results.Add(HeadActions[Head]);
        ++Stats.HeadEvaluations;
    }
    Actions = MoveTemp(Results);
    return true;
}

bool UNPCInferencePolicy::SetSearchCandidates(const TArray<FStrategySearchCandidate>& Candidates)
{
    if (Config.Paradigm != ENPCInferenceParadigm::AdvancedStrategy || Candidates.IsEmpty())
    {
        return false;
    }
    // Require enough budget to evaluate every supplied opponent reply: no unvisited-action bias.
    int64 RequiredBudget = 0;
    for (const FStrategySearchCandidate& Candidate : Candidates)
    {
        if (!IsPoint(Candidate.Action) || Candidate.OpponentReplyScores.IsEmpty()) { return false; }
        RequiredBudget += Candidate.OpponentReplyScores.Num();
        for (float Score : Candidate.OpponentReplyScores)
        {
            if (!FMath::IsFinite(Score)) { return false; }
        }
    }
    if (RequiredBudget > Config.SearchBudget) { return false; }
    SearchCandidates = Candidates;
    return true;
}

bool UNPCInferencePolicy::ValidateReady(const TArray<TInstancedStruct<FPoint>>& Observations) const
{
    if (!bInitialized || Observations.IsEmpty()) { return false; }
    for (const auto& Observation : Observations)
    {
        if (!IsPoint(Observation)) { return false; }
    }
    switch (Config.Paradigm)
    {
    case ENPCInferenceParadigm::EventBased: return bEventPending;
    case ENPCInferenceParadigm::SharedPolicy: return HeadActions.Contains(Config.Head);
    case ENPCInferenceParadigm::AdvancedStrategy: return !SearchCandidates.IsEmpty();
    default: return true;
    }
}

TInstancedStruct<FPoint> UNPCInferencePolicy::InferMock(const TInstancedStruct<FPoint>& Observation)
{
    switch (Config.Paradigm)
    {
    case ENPCInferenceParadigm::Adaptive:
        // Simulate adaptation with most recent nonnegative feedback for identical player behavior.
        for (int32 Index = FeedbackSamples.Num() - 1; Index >= 0; --Index)
        {
            const auto& Feedback = FeedbackSamples[Index];
            if (Feedback.Reward >= 0.0f && Feedback.PlayerBehavior == Observation)
            {
                return Feedback.PreferredAction;
            }
        }
        return Observation;
    case ENPCInferenceParadigm::SharedPolicy:
        ++Stats.TrunkEvaluations;
        ++Stats.HeadEvaluations;
        return HeadActions[Config.Head];
    case ENPCInferenceParadigm::AdvancedStrategy:
    {
        double BestScore = -TNumericLimits<double>::Max();
        int32 BestIndex = 0;
        for (int32 Index = 0; Index < SearchCandidates.Num(); ++Index)
        {
            const auto& Scores = SearchCandidates[Index].OpponentReplyScores;
            double Score = Config.SearchStrategy == ENPCSearchStrategy::Minimax
                ? TNumericLimits<double>::Max() : 0.0;
            for (float Reply : Scores)
            {
                Score = Config.SearchStrategy == ENPCSearchStrategy::Minimax
                    ? FMath::Min(Score, static_cast<double>(Reply)) : Score + Reply;
                ++Stats.SearchEvaluations;
            }
            // Mock MCTS uses the mean of supplied rollouts; no real tree or world simulation.
            if (Config.SearchStrategy == ENPCSearchStrategy::MonteCarloTreeSearch) { Score /= Scores.Num(); }
            if (Score > BestScore) { BestScore = Score; BestIndex = Index; }
        }
        return SearchCandidates[BestIndex].Action;
    }
    default: return Observation;
    }
}

bool UNPCInferencePolicy::Think(const TInstancedStruct<FPoint>& Observation, TInstancedStruct<FPoint>& Action)
{
    TArray<TInstancedStruct<FPoint>> Results;
    if (!BatchedThink({Observation}, Results)) { return false; }
    Action = Results[0];
    return true;
}

bool UNPCInferencePolicy::BatchedThink(const TArray<TInstancedStruct<FPoint>>& Observations,
    TArray<TInstancedStruct<FPoint>>& Actions)
{
    TStrongObjectPtr<UNPCInferencePolicy> KeepAlive;
    bool bBackground;
    {
        FNPCPolicyGCGuard GCGuard;
        bBackground=GetConfig().Paradigm==ENPCInferenceParadigm::Background;
        if(bBackground)KeepAlive.Reset(this);
    }
    if (bBackground) return ThinkBackground(Observations, {}, Actions);
    if (!ValidateReady(Observations)) return false;
    TArray<TInstancedStruct<FPoint>> Results;
    Results.Reserve(Observations.Num());
    for (const auto& Observation : Observations) Results.Add(InferMock(Observation));
    if (Config.Paradigm == ENPCInferenceParadigm::EventBased) { bEventPending = false; }
    Actions = MoveTemp(Results);
    return true;
}

bool UNPCInferencePolicy::ThinkEntities(const TArray<int64>& EntityIds,
    const TArray<TInstancedStruct<FPoint>>& Observations, TArray<TInstancedStruct<FPoint>>& Actions)
{
    TStrongObjectPtr<UNPCInferencePolicy> KeepAlive;
    FNPCInferenceConfig CurrentConfig;
    { FNPCPolicyGCGuard GCGuard; KeepAlive.Reset(this); CurrentConfig=GetConfig(); }
    if (CurrentConfig.Paradigm != ENPCInferenceParadigm::Background || CurrentConfig.FanOut != ENPCInferenceFanOut::MassEntity
        || EntityIds.Num() != Observations.Num()) { return false; }
    TSet<int64> Seen;
    for (int64 Id : EntityIds)
    {
        if (Id < 0 || Seen.Contains(Id)) { return false; }
        Seen.Add(Id);
    }
    return ThinkBackground(Observations, EntityIds, Actions);
}

void UNPCInferencePolicy::ReleaseBackground()
{
    bInitialized = false;
    for (const auto& Pair : BackgroundRequests)
        if (BackgroundScheduler) BackgroundScheduler->Cancel(Pair.Value.BackendRequest);
    BackgroundRequests.Empty();
    HeldBackgroundRows = 0;
    if (auto* Subsystem = BackgroundSubsystem.Get()) Subsystem->ReleaseBucket(BackgroundBucket);
    else if (BackgroundScheduler) BackgroundScheduler->Shutdown();
    RecordBackgroundCallsLocked();
    BackgroundScheduler.Reset(); BackgroundCodec.Reset(); BackgroundCohorts.Reset(); BackgroundSubsystem.Reset(); BackgroundBucket = {};
    ++BackgroundGeneration;
}
void UNPCInferencePolicy::BeginDestroy()
{
    { FScopeLock Lock(&BackgroundMutex); ReleaseBackground(); }
    Super::BeginDestroy();
}
FNPCInferenceConfig UNPCInferencePolicy::GetConfig() const
{ FScopeLock Lock(&BackgroundMutex); return Config; }
FNPCInferenceStats UNPCInferencePolicy::GetStats() const
{
    FScopeLock Lock(&BackgroundMutex);
    auto Out=Stats;
    if(BackgroundScheduler) Out.BatchDispatches+=int32(FMath::Max(int64(0),BackgroundScheduler->GetBackendCallCount()-RecordedBackendCalls));
    return Out;
}
void UNPCInferencePolicy::RecordBackgroundCallsLocked()
{
    if(!BackgroundScheduler)return;
    const int64 Calls=BackgroundScheduler->GetBackendCallCount();
    Stats.BatchDispatches+=int32(FMath::Max(int64(0),Calls-RecordedBackendCalls));
    RecordedBackendCalls=Calls;
}
bool UNPCInferencePolicy::IsInferenceBusy() const
{
    FScopeLock Lock(&BackgroundMutex);
    return Config.Paradigm == ENPCInferenceParadigm::Background &&
        (!BackgroundScheduler || IsBackgroundFullLocked(LastSubmissionRows));
}
bool UNPCInferencePolicy::IsBackgroundFullLocked(int32 Rows) const
{
    return HeldBackgroundRows + Rows > Config.BackgroundSettings.MaxPendingRows || !BackgroundScheduler->CanAccept(Rows);
}
void UNPCInferencePolicy::RemoveBackgroundRequestLocked(int64 Id)
{
    FNPCBackgroundPointRequest Removed;
    if (BackgroundRequests.RemoveAndCopyValue(Id, Removed)) HeldBackgroundRows -= Removed.Rows;
}
FNPCBackgroundStats UNPCInferencePolicy::GetBackgroundStats() const
{ FScopeLock Lock(&BackgroundMutex); return BackgroundScheduler ? BackgroundScheduler->GetStats() : FNPCBackgroundStats{}; }
bool UNPCInferencePolicy::GetBackgroundBinding(TSharedPtr<FNPCBackgroundScheduler, ESPMode::ThreadSafe>& Scheduler,
    TSharedPtr<const FNPCRowCodec, ESPMode::ThreadSafe>& Codec, uint32& Generation) const
{
    FScopeLock Lock(&BackgroundMutex);
    Scheduler = BackgroundScheduler; Codec = BackgroundCodec; Generation = BackgroundGeneration;
    return bInitialized && Scheduler.IsValid() && !Scheduler->IsShuttingDown();
}

ENPCBackgroundStatus UNPCInferencePolicy::SubmitBackgroundLocked(const TArray<TInstancedStruct<FPoint>>& Observations,
    const TArray<int64>& Keys, FNPCBackgroundRequest& OutRequest)
{
    OutRequest = {};
    if (Config.Paradigm != ENPCInferenceParadigm::Background || !BackgroundScheduler ||
        !ValidateReady(Observations) || (!Keys.IsEmpty() && Keys.Num() != Observations.Num())) return ENPCBackgroundStatus::Invalid;
    LastSubmissionRows = Observations.Num();
    if (IsBackgroundFullLocked(LastSubmissionRows)) return ENPCBackgroundStatus::Busy;
    TArray<float> Rows;
    Rows.SetNumUninitialized(Observations.Num() * BackgroundScheduler->GetObservationSize());
    if (BackgroundCodec)
    {
        const int32 Width = BackgroundScheduler->GetObservationSize();
        for (int32 I = 0; I < Observations.Num(); ++I)
            if (!BackgroundCodec->Encode(Observations[I], MakeArrayView(Rows).Slice(I * Width, Width))) return ENPCBackgroundStatus::Invalid;
    }
    else
    {
        if (Observations.Num() > 16777216) return ENPCBackgroundStatus::Invalid;
        for (int32 I = 0; I < Observations.Num(); ++I) Rows[I] = float(I);
    }
    TArray<int64> DefaultKeys;
    if (Keys.IsEmpty() && Config.FanOut == ENPCInferenceFanOut::MassEntity)
        for (int32 I = 0; I < Observations.Num(); ++I) DefaultKeys.Add(I / Config.MaxBatchSize);
    FNPCBackgroundSubmitOptions Options;
    Options.Keys = Keys.IsEmpty() ? TConstArrayView<int64>(DefaultKeys) : TConstArrayView<int64>(Keys);
    Options.UserTag = uint64(BackgroundGeneration) << 32;
    FNPCBackgroundPointRequest Stored;
    auto Status = BackgroundScheduler->Submit(Rows, Options, Stored.BackendRequest);
    if (Status != ENPCBackgroundStatus::Pending) return Status;
    if (!BackgroundCodec) Stored.Observations = Observations;
    Stored.Rows = Observations.Num();
    HeldBackgroundRows += Stored.Rows;
    OutRequest.Id = NextRequestId++;
    BackgroundRequests.Add(OutRequest.Id, MoveTemp(Stored));
    return Status;
}
ENPCBackgroundStatus UNPCInferencePolicy::SubmitBackground(const TArray<TInstancedStruct<FPoint>>& Observations,
    const TArray<int64>& Keys, FNPCBackgroundRequest& OutRequest)
{ FNPCPolicyGCGuard GCGuard; FScopeLock Lock(&BackgroundMutex); return SubmitBackgroundLocked(Observations, Keys, OutRequest); }
ENPCBackgroundStatus UNPCInferencePolicy::GetBackgroundStatus(FNPCBackgroundRequest Request) const
{
    FScopeLock Lock(&BackgroundMutex);
    const auto* Stored = BackgroundRequests.Find(Request.Id);
    if (!Stored || !BackgroundScheduler) return ENPCBackgroundStatus::Invalid;
    return (Stored->bCancelled || BackgroundScheduler->IsShuttingDown()) ? ENPCBackgroundStatus::Cancelled : BackgroundScheduler->GetStatus(Stored->BackendRequest);
}
bool UNPCInferencePolicy::CancelBackground(FNPCBackgroundRequest Request)
{
    FNPCPolicyGCGuard GCGuard;
    FScopeLock Lock(&BackgroundMutex);
    auto* Stored = BackgroundRequests.Find(Request.Id);
    if (!Stored || !BackgroundScheduler || Stored->bCancelled) return false;
    if (!BackgroundScheduler->Cancel(Stored->BackendRequest)) return false;
    Stored->bCancelled = true;
    Stored->Observations.Empty();
    return true;
}
ENPCBackgroundStatus UNPCInferencePolicy::ConsumeBackgroundLocked(FNPCBackgroundRequest Request,
    TArray<TInstancedStruct<FPoint>>& Actions)
{
    auto* Stored = BackgroundRequests.Find(Request.Id);
    if (!Stored || !BackgroundScheduler) return ENPCBackgroundStatus::Invalid;
    if (Stored->bCancelled) { RemoveBackgroundRequestLocked(Request.Id); return ENPCBackgroundStatus::Cancelled; }
    FNPCBackgroundResult Result;
    const auto Status = BackgroundScheduler->Consume(Stored->BackendRequest, Result);
    RecordBackgroundCallsLocked();
    if (Status == ENPCBackgroundStatus::Pending) return Status;
    TArray<TInstancedStruct<FPoint>> Decoded;
    const bool bStopping = BackgroundScheduler->IsShuttingDown();
    bool bValid = !bStopping && Status == ENPCBackgroundStatus::Completed && uint32(Result.UserTag >> 32) == BackgroundGeneration;
    if (bValid)
    {
        Decoded.SetNum(Result.Rows);
        for (int32 I = 0; I < Result.Rows && bValid; ++I)
        {
            if (BackgroundCodec)
                bValid = BackgroundCodec->Decode(MakeArrayView(Result.Actions).Slice(I * Result.ActionSize, Result.ActionSize), Decoded[I]);
            else
            {
                const float Index = Result.Actions[I];
                bValid = FMath::IsFinite(Index) && Index >= 0 && Index < Stored->Observations.Num() && FMath::FloorToFloat(Index) == Index;
                if (bValid) Decoded[I] = Stored->Observations[int32(Index)];
            }
        }
    }
    if (bValid)
    {
        Stats.WorkerLanes = Result.RowWorkers;
        Stats.EntityIds = Result.Keys;
        Actions = MoveTemp(Decoded);
    }
    RemoveBackgroundRequestLocked(Request.Id);
    return bStopping ? ENPCBackgroundStatus::Cancelled : Status == ENPCBackgroundStatus::Completed && !bValid ? ENPCBackgroundStatus::Failed : Status;
}
ENPCBackgroundStatus UNPCInferencePolicy::ConsumeBackground(FNPCBackgroundRequest Request,
    TArray<TInstancedStruct<FPoint>>& Actions)
{ FNPCPolicyGCGuard GCGuard; FScopeLock Lock(&BackgroundMutex); return ConsumeBackgroundLocked(Request, Actions); }
bool UNPCInferencePolicy::ThinkBackground(const TArray<TInstancedStruct<FPoint>>& Observations,
    const TArray<int64>& Keys, TArray<TInstancedStruct<FPoint>>& Actions)
{
    TSharedPtr<FNPCBackgroundScheduler,ESPMode::ThreadSafe> Scheduler;
    FNPCBackgroundRequest Request,BackendRequest;
    uint32 Generation;
    {
        FNPCPolicyGCGuard GCGuard;
        FScopeLock Lock(&BackgroundMutex);
        if (IsInGameThread() && BackgroundCodec && !bWarnedSynchronous)
        {
            UE_LOG(LogScholaInferenceUtils, Warning, TEXT("Synchronous Background inference blocks the game thread; use SubmitBackground or BackgroundStepper for deferred delivery"));
            bWarnedSynchronous = true;
        }
        if (SubmitBackgroundLocked(Observations,Keys,Request)!=ENPCBackgroundStatus::Pending)return false;
        Scheduler=BackgroundScheduler;Generation=BackgroundGeneration;
        BackendRequest=BackgroundRequests.FindChecked(Request.Id).BackendRequest;
    }
    Scheduler->Wait(BackendRequest);
    FNPCPolicyGCGuard GCGuard;
    FScopeLock Lock(&BackgroundMutex);
    if(Scheduler!=BackgroundScheduler || Generation!=BackgroundGeneration)return false;
    const auto Status=ConsumeBackgroundLocked(Request,Actions);
    if(Keys.IsEmpty())Stats.EntityIds.Empty();
    return Status==ENPCBackgroundStatus::Completed;
}

TSharedPtr<FNPCCohortScheduler> UNPCInferencePolicy::GetBackgroundCohorts() const
{ FScopeLock Lock(&BackgroundMutex); return BackgroundCohorts; }
