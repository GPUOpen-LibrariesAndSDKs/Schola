// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "Policies/NPCInferencePolicy.h"

namespace
{
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
    // Mock points are not validated against model shapes; no model is loaded.
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
    if (!ValidateReady(Observations)) { return false; }
    TArray<TInstancedStruct<FPoint>> Results;
    Results.Reserve(Observations.Num());
    if (Config.Paradigm == ENPCInferenceParadigm::Background)
    {
        Stats.WorkerLanes.Reset();
        Stats.EntityIds.Reset();
        for (int32 Index = 0; Index < Observations.Num(); ++Index)
        {
            const int32 Batch = Index / Config.MaxBatchSize;
            if (Index % Config.MaxBatchSize == 0) { ++Stats.BatchDispatches; }
            Stats.WorkerLanes.Add(Config.FanOut == ENPCInferenceFanOut::RoundRobin
                ? Index % Config.WorkerCount : Batch % Config.WorkerCount);
            Results.Add(InferMock(Observations[Index]));
        }
    }
    else
    {
        for (const auto& Observation : Observations) { Results.Add(InferMock(Observation)); }
    }
    if (Config.Paradigm == ENPCInferenceParadigm::EventBased) { bEventPending = false; }
    Actions = MoveTemp(Results);
    return true;
}

bool UNPCInferencePolicy::ThinkEntities(const TArray<int64>& EntityIds,
    const TArray<TInstancedStruct<FPoint>>& Observations, TArray<TInstancedStruct<FPoint>>& Actions)
{
    if (Config.Paradigm != ENPCInferenceParadigm::Background || Config.FanOut != ENPCInferenceFanOut::MassEntity
        || EntityIds.Num() != Observations.Num()) { return false; }
    TSet<int64> Seen;
    for (int64 Id : EntityIds)
    {
        if (Id < 0 || Seen.Contains(Id)) { return false; }
        Seen.Add(Id);
    }
    if (!BatchedThink(Observations, Actions)) { return false; }
    Stats.EntityIds = EntityIds;
    for (int32 Index = 0; Index < EntityIds.Num(); ++Index)
    {
        Stats.WorkerLanes[Index] = static_cast<int32>(EntityIds[Index] % Config.WorkerCount);
    }
    return true;
}
