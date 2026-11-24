// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "Steppers/AdaptiveStepper.h"
#include "Steppers/AdaptiveAgentInterface.h"
#include "Policies/AdaptiveNPCPolicy.h"
#include "LogScholaInferenceUtils.h"
#include "Logging/StructuredLog.h"

bool UAdaptiveStepper::Init(const TArray<TScriptInterface<IAgent>>& InAgents, const TScriptInterface<IPolicy>& InPolicy)
{
    check(IsInGameThread());
    Shutdown();
    Stats = FAdaptiveStepperStats();
    LastError.Reset();

    auto* Template = Cast<UAdaptiveNPCPolicy>(InPolicy.GetObject());
    if (!Template)
    {
        Fail(TEXT("UAdaptiveStepper::Init requires a UAdaptiveNPCPolicy template."));
        return false;
    }
    if (InAgents.Num() == 0)
    {
        Fail(TEXT("UAdaptiveStepper::Init was given no agents."));
        return false;
    }
    for (const TScriptInterface<IAgent>& Agent : InAgents)
    {
        UObject* Object = Agent.GetObject();
        if (!Object || !Object->Implements<UAgent>() || !Object->Implements<UAdaptiveAgent>())
        {
            Fail(TEXT("Every agent must implement both IAgent and IAdaptiveAgent."));
            return false;
        }
    }

    // Validate every binding before keeping any, so a failed Init leaves no half-built stepper.
    TArray<TObjectPtr<UAdaptiveNPCPolicy>> NewPolicies;
    for (int32 Index = 0; Index < InAgents.Num(); ++Index)
    {
        FInteractionDefinition Definition;
        IAgent::Execute_Define(InAgents[Index].GetObject(), Definition);
        // The template's outer supplies the world that native learner bindings are created in.
        auto* Policy = DuplicateObject<UAdaptiveNPCPolicy>(Template, Template->GetOuter());
        if (!Policy || !Policy->Init(Definition))
        {
            const FString Reason = Policy ? Policy->GetLastError() : TEXT("duplication failed");
            for (UAdaptiveNPCPolicy* Created : NewPolicies) Created->Shutdown();
            if (Policy) Policy->Shutdown();
            Fail(FString::Printf(TEXT("Policy for agent %d failed to initialize: %s"), Index, *Reason));
            return false;
        }
        NewPolicies.Add(Policy);
    }

    Agents = InAgents;
    Policies = MoveTemp(NewPolicies);
    Slots.SetNum(Agents.Num());
    for (int32 Index = 0; Index < Agents.Num(); ++Index)
    {
        Slots[Index] = FSlot();
        Slots[Index].Agent = Agents[Index];
        Slots[Index].Policy = Policies[Index];
    }
    return true;
}

void UAdaptiveStepper::Step()
{
    check(IsInGameThread());
    for (FSlot& Slot : Slots)
    {
        if (Slot.Policy && Slot.Policy->GetSessionState() == EAdaptiveSessionState::Ready
            && IAgent::Execute_GetStatus(Slot.Agent.GetObject()) == EAgentStatus::Running)
        {
            StepSlot(Slot);
        }
    }
}

void UAdaptiveStepper::StepSlot(FSlot& Slot)
{
    if (Slot.PendingTicket != INDEX_NONE)
    {
        // A request is outstanding: either its action arrives now, or we keep waiting this Step.
        if (PollPending(Slot) || Slot.PendingTicket != INDEX_NONE) return;
    }

    TInstancedStruct<FPoint> Observation;
    if (Slot.bIntervalOpen) CloseInterval(Slot, Observation);
    else IAgent::Execute_Observe(Slot.Agent.GetObject(), Observation);

    int64 Ticket = INDEX_NONE;
    if (!Slot.Policy->RequestDecision(Observation, Ticket, DecisionTimeoutSeconds))
    {
        Fail(TEXT("RequestDecision failed: ") + Slot.Policy->GetLastError());
        return;
    }
    Slot.PendingTicket = Ticket;
    Slot.PendingObservation = MoveTemp(Observation);
    PollPending(Slot); // synchronous backends complete during RequestDecision
}

bool UAdaptiveStepper::PollPending(FSlot& Slot)
{
    FAdaptiveDecision Decision;
    switch (Slot.Policy->TryGetDecision(Slot.PendingTicket, Decision))
    {
    case EAdaptiveRequestStatus::Pending:
        return false;
    case EAdaptiveRequestStatus::Ready:
        // Reward earned while waiting belongs to no decision; drop it so the new interval starts clean.
        IAdaptiveAgent::Execute_ConsumeReward(Slot.Agent.GetObject());
        IAgent::Execute_Act(Slot.Agent.GetObject(), Decision.Action);
        Slot.Applied = MoveTemp(Decision);
        Slot.AppliedObservation = MoveTemp(Slot.PendingObservation);
        Slot.bIntervalOpen = true;
        Slot.PendingTicket = INDEX_NONE;
        ++Stats.Decisions;
        return true;
    default:
        // Expired, Failed, Cancelled or Unknown: nothing was applied; request again next Step.
        ++Stats.Abandoned;
        Slot.PendingTicket = INDEX_NONE;
        Slot.PendingObservation.Reset();
        return false;
    }
}

void UAdaptiveStepper::CloseInterval(FSlot& Slot, TInstancedStruct<FPoint>& OutNextObservation)
{
    UObject* AgentObject = Slot.Agent.GetObject();
    IAgent::Execute_Observe(AgentObject, OutNextObservation);
    bool bTerminated = false, bTruncated = false;
    IAdaptiveAgent::Execute_GetEpisodeStatus(AgentObject, bTerminated, bTruncated);

    FAdaptiveExperience Experience;
    Experience.SessionId = Slot.Applied.SessionId;
    Experience.AgentId = Slot.Applied.AgentId;
    Experience.EpisodeId = Slot.Applied.EpisodeId;
    Experience.DecisionId = Slot.Applied.DecisionId;
    Experience.PolicyVersion = Slot.Applied.PolicyVersion;
    Experience.Observation = Slot.AppliedObservation;
    Experience.ExecutedAction = Slot.Applied.Action;
    Experience.Reward = IAdaptiveAgent::Execute_ConsumeReward(AgentObject);
    Experience.NextObservation = OutNextObservation;
    Experience.bTerminated = bTerminated;
    Experience.bTruncated = bTruncated;
    Slot.bIntervalOpen = false;

    // Submit before ResetEpisode: the policy rejects an experience whose episode is no longer current.
    if (Slot.Policy->SubmitExperience(Experience) == EAdaptiveSubmitResult::Accepted) ++Stats.Accepted;
    else
    {
        ++Stats.NotAccepted;
        LastError = Slot.Policy->GetLastError();
    }

    if (bTerminated || bTruncated)
    {
        IAdaptiveAgent::Execute_ResetEpisode(AgentObject);
        Slot.Policy->ResetEpisode();
        ++Stats.Episodes;
        // The ending observation belongs to the old encounter; the next decision sees the new one.
        OutNextObservation.Reset();
        IAgent::Execute_Observe(AgentObject, OutNextObservation);
    }
}

void UAdaptiveStepper::Shutdown()
{
    for (UAdaptiveNPCPolicy* Policy : Policies)
    {
        if (Policy) Policy->Shutdown();
    }
    Slots.Reset();
    Policies.Reset();
    Agents.Reset();
}

UAdaptiveNPCPolicy* UAdaptiveStepper::GetPolicy(int32 Index) const
{
    return Policies.IsValidIndex(Index) ? Policies[Index].Get() : nullptr;
}

bool UAdaptiveStepper::AreAllReady() const
{
    if (Policies.Num() == 0) return false;
    for (const UAdaptiveNPCPolicy* Policy : Policies)
    {
        if (!Policy || Policy->GetSessionState() != EAdaptiveSessionState::Ready) return false;
    }
    return true;
}

void UAdaptiveStepper::Fail(const FString& Error)
{
    LastError = Error;
    UE_LOGFMT(LogScholaInferenceUtils, Warning, "UAdaptiveStepper: {0}", Error);
}
