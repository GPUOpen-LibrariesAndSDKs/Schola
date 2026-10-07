// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Steppers/PipelinedStepper.h"
#include "Async/Async.h"
#include "UObject/GarbageCollection.h"
#include "LogScholaInferenceUtils.h"

void UPipelinedStepper::Step()
{
    if (!Policy || Agents.Num() == 0)
    {
        UE_LOGFMT(LogScholaInferenceUtils, Error, "PipelinedStepper::Step(): Invalid state - missing policy or agents");
        return;
    }

    const int32 CurrentFrame = TickCounter % PIPELINE_STAGES;
    const bool bHavePrevious = TickCounter > 0;
    const int32 PrevFrame = bHavePrevious ? (TickCounter - 1) % PIPELINE_STAGES : -1;

	if (TickCounter > 0 && Frames[PrevFrame].bActionsReady)
	{
		UE_LOGFMT(LogScholaInferenceUtils, Verbose, "PipelinedStepper::Step(): PrevFrame actions ready - DispatchId={0} TickCounter={1} ThreadId={2}",
			static_cast<uint64>(Frames[PrevFrame].DebugDispatchId),
			static_cast<uint64>(TickCounter),
			FPlatformTLS::GetCurrentThreadId());

        auto& Frame = Frames[PrevFrame];

        if (Frame.Actions.Num() != Frame.DispatchedAgents.Num())
        {
            UE_LOGFMT(LogScholaInferenceUtils, Error, "PipelinedStepper::Step(): Action count mismatch - {0} actions for {1} agents", Frame.Actions.Num(), Frame.DispatchedAgents.Num());
        }
        else
        {
            // Agents may have been added or removed since dispatch, so only act on dispatched agents that are still managed
            const bool bMembershipChanged = Frame.DispatchedMembershipVersion != MembershipVersion;
            for (int i = 0; i < Frame.DispatchedAgents.Num(); ++i)
            {
                UObject* Agent = Frame.DispatchedAgents[i].Get();
                if (!Agent || (bMembershipChanged && FindAgentIndex(Agent) == INDEX_NONE))
                {
                    continue;
                }
                IAgent::Execute_Act(Agent, Frame.Actions[i]);
            }
        }

        Frame.Actions.Reset();
        Frame.DispatchedAgents.Reset();
        Frame.bActionsReady = false;
    }
    
    auto& Frame = Frames[CurrentFrame];
    Frame.Observations.Reset();
    Frame.Actions.Reset();
    Frame.bActionsReady = false;
    Frame.bThinkInFlight = true;

    Frame.Observations.Reserve(Agents.Num());

    for (int i = 0; i < Agents.Num(); ++i)
    {
        TInstancedStruct<FPoint> Obs;
        IAgent::Execute_Observe(Agents[i].GetObject(),Obs);
        Frame.Observations.Add(Obs);
    }
    
    if (Policy->IsInferenceBusy() || bDispatchInFlight)
    {
		return;
    }
    DispatchThink(CurrentFrame);
    
    ++TickCounter;
}

void UPipelinedStepper::DispatchThink(int32 FrameIndex)
{
    FPipelinedStepperFrame* FramePtr = &Frames[FrameIndex];
    TArray<TInstancedStruct<FPoint>> ObservationsCopy = FramePtr->Observations;
    TArray<TScriptInterface<IPolicyState>> StatesCopy = CurrentStates;
    bDispatchInFlight = true;

    FramePtr->DispatchedAgents.Reset(Agents.Num());
    for (const TScriptInterface<IAgent>& Agent : Agents)
    {
        FramePtr->DispatchedAgents.Add(Agent.GetObject());
    }
    FramePtr->DispatchedMembershipVersion = MembershipVersion;

    TWeakObjectPtr<UPipelinedStepper> WeakThis(this);
    TScriptInterface<IPolicy> PolicyLocal = Policy;

    // Reserve a unique dispatch id on the Game Thread
    const uint64 DispatchId = DebugDispatchSeq.fetch_add(1, std::memory_order_relaxed) + 1;
    FramePtr->DebugDispatchId = DispatchId;
    UE_LOGFMT(LogScholaInferenceUtils, Verbose, "PipelinedStepper::DispatchThink(): Scheduled - DispatchId={0} FrameIndex={1} ThreadId={2}",
        DispatchId, FrameIndex, FPlatformTLS::GetCurrentThreadId());

    Async(EAsyncExecution::ThreadPool, [WeakThis, FrameIndex, Observations = MoveTemp(ObservationsCopy), States = MoveTemp(StatesCopy), PolicyLocal, DispatchId]() mutable {
        // The policy and state objects are only kept alive by the stepper's UPROPERTYs, so block GC while
        // this thread uses them, and check the stepper is still alive only once GC can no longer run.
        FGCScopeGuard GCGuard;
        if (!WeakThis.IsValid() || WeakThis->bShuttingDown || !PolicyLocal)
            return;

        UE_LOGFMT(LogScholaInferenceUtils, Verbose, "PipelinedStepper::DispatchThink(): Think start - DispatchId={0} FrameIndex={1} ThreadId={2}",
            DispatchId, FrameIndex, FPlatformTLS::GetCurrentThreadId());

        TArray<TInstancedStruct<FPoint>> ActionsLocal;
        const bool bSuccess = PolicyLocal->BatchedThink(Observations, States, ActionsLocal);

        AsyncTask(ENamedThreads::GameThread, [WeakThis, FrameIndex, bSuccess, Actions = MoveTemp(ActionsLocal), DispatchId]() mutable {
            if (!WeakThis.IsValid())
                return;
            if (WeakThis->bShuttingDown)
                return;

            WeakThis->CompleteThink();

            auto& Frame = WeakThis->Frames[FrameIndex];
            if (!bSuccess)
            {
                UE_LOGFMT(LogScholaInferenceUtils, Error, "PipelinedStepper::DispatchThink(): Think failed - DispatchId={0} FrameIndex={1}",
                    DispatchId, FrameIndex);
                Frame.bThinkInFlight = false;
                return;
            }
            Frame.Actions = MoveTemp(Actions);
            Frame.bActionsReady = true;
            Frame.bThinkInFlight = false;

            UE_LOGFMT(LogScholaInferenceUtils, Verbose, "PipelinedStepper::DispatchThink(): Think complete - DispatchId={0} FrameIndex={1} ThreadId={2}",
                DispatchId, FrameIndex, FPlatformTLS::GetCurrentThreadId());
        });
    });
}

void UPipelinedStepper::CompleteThink()
{
    bDispatchInFlight = false;

    for (const TScriptInterface<IPolicyState>& State : PendingStateResets)
    {
        State->Reset();
    }
    PendingStateResets.Reset();
    RetiredStates.Reset();
}

bool UPipelinedStepper::AddAgent(const TScriptInterface<IAgent>& InAgent)
{
    if (!Policy)
    {
        UE_LOGFMT(LogScholaInferenceUtils, Error, "PipelinedStepper::AddAgent(): No policy, call Init first");
        return false;
    }
    if (!InAgent.GetObject())
    {
        UE_LOGFMT(LogScholaInferenceUtils, Error, "PipelinedStepper::AddAgent(): Agent is null");
        return false;
    }
    if (FindAgentIndex(InAgent.GetObject()) != INDEX_NONE)
    {
        UE_LOGFMT(LogScholaInferenceUtils, Error, "PipelinedStepper::AddAgent(): Agent {0} is already managed by this stepper", InAgent.GetObject()->GetName());
        return false;
    }

    TScriptInterface<IPolicyState> State;
    if (!CreateAgentState(*Policy, State))
    {
        UE_LOGFMT(LogScholaInferenceUtils, Error, "PipelinedStepper::AddAgent(): Policy failed to create the initial state for agent {0}", InAgent.GetObject()->GetName());
        return false;
    }

    Agents.Add(InAgent);
    CurrentStates.Add(State);
    ++MembershipVersion;
    return true;
}

bool UPipelinedStepper::RemoveAgent(const TScriptInterface<IAgent>& InAgent)
{
    UObject* RemovedAgent = InAgent.GetObject();
    const int32 AgentIndex = FindAgentIndex(RemovedAgent);
    if (AgentIndex == INDEX_NONE)
    {
        UE_LOGFMT(LogScholaInferenceUtils, Error, "PipelinedStepper::RemoveAgent(): Agent is not managed by this stepper");
        return false;
    }

    // Preserve frame indexing but prevent any action dispatched for this membership
    // from reaching the same UObject if it is re-added before the action is applied.
    for (FPipelinedStepperFrame& Frame : Frames)
    {
        for (TWeakObjectPtr<UObject>& DispatchedAgent : Frame.DispatchedAgents)
        {
            if (DispatchedAgent.Get() == RemovedAgent)
            {
                DispatchedAgent.Reset();
            }
        }
    }

    if (bDispatchInFlight && CurrentStates[AgentIndex])
    {
        RetiredStates.Add(CurrentStates[AgentIndex]);
    }
    Agents.RemoveAt(AgentIndex);
    CurrentStates.RemoveAt(AgentIndex);
    ++MembershipVersion;
    return true;
}

int32 UPipelinedStepper::FindAgentIndex(const UObject* InAgent) const
{
    if (!InAgent)
    {
        return INDEX_NONE;
    }
    return Agents.IndexOfByPredicate([InAgent](const TScriptInterface<IAgent>& Agent) { return Agent.GetObject() == InAgent; });
}

void UPipelinedStepper::ResetStates()
{
    for (int32 i = 0; i < CurrentStates.Num(); ++i)
    {
        ResetAgentState(i);
    }
}

void UPipelinedStepper::ResetAgentState(int32 AgentIndex)
{
    if (!CurrentStates.IsValidIndex(AgentIndex))
    {
        UE_LOGFMT(LogScholaInferenceUtils, Error, "PipelinedStepper::ResetAgentState(): Invalid agent index {0}", AgentIndex);
        return;
    }
    const TScriptInterface<IPolicyState>& State = CurrentStates[AgentIndex];
    if (!State)
    {
        return;
    }
    if (bDispatchInFlight)
    {
        PendingStateResets.AddUnique(State);
        return;
    }
    State->Reset();
}
