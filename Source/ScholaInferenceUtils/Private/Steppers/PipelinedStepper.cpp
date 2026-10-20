// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Steppers/PipelinedStepper.h"
#include "Async/Async.h"
#include "LogScholaInferenceUtils.h"

void UPipelinedStepper::Step()
{
    if (!Policy || Agents.Num() == 0)
    {
        UE_LOGFMT(LogScholaInferenceUtils, Error, "PipelinedStepper::Step(): Invalid state - missing policy or agents");
        return;
    }

    // Apply only the newest completed result. Completions may arrive out of order, so an older
    // result that completes after a newer one is discarded instead of overwriting it.
    int32 NewestReady = INDEX_NONE;
    for (int32 Index = 0; Index < PIPELINE_STAGES; ++Index)
    {
        const FPipelinedStepperFrame& Candidate = Frames[Index];
        if (Candidate.bActionsReady && Candidate.DebugDispatchId > LastAppliedDispatchId
            && (NewestReady == INDEX_NONE || Candidate.DebugDispatchId > Frames[NewestReady].DebugDispatchId))
        {
            NewestReady = Index;
        }
    }

    if (NewestReady != INDEX_NONE)
    {
        auto& Frame = Frames[NewestReady];
        UE_LOGFMT(LogScholaInferenceUtils, Verbose, "PipelinedStepper::Step(): Applying actions - DispatchId={0} TickCounter={1}",
            Frame.DebugDispatchId, static_cast<uint64>(TickCounter));

        if (Frame.Actions.Num() != Agents.Num())
        {
            UE_LOGFMT(LogScholaInferenceUtils, Error, "PipelinedStepper::Step(): Action count mismatch - {0} actions for {1} agents", Frame.Actions.Num(), Agents.Num());
        }
        else
        {
            for (int i = 0; i < Agents.Num(); ++i)
            {
                IAgent::Execute_Act(Agents[i].GetObject(),Frame.Actions[i]);
            }
        }
        LastAppliedDispatchId = Frame.DebugDispatchId;
    }

    for (auto& Frame : Frames)
    {
        if (Frame.bActionsReady && Frame.DebugDispatchId <= LastAppliedDispatchId)
        {
            Frame.Actions.Reset();
            Frame.bActionsReady = false;
        }
    }

    const int32 CurrentFrame = TickCounter % PIPELINE_STAGES;
    auto& Frame = Frames[CurrentFrame];

    // Never reuse a slot whose inference is still running: its completion would overwrite this
    // frame's state. Skip dispatch this tick and retry once the slot completes.
    if (Frame.bThinkInFlight || Frame.bActionsReady)
    {
        return;
    }

    if (Policy->IsInferenceBusy())
    {
        return;
    }

    Frame.Observations.Reset();
    Frame.Actions.Reset();
    Frame.Observations.Reserve(Agents.Num());

    for (int i = 0; i < Agents.Num(); ++i)
    {
        TInstancedStruct<FPoint> Obs;
        IAgent::Execute_Observe(Agents[i].GetObject(),Obs);
        Frame.Observations.Add(Obs);
    }

    Frame.bThinkInFlight = true;
    DispatchThink(CurrentFrame);

    ++TickCounter;
}

void UPipelinedStepper::DispatchThink(int32 FrameIndex)
{
    FPipelinedStepperFrame* FramePtr = &Frames[FrameIndex];
    TArray<TInstancedStruct<FPoint>> ObservationsCopy = FramePtr->Observations;

    TWeakObjectPtr<UPipelinedStepper> WeakThis(this);
    TScriptInterface<IPolicy> PolicyLocal = Policy;

    // Reserve a unique dispatch id on the Game Thread
    const uint64 DispatchId = DebugDispatchSeq.fetch_add(1, std::memory_order_relaxed) + 1;
    FramePtr->DebugDispatchId = DispatchId;
    const uint64 DispatchGeneration = Generation;
    UE_LOGFMT(LogScholaInferenceUtils, Verbose, "PipelinedStepper::DispatchThink(): Scheduled - DispatchId={0} FrameIndex={1} ThreadId={2}",
        DispatchId, FrameIndex, FPlatformTLS::GetCurrentThreadId());

    Async(EAsyncExecution::ThreadPool, [WeakThis, FrameIndex, Observations = MoveTemp(ObservationsCopy), PolicyLocal, DispatchId, DispatchGeneration]() {
        if (!WeakThis.IsValid() || !PolicyLocal)
            return;

        UE_LOGFMT(LogScholaInferenceUtils, Verbose, "PipelinedStepper::DispatchThink(): Think start - DispatchId={0} FrameIndex={1} ThreadId={2}",
            DispatchId, FrameIndex, FPlatformTLS::GetCurrentThreadId());

        TArray<TInstancedStruct<FPoint>> ActionsLocal;
        const bool bSuccess = PolicyLocal->BatchedThink(const_cast<TArray<TInstancedStruct<FPoint>>&>(Observations), ActionsLocal);

        AsyncTask(ENamedThreads::GameThread, [WeakThis, FrameIndex, bSuccess, Actions = MoveTemp(ActionsLocal), DispatchId, DispatchGeneration]() mutable {
            if (!WeakThis.IsValid())
                return;
            if (WeakThis->bShuttingDown)
                return;

            auto& Frame = WeakThis->Frames[FrameIndex];
            // The slot may have been reinitialized (Init) since dispatch; it no longer belongs to this result.
            if (WeakThis->Generation != DispatchGeneration || Frame.DebugDispatchId != DispatchId)
            {
                UE_LOGFMT(LogScholaInferenceUtils, Verbose, "PipelinedStepper::DispatchThink(): Discarding stale completion - DispatchId={0} FrameIndex={1}",
                    DispatchId, FrameIndex);
                return;
            }
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
