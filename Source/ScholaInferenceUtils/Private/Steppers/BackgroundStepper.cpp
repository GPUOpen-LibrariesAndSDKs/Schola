// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "Steppers/BackgroundStepper.h"
#include "Engine/World.h"
#include "ProfilingDebugging/CpuProfilerTrace.h"
#include "Misc/ScopeExit.h"
#include "UObject/GCObjectScopeGuard.h"

bool UBackgroundStepper::Init(UNPCInferencePolicy* InPolicy, const TArray<TScriptInterface<IAgent>>& Agents,
    const TArray<float>& DecisionHz)
{
    check(IsInGameThread());
    if (bStepping) { bResetRequested = true; return false; }
    Reset();
    if (!IsValid(InPolicy) || Agents.Num() != DecisionHz.Num()) return false;
    if (!InPolicy->GetBackgroundBinding(Scheduler, Codec, Generation)) return false;
    Cohorts = InPolicy->GetBackgroundCohorts();
    if (!Cohorts) { Reset(); return false; }
    Policy = InPolicy;
    for (int32 I = 0; I < Agents.Num(); ++I)
        if (AddAgent(Agents[I], DecisionHz[I]) == INDEX_NONE) { Reset(); return false; }
    if (GetWorld()) GetWorld()->GetSubsystem<UNPCBackgroundInferenceSubsystem>()->RegisterStepper(this);
    return true;
}
int32 UBackgroundStepper::AddAgent(TScriptInterface<IAgent> Agent, float Hz)
{
    check(IsInGameThread());
    if (bStepping || !Scheduler || !Cohorts || !IsValid(Agent.GetObject()) ||
        !Agent.GetObject()->GetClass()->ImplementsInterface(UAgent::StaticClass()) || !FMath::IsFinite(Hz) || Hz <= 0 || Hz > 60 || !FMath::IsFinite(1.f / Hz)) return INDEX_NONE;
    int32 Index = INDEX_NONE;
    for (int32 I = 0; I < Slots.Num(); ++I) if (!Slots[I].Agent.GetObject()) { Index = I; break; }
    if (Index == INDEX_NONE) Index = Slots.AddDefaulted();
    auto& S = Slots[Index];
    S.Agent = Agent; S.Period = 1.f / Hz; S.SubmittedSerial = S.AppliedSerial = 0; S.bHasAction = false; S.bUrgent=false;S.bRemoving=false;
    S.Phase = Cohorts->Acquire(S.Period, PhaseLayout, HashCombineFast(uint32(Index), S.Serial));
    S.NextDue = bStarted ? Cohorts->NextDue(S.Period,S.Phase,CurrentTime) : 0;
    return Index;
}
void UBackgroundStepper::RemoveAgent(int32 Index)
{
    check(IsInGameThread());
    if (!Slots.IsValidIndex(Index) || !Slots[Index].Agent.GetObject()) return;
    auto& S = Slots[Index];
    if (bStepping)
    {
        if (!S.bRemoving) { S.bRemoving = true; DeferredRemovals.Add(Index); }
        return;
    }
    Cohorts->Release(S.Period, S.Phase);
    ++S.Serial; S.Agent = {}; S.bHasAction = false; S.bRemoving = false; S.Observation.Reset(); S.Action.Reset(); S.HeldRow.Reset();
}
void UBackgroundStepper::RequestDecisionNow(int32 Index)
{
    check(IsInGameThread());
    if (Slots.IsValidIndex(Index) && Slots[Index].Agent.GetObject() && !Slots[Index].bRemoving)
    { Slots[Index].bUrgent=true;Slots[Index].NextDue=FMath::Min(Slots[Index].NextDue,CurrentTime); }
}
void UBackgroundStepper::Reset()
{
    check(IsInGameThread());
    // Agent events may reset their driver. Keep their reference parameters alive
    // until the event returns, then abort the step and perform teardown.
    if (bStepping) { bResetRequested = true; return; }
    bResetRequested = false;
    if (GetWorld())
        if (auto* Registry = GetWorld()->GetSubsystem<UNPCBackgroundInferenceSubsystem>()) Registry->UnregisterStepper(this);
    for (const auto& Request : Pending) if (Scheduler) Scheduler->Cancel(Request.Handle);
    Pending.Empty(); FreeRequests.Empty(); Keys.Empty();
    for (int32 I = 0; I < Slots.Num(); ++I) RemoveAgent(I);
    Slots.Empty(); Due.Empty(); DeferredRemovals.Reset(); Scheduler.Reset(); Codec.Reset(); Cohorts.Reset(); Policy = nullptr;
    bStarted = false; CurrentTime = 0;
    DeferredRows = SubmittedRows = AppliedRows = DroppedRows = 0;
}
void UBackgroundStepper::BeginDestroy() { Reset(); Super::BeginDestroy(); }
void UBackgroundStepper::ApplyHeld(int32 Index)
{
    auto& S = Slots[Index];
    if (auto* RowAgent = Cast<INPCBackgroundRowAgent>(S.Agent.GetObject())) RowAgent->ApplyActionRow(S.HeldRow);
    else IAgent::Execute_Act(S.Agent.GetObject(), S.Action);
}
void UBackgroundStepper::Step(double Now)
{
    check(IsInGameThread());
    if (bStepping || !Scheduler || !FMath::IsFinite(Now) || (bStarted && Now < CurrentTime)) return;
    FGCObjectScopeGuard KeepAlive(this);
    bStepping = true;
    ON_SCOPE_EXIT
    {
        bStepping = false;
        if (bResetRequested) Reset();
        else
        {
            for (int32 Index : DeferredRemovals) RemoveAgent(Index);
            DeferredRemovals.Reset();
        }
    };
    TSharedPtr<FNPCBackgroundScheduler, ESPMode::ThreadSafe> Bound;
    TSharedPtr<const FNPCRowCodec, ESPMode::ThreadSafe> BoundCodec;
    uint32 BoundGeneration = 0;
    if (!IsValid(Policy) || !Policy->GetBackgroundBinding(Bound, BoundCodec, BoundGeneration) || Bound != Scheduler || BoundGeneration != Generation)
    { Reset(); return; }
    auto CanContinue = [&]()
    {
        if (bResetRequested) return false;
        TSharedPtr<FNPCBackgroundScheduler, ESPMode::ThreadSafe> Current;
        TSharedPtr<const FNPCRowCodec, ESPMode::ThreadSafe> CurrentCodec;
        uint32 CurrentGeneration = 0;
        if (!IsValid(Policy) || !Policy->GetBackgroundBinding(Current, CurrentCodec, CurrentGeneration) ||
            Current != Bound || CurrentGeneration != BoundGeneration)
        { bResetRequested = true; return false; }
        return true;
    };
    if (!bStarted)
    {
        for (auto& S : Slots) if(S.Agent.GetObject())
        {
            const double Periodic=Cohorts->NextDue(S.Period,S.Phase,Now);
            S.NextDue=S.bUrgent?Now:Periodic;
        }
        bStarted = true;
    }
    const double StepWallStart = FPlatformTime::Seconds();
    CurrentTime = Now;
    for (auto& S : Slots) S.bAppliedThisStep = false;
    {
        TRACE_CPUPROFILER_EVENT_SCOPE_STR("Schola.Bg.Drain");
        for (int32 P = Pending.Num() - 1; P >= 0; --P)
        {
            auto& Request = Pending[P];
            if (Scheduler->GetStatus(Request.Handle) == ENPCBackgroundStatus::Pending) continue;
            const auto Status = Scheduler->Consume(Request.Handle, Result);
            TRACE_CPUPROFILER_EVENT_SCOPE_STR("Schola.Bg.Apply");
            for (int32 R = 0; R < Request.Slots.Num(); ++R)
            {
                const int32 Index = Request.Slots[R];
                if (!Slots.IsValidIndex(Index)) { ++DroppedRows; continue; }
                auto& S = Slots[Index];
                if (Status != ENPCBackgroundStatus::Completed || uint32(Result.UserTag >> 32) != Generation ||
                    uint32(Result.UserTag) != Request.Serial || S.bRemoving || !IsValid(S.Agent.GetObject()) || S.Serial != Request.SlotSerials[R] ||
                    S.SubmittedSerial != Request.Serial || Request.Serial <= S.AppliedSerial ||
                    (MaxActionAgeMs > 0 && (Now + FPlatformTime::Seconds() - StepWallStart - Request.ObsTime) * 1000.0 > MaxActionAgeMs)) { ++DroppedRows; continue; }
                const auto AgentStatus = IAgent::Execute_GetStatus(S.Agent.GetObject());
                if (!CanContinue()) return;
                if (S.bRemoving || AgentStatus != EAgentStatus::Running) { ++DroppedRows; continue; }
                const auto Row = MakeArrayView(Result.Actions).Slice(R * Result.ActionSize, Result.ActionSize);
                bool bDecoded = true;
                if (Cast<INPCBackgroundRowAgent>(S.Agent.GetObject())) S.HeldRow = Row;
                else if (Codec) bDecoded = Codec->Decode(Row, S.Action);
                else
                {
                    const float PointIndex = Row[0];
                    bDecoded = PointIndex >= 0 && PointIndex < Request.PassthroughPoints.Num() && FMath::FloorToFloat(PointIndex) == PointIndex;
                    if (bDecoded) S.Action = Request.PassthroughPoints[int32(PointIndex)];
                }
                if (!bDecoded) { ++DroppedRows; continue; }
                S.AppliedSerial = Request.Serial; S.bHasAction = true; S.bAppliedThisStep = true;
                ApplyHeld(Index);
                if (!CanContinue()) return;
                ++AppliedRows;
            }
            Request.Slots.Reset(); Request.SlotSerials.Reset(); Request.PassthroughPoints.Reset();
            FreeRequests.Add(MoveTemp(Request));
            Pending.RemoveAtSwap(P, 1, EAllowShrinking::No);
        }
    }
    if (bReapplyHeldAction)
    {
        TRACE_CPUPROFILER_EVENT_SCOPE_STR("Schola.Bg.ReapplyHeld");
        for (int32 I = 0; I < Slots.Num(); ++I)
        {
            if (Slots[I].bRemoving || !IsValid(Slots[I].Agent.GetObject()) || !Slots[I].bHasAction || Slots[I].bAppliedThisStep) continue;
            const auto AgentStatus = IAgent::Execute_GetStatus(Slots[I].Agent.GetObject());
            if (!CanContinue()) return;
            if (!Slots[I].bRemoving && AgentStatus == EAgentStatus::Running) ApplyHeld(I);
            if (!CanContinue()) return;
        }
    }
    {
        TRACE_CPUPROFILER_EVENT_SCOPE_STR("Schola.Bg.Gather");
        Due.Reset();
        for (int32 I = 0; I < Slots.Num(); ++I)
        {
            auto& S = Slots[I];
            if (S.Agent.GetObject() && !IsValid(S.Agent.GetObject())) { RemoveAgent(I); continue; }
            if (S.bRemoving || !IsValid(S.Agent.GetObject()) || Now + 1e-6 < S.NextDue) continue;
            const auto AgentStatus = IAgent::Execute_GetStatus(S.Agent.GetObject());
            if (!CanContinue()) return;
            if (!S.bRemoving && AgentStatus == EAgentStatus::Running) Due.Add(I);
        }
        Due.Sort([this](int32 A, int32 B) { return Slots[A].NextDue < Slots[B].NextDue; });
    }
    if (Due.IsEmpty()) return;
    const int32 Count = FMath::Min(Due.Num(), FMath::Max(1, MaxRowsPerFrame));
    DeferredRows += Due.Num() - Count;
    FNPCBackgroundStepperRequest Request = FreeRequests.IsEmpty() ? FNPCBackgroundStepperRequest{} : FreeRequests.Pop(EAllowShrinking::No);
    Request.Serial = Cohorts->NextRequestSerial(); Request.ObsTime = Now;
    Request.Slots.Append(Due.GetData(), Count);
    Keys.Reset(); Keys.Reserve(Count);
    for (int32 Index : Request.Slots)
    {
        Request.SlotSerials.Add(Slots[Index].Serial);
        Keys.Add((int64(Slots[Index].Serial) << 32) | uint32(Index));
    }
    FNPCBackgroundSubmitOptions Options;
    Options.Keys = Keys; Options.UserTag = (uint64(Generation) << 32) | Request.Serial;
    Options.MaxAgeMs = MaxActionAgeMs;
    bool bValid = true;
    ENPCBackgroundStatus Status;
    {
        TRACE_CPUPROFILER_EVENT_SCOPE_STR("Schola.Bg.Submit");
        Status = Bound->SubmitWith(Count, [&](TArrayView<float> Rows)
        {
            if (!Codec) Request.PassthroughPoints.SetNum(Count);
            const int32 Width = Scheduler->GetObservationSize();
            for (int32 R = 0; R < Count; ++R)
            {
                auto& S = Slots[Request.Slots[R]];
                if (S.bRemoving || !IsValid(S.Agent.GetObject()))
                {
                    bValid = false; FMemory::Memzero(Rows.GetData(), Rows.Num() * sizeof(float)); return;
                }
                auto Row = Rows.Slice(R * Width, Width);
                if (auto* RowAgent = Cast<INPCBackgroundRowAgent>(S.Agent.GetObject())) RowAgent->WriteObservationRow(Row);
                else
                {
                    IAgent::Execute_Observe(S.Agent.GetObject(), S.Observation);
                    if (Codec) bValid &= Codec->Encode(S.Observation, Row);
                    else { Request.PassthroughPoints[R] = S.Observation; Row[0] = float(R); }
                }
                if (!CanContinue() || S.bRemoving)
                {
                    bValid = false;
                    FMemory::Memzero(Rows.GetData(), Rows.Num() * sizeof(float));
                    return;
                }
                if (!bValid) FMemory::Memzero(Row.GetData(), Row.Num() * sizeof(float));
            }
        }, Options, Request.Handle);
    }
    if (!CanContinue())
    {
        if (Status == ENPCBackgroundStatus::Pending) Bound->Cancel(Request.Handle);
        return;
    }
    if (Status != ENPCBackgroundStatus::Pending) { DeferredRows += Count; Request.Slots.Reset(); Request.SlotSerials.Reset(); Request.PassthroughPoints.Reset(); FreeRequests.Add(MoveTemp(Request)); return; }
    if (!bValid) { Scheduler->Cancel(Request.Handle); DeferredRows += Count; Request.Slots.Reset(); Request.SlotSerials.Reset(); Request.PassthroughPoints.Reset(); FreeRequests.Add(MoveTemp(Request)); return; }
    for (int32 Index : Request.Slots)
    {
        auto& S = Slots[Index]; S.SubmittedSerial = Request.Serial;
        // Preserve phase and skip missed periods; Busy agents remain due until admitted.
        if(S.bUrgent)
        {
            // An explicit wake-up is additional to the balanced periodic phase.
            S.NextDue=Cohorts->NextDue(S.Period,S.Phase,Now+2e-6);S.bUrgent=false;
        }
        else S.NextDue += (FMath::FloorToDouble(FMath::Max(0.0, Now - S.NextDue) / S.Period) + 1.0) * S.Period;
    }
    SubmittedRows += Count;
    Pending.Add(MoveTemp(Request));
}
