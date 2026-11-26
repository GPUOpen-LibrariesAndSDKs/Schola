// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "CrowdProcessors.h"
#include "CampusMigrationDay.h"
#include "CampusScene.h"
#include "MassExecutionContext.h"
#include "NNERuntimeCPU.h"
#include "ProfilingDebugging/CpuProfilerTrace.h"

uint32 CrowdSim::NextRandom(uint32& State)
{
    State ^= State << 13; State ^= State >> 17; State ^= State << 5;
    return State;
}

FVector2f CrowdSim::RandomPoint(uint32& State, float HalfSize)
{
    const float X = NextRandom(State) / float(MAX_uint32), Y = NextRandom(State) / float(MAX_uint32);
    return FVector2f((X * 2 - 1) * HalfSize * .95f, (Y * 2 - 1) * HalfSize * .95f);
}

void CrowdSim::WriteObservation(const FCrowdAgentFragment& Agent, float HalfSize, TArrayView<float> Row)
{
    check(Row.Num() == ObservationSize);
    const FVector2f ToGoal = Agent.Goal - Agent.Position;
    const FVector2f Direction = ToGoal.GetSafeNormal();
    Row[0] = Direction.X;
    Row[1] = Direction.Y;
    Row[2] = FMath::Min(ToGoal.Size() / 1000.f, 4.f);
    Row[3] = Agent.Velocity.X / Speed;
    Row[4] = Agent.Velocity.Y / Speed;
    // Smooth positional features: the policy's wander term varies across the arena.
    const FVector2f P = Agent.Position / FMath::Max(HalfSize, 1.f);
    for (int32 I = 5; I < ObservationSize; ++I)
    {
        const float Frequency = 1.f + (I - 5) / 2 * .5f;
        Row[I] = (I & 1) ? FMath::Sin(P.X * Frequency + P.Y) : FMath::Cos(P.Y * Frequency - P.X);
    }
}

void CrowdSim::ApplyAction(FCrowdAgentFragment& Agent, TConstArrayView<float> Row, FCrowdFrameCounters& Counters)
{
    const FVector2f Action(Row[0], Row[1]);
    if (Action.SizeSquared() > 1e-6f) Agent.Steering = Action.GetSafeNormal();
    ++Counters.Decisions;
    Counters.LatencySumMs += (FPlatformTime::Seconds() - Agent.ObservationTime) * 1000.0;
}

UCrowdGatherProcessor::UCrowdGatherProcessor() { bAutoRegisterWithProcessingPhases = false; }

void UCrowdGatherProcessor::ConfigureObservationRequirements(FMassEntityQuery& InQuery)
{
    InQuery.AddRequirement<FCrowdAgentFragment>(EMassFragmentAccess::ReadWrite);
}

void UCrowdGatherProcessor::WriteObservations(FMassExecutionContext& Context, TConstArrayView<int32> DueIndices, TArrayView<float> Rows)
{
    // May run on several worker threads at once; only this chunk's rows and fragments are touched.
    auto Agents = Context.GetMutableFragmentView<FCrowdAgentFragment>();
    const double Now = FPlatformTime::Seconds();
    for (int32 R = 0; R < DueIndices.Num(); ++R)
    {
        auto& Agent = Agents[DueIndices[R]];
        Agent.ObservationTime = Now;
        const int32 Size = MigrationDay ? MigrationDay->GetObservationSize() : CrowdSim::ObservationSize;
        auto Row = Rows.Slice(R * Size, Size);
        if (MigrationDay) MigrationDay->WriteObservation(Agent.Instance, Row);
        else CrowdSim::WriteObservation(Agent, HalfSize, Row);
    }
}

UCrowdApplyProcessor::UCrowdApplyProcessor() { bAutoRegisterWithProcessingPhases = false; }

void UCrowdApplyProcessor::ConfigureActionRequirements(FMassEntityQuery& InQuery)
{
    InQuery.AddRequirement<FCrowdAgentFragment>(EMassFragmentAccess::ReadWrite);
}

void UCrowdApplyProcessor::ApplyActions(FMassExecutionContext& Context, TConstArrayView<int32> EntityIndices, TConstArrayView<float> Rows)
{
    auto Agents = Context.GetMutableFragmentView<FCrowdAgentFragment>();
    for (int32 R = 0; R < EntityIndices.Num(); ++R)
    {
        auto& Agent = Agents[EntityIndices[R]];
        const int32 Size = MigrationDay ? MigrationDay->GetActionSize() : CrowdSim::ActionSize;
        const auto Row = Rows.Slice(R * Size, Size);
        if (MigrationDay) MigrationDay->ApplyAction(Agent.Instance, Row);
        else CrowdSim::ApplyAction(Agent, Row, *Counters);
    }
}

UCrowdSyncInferenceProcessor::UCrowdSyncInferenceProcessor() : Query(*this)
{
    bAutoRegisterWithProcessingPhases = false;
    bRequiresGameThreadExecution = true;
}

void UCrowdSyncInferenceProcessor::ConfigureQueries(const TSharedRef<FMassEntityManager>&)
{
    Query.AddRequirement<FCrowdAgentFragment>(EMassFragmentAccess::ReadWrite);
}

void UCrowdSyncInferenceProcessor::Execute(FMassEntityManager&, FMassExecutionContext& Context)
{
    if (!Instance || bFailed) return;
    TRACE_CPUPROFILER_EVENT_SCOPE_STR("Crowd.SyncInference");
    float Observation[FMath::Max(CrowdSim::ObservationSize, CampusTurn::ObservationSize)];
    float Action[FMath::Max(CrowdSim::ActionSize, CampusTurn::ExitCount)];
    const int32 Size = MigrationDay ? MigrationDay->GetObservationSize() : CrowdSim::ObservationSize;
    auto ObservationRow = MakeArrayView(Observation, Size);
    const UE::NNE::FTensorBindingCPU Input{Observation, sizeof(float) * Size};
    const int32 ActionSize = MigrationDay ? MigrationDay->GetActionSize() : CrowdSim::ActionSize;
    const auto ActionRow = MakeArrayView(Action, ActionSize);
    const UE::NNE::FTensorBindingCPU Output{Action, sizeof(float) * ActionSize};
    const bool bTurns = MigrationDay && MigrationDay->GetTurns();
    Query.ForEachEntityChunk(Context, [&](FMassExecutionContext& Chunk)
    {
        auto Agents = Chunk.GetMutableFragmentView<FCrowdAgentFragment>();
        for (auto& Agent : Agents)
        {
            const double Now = FPlatformTime::Seconds();
            const double ScheduleTime = bUseSimulationClock ? MigrationDay->GetTime() : Now;
            if (bFailed || (!bTurns && ScheduleTime + 1e-6 < Agent.SyncNextDue)) continue;
            if (MigrationDay && !MigrationDay->WantsDecision(Agent.Instance)) continue;
            Agent.ObservationTime = Now;
            if (MigrationDay) MigrationDay->WriteObservation(Agent.Instance, ObservationRow);
            else CrowdSim::WriteObservation(Agent, HalfSize, Observation);
            if (Instance->RunSync(MakeArrayView(&Input, 1), MakeArrayView(&Output, 1)) != UE::NNE::EResultStatus::Ok)
            {
                bFailed = true;
                return;
            }
            ++ModelCalls;
            if (MigrationDay) MigrationDay->ApplyAction(Agent.Instance, ActionRow);
            else CrowdSim::ApplyAction(Agent, ActionRow, *Counters);
            // Skip missed periods rather than queueing a backlog of stale decisions.
            if (!bTurns) Agent.SyncNextDue += (FMath::FloorToDouble((ScheduleTime + 1e-6 - Agent.SyncNextDue) / PeriodSeconds) + 1) * PeriodSeconds;
        }
    });
}

UCrowdMoveProcessor::UCrowdMoveProcessor() : Query(*this)
{
    bAutoRegisterWithProcessingPhases = false;
}

void UCrowdMoveProcessor::ConfigureQueries(const TSharedRef<FMassEntityManager>&)
{
    Query.AddRequirement<FCrowdAgentFragment>(EMassFragmentAccess::ReadWrite);
}

void UCrowdMoveProcessor::Execute(FMassEntityManager&, FMassExecutionContext& Context)
{
    TRACE_CPUPROFILER_EVENT_SCOPE_STR("Crowd.Move");
    if (!Campus || Campus->Nodes.IsEmpty() || !Transforms) return;
    const float DeltaTime = FMath::Min(Context.GetDeltaTimeSeconds(), .1f);
    const float Blend = 1.f - FMath::Exp(-3.f * DeltaTime);
    constexpr float CellSize = 150.f, SeparationRadius = 100.f;
    static_assert(SeparationRadius <= CellSize, "Separation only searches neighbouring cells");

    // Snapshot positions by instance so every agent reads last frame's neighbours.
    const int32 Count = Transforms->Num();
    Positions.SetNumUninitialized(Count, EAllowShrinking::No);
    Query.ParallelForEachEntityChunk(Context, [this](FMassExecutionContext& Chunk)
    {
        for (const auto& Agent : Chunk.GetFragmentView<FCrowdAgentFragment>()) Positions[Agent.Instance] = Agent.Position;
    }, FMassEntityQuery::EParallelExecutionFlags::Force);

    // Dense grid over the campus, filled by counting sort; positions outside it clamp to the border cells.
    const float Half = FMath::Max(HalfSize, CellSize);
    const int32 Side = FMath::CeilToInt32(2 * Half / CellSize) + 1;
    const auto CellXY = [Half, Side](FVector2f P)
    {
        return FIntPoint(FMath::Clamp(FMath::FloorToInt32((P.X + Half) / CellSize), 0, Side - 1),
            FMath::Clamp(FMath::FloorToInt32((P.Y + Half) / CellSize), 0, Side - 1));
    };
    {
        TRACE_CPUPROFILER_EVENT_SCOPE_STR("Crowd.Move.Grid");
        CellOf.SetNumUninitialized(Count, EAllowShrinking::No);
        CellStart.SetNumUninitialized(Side * Side + 1, EAllowShrinking::No);
        FMemory::Memzero(CellStart.GetData(), CellStart.Num() * sizeof(int32));
        Sorted.SetNumUninitialized(Count, EAllowShrinking::No);
        for (int32 I = 0; I < Count; ++I)
        {
            const FIntPoint C = CellXY(Positions[I]);
            CellOf[I] = C.Y * Side + C.X;
            ++CellStart[CellOf[I] + 1];
        }
        for (int32 C = 1; C < CellStart.Num(); ++C) CellStart[C] += CellStart[C - 1];
        CellFill = CellStart;
        for (int32 I = 0; I < Count; ++I) Sorted[CellFill[CellOf[I]]++] = I;
    }

    Query.ParallelForEachEntityChunk(Context, [&](FMassExecutionContext& Chunk)
    {
        // Each agent writes only its own fragment and transform; neighbours come from the snapshot.
        for (auto& Agent : Chunk.GetMutableFragmentView<FCrowdAgentFragment>())
        {
            if (Agent.WaitSeconds > 0)
            {
                Agent.WaitSeconds -= DeltaTime;
                Agent.Velocity *= .8f;
            }
            else
            {
                if (FVector2f::DistSquared(Agent.Position, Agent.Goal) < FMath::Square(110.f))
                {
                    Agent.PreviousNode = Agent.CurrentNode;
                    Agent.CurrentNode = Agent.NextNode;
                    const auto& Links = Campus->Nodes[Agent.CurrentNode].Links;
                    int32 Choice = CrowdSim::NextRandom(Agent.Random) % Links.Num();
                    if (Links.Num() > 1 && Links[Choice] == Agent.PreviousNode) Choice = (Choice + 1) % Links.Num();
                    Agent.NextNode = Links[Choice];
                    Agent.Goal = Campus->Nodes[Agent.NextNode].Position;
                    if (CrowdSim::NextRandom(Agent.Random) % 12 == 0) Agent.WaitSeconds = 1.f + CrowdSim::NextRandom(Agent.Random) % 4;
                }
                const FVector2f Start = Campus->Nodes[Agent.CurrentNode].Position;
                const FVector2f End = Campus->Nodes[Agent.NextNode].Position;
                const FVector2f Along = (End - Start).GetSafeNormal();
                const FVector2f Side2(-Along.Y, Along.X);
                const FVector2f LaneTarget = End + Side2 * Agent.Lane;
                const FVector2f Desired = (LaneTarget - Agent.Position).GetSafeNormal();
                // ONNX steering is bounded by the route corridor, which keeps students outside CAD footprints.
                FVector2f Separation = FVector2f::ZeroVector;
                const FIntPoint Here = CellXY(Agent.Position);
                for (int32 CY = FMath::Max(Here.Y - 1, 0); CY <= FMath::Min(Here.Y + 1, Side - 1); ++CY)
                for (int32 CX = FMath::Max(Here.X - 1, 0); CX <= FMath::Min(Here.X + 1, Side - 1); ++CX)
                {
                    const int32 C = CY * Side + CX;
                    for (int32 S = CellStart[C]; S < CellStart[C + 1]; ++S)
                    {
                        const int32 Other = Sorted[S];
                        if (Other == Agent.Instance) continue;
                        const FVector2f Away = Agent.Position - Positions[Other];
                        const float Distance = Away.Size();
                        if (Distance > .1f && Distance < SeparationRadius)
                            Separation += Away / Distance * (1.f - Distance / SeparationRadius);
                    }
                }
                Separation = Separation.GetClampedToMaxSize(1.f);
                const FVector2f Direction = (Desired * .85f + Agent.Steering * .15f + Separation * .9f).GetSafeNormal();
                Agent.Velocity += (Direction * Agent.WalkSpeed - Agent.Velocity) * Blend;
                Agent.Position += Agent.Velocity * DeltaTime;
                const float CrossTrack = FVector2f::DotProduct(Agent.Position - Start, Side2);
                Agent.Position -= Side2 * (CrossTrack - FMath::Clamp(CrossTrack, -160.f, 160.f));
            }
            const float Yaw = FMath::RadiansToDegrees(FMath::Atan2(Agent.Velocity.Y, Agent.Velocity.X));
            (*Transforms)[Agent.Instance] = FTransform(FRotator(0, Yaw, 0),
                FVector(Agent.Position.X, Agent.Position.Y, 94), FVector(.6f, .6f, 1.8f));
        }
    }, FMassEntityQuery::EParallelExecutionFlags::Force);
}
