// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "ScholaMassInferenceProcessors.h"
#include "ScholaMassInferenceSubsystem.h"
#include "MassExecutionContext.h"
#include "MassEntityManager.h"
#include "MassEntityUtils.h"
#include "MassSignalSubsystem.h"
#include "Misc/ScopeLock.h"
#include "ProfilingDebugging/CpuProfilerTrace.h"

UScholaMassGatherProcessor::UScholaMassGatherProcessor() : Query(*this)
{
    bRequiresGameThreadExecution = false;
    ExecutionOrder.ExecuteInGroup = TEXT("ScholaInference.Gather");
    ExecutionOrder.ExecuteAfter.Add(TEXT("ScholaInference.Apply"));
}
void UScholaMassGatherProcessor::ConfigureQueries(const TSharedRef<FMassEntityManager>&)
{
    Query.AddRequirement<FScholaInferenceScheduleFragment>(EMassFragmentAccess::ReadWrite);
    Query.AddRequirement<FScholaInferenceStateFragment>(EMassFragmentAccess::ReadWrite);
    Query.AddConstSharedRequirement<FScholaInferenceBucketFragment>();
    Query.AddSubsystemRequirement<UScholaMassInferenceSubsystem>(EMassFragmentAccess::ReadWrite);
    ConfigureObservationRequirements(Query);
}
void UScholaMassGatherProcessor::Execute(FMassEntityManager& Manager, FMassExecutionContext& Context)
{
    auto* Service = Context.GetWorld()->GetSubsystem<UScholaMassInferenceSubsystem>();
    if (!Service) return;
    FScopeLock Lock(&Service->Mutex);
    auto& Bridge = Service->Bridge;
    const double Now = FPlatformTime::Seconds();
    struct FDue { FMassEntityHandle Entity; double Time; int32 Bucket; };
    TArray<FDue> AllDue;
    {
        TRACE_CPUPROFILER_EVENT_SCOPE_STR("Schola.Bg.Mass.CountDue");
        Query.ForEachEntityChunk(Context, [&](FMassExecutionContext& Chunk)
        {
            const int32 Id = Chunk.GetConstSharedFragment<FScholaInferenceBucketFragment>().BucketId;
            auto Schedules = Chunk.GetMutableFragmentView<FScholaInferenceScheduleFragment>();
            auto States = Chunk.GetMutableFragmentView<FScholaInferenceStateFragment>();
            for (int32 I = 0; I < Chunk.GetNumEntities(); ++I)
                if (Bridge.InitializeSlot(Chunk.GetEntity(I), Id, Schedules[I], States[I], Now, PhaseLayout) && Now + 1e-6 >= Schedules[I].NextDueTime)
                    AllDue.Add({Chunk.GetEntity(I), Schedules[I].NextDueTime,Id});
        });
    }
    // The frame budget is shared across buckets: select globally oldest rows
    // before regrouping compatible models into their separate submissions.
    AllDue.Sort([](const FDue& A,const FDue& B){return A.Time<B.Time;});
    const int32 AdmittedCount=FMath::Min(FMath::Max(1,MaxRowsPerFrame),AllDue.Num());
    Bridge.DeferredRows+=AllDue.Num()-AdmittedCount;
    TMap<int32,TArray<FDue>> Buckets;
    for(int32 I=0;I<AdmittedCount;++I)Buckets.FindOrAdd(AllDue[I].Bucket).Add(AllDue[I]);
    for (auto& Pair : Buckets)
    {
        auto Scheduler = Bridge.GetScheduler(Pair.Key);
        if (!Scheduler) continue;
        auto& Due = Pair.Value;
        const int32 Count=Due.Num();
        TArray<FMassEntityHandle> Entities;
        // Chunk-prefix offsets preserve contiguous writes. Admission ordering selected the oldest subset above.
        TSet<FMassEntityHandle> Admitted;
        for (int32 I = 0; I < Count; ++I) Admitted.Add(Due[I].Entity);
        struct FChunkRows { TArray<int32> Indices; int32 Offset = 0; };
        TMap<FMassEntityHandle, FChunkRows> Chunks;
        Query.ForEachEntityChunk(Context, [&](FMassExecutionContext& Chunk)
        {
            if (Chunk.GetConstSharedFragment<FScholaInferenceBucketFragment>().BucketId != Pair.Key) return;
            FChunkRows Block; Block.Offset = Entities.Num();
            for (int32 I = 0; I < Chunk.GetNumEntities(); ++I)
                if (Admitted.Contains(Chunk.GetEntity(I))) { Block.Indices.Add(I); Entities.Add(Chunk.GetEntity(I)); }
            if (!Block.Indices.IsEmpty()) Chunks.Add(Chunk.GetEntity(0), MoveTemp(Block));
        });
        TRACE_CPUPROFILER_EVENT_SCOPE_STR("Schola.Bg.Mass.Submit");
        Bridge.Submit(Pair.Key, Manager, Entities, [&](TArrayView<float> Rows)
        {
            auto FillChunk = [&](FMassExecutionContext& Chunk)
            {
                const auto* Block = Chunks.Find(Chunk.GetEntity(0));
                if (!Block) return;
                WriteObservations(Chunk, Block->Indices,
                    Rows.Slice(Block->Offset * Scheduler->GetObservationSize(), Block->Indices.Num() * Scheduler->GetObservationSize()));
            };
            if (Entities.Num() >= ParallelGatherMinEntities)
                Query.ParallelForEachEntityChunk(Context, FillChunk, FMassEntityQuery::EParallelExecutionFlags::Force);
            else Query.ForEachEntityChunk(Context, FillChunk);
        }, Now, MaxActionAgeMs);
    }
}

UScholaMassApplyProcessor::UScholaMassApplyProcessor() : Query(*this)
{
    bRequiresGameThreadExecution = false;
    ExecutionOrder.ExecuteInGroup = TEXT("ScholaInference.Apply");
}
void UScholaMassApplyProcessor::ConfigureQueries(const TSharedRef<FMassEntityManager>&)
{
    Query.AddRequirement<FScholaInferenceStateFragment>(EMassFragmentAccess::ReadWrite);
    Query.AddConstSharedRequirement<FScholaInferenceBucketFragment>();
    Query.AddSubsystemRequirement<UScholaMassInferenceSubsystem>(EMassFragmentAccess::ReadWrite);
    Query.AddSubsystemRequirement<UMassSignalSubsystem>(EMassFragmentAccess::ReadWrite);
    ConfigureActionRequirements(Query);
}
void UScholaMassApplyProcessor::Execute(FMassEntityManager& Manager, FMassExecutionContext& Context)
{
    auto* Service = Context.GetWorld()->GetSubsystem<UScholaMassInferenceSubsystem>();
    if (!Service) return;
    FScopeLock Lock(&Service->Mutex);
    TRACE_CPUPROFILER_EVENT_SCOPE_STR("Schola.Bg.Mass.Apply");
    TArray<FMassEntityHandle> Ready;
    Service->Bridge.Drain(Manager, FPlatformTime::Seconds(), MaxActionAgeMs, Ready);
    TArray<FMassArchetypeEntityCollection> Collections;
    UE::Mass::Utils::CreateEntityCollections(Manager, Ready, FMassArchetypeEntityCollection::NoDuplicates, Collections);
    for (const auto& Collection : Collections)
        Query.ForEachEntityChunk(Collection, Context, [&](FMassExecutionContext& Chunk)
        {
            TArray<int32> Indices; TArray<float> Rows;
            const auto States = Chunk.GetFragmentView<FScholaInferenceStateFragment>();
            for (int32 I = 0; I < Chunk.GetNumEntities(); ++I)
            {
                const auto Row = Service->Bridge.GetHeldAction(States[I]);
                if (!Row.IsEmpty() && (MaxActionAgeMs<=0 || (FPlatformTime::Seconds()-Service->Bridge.GetHeldObservationTime(States[I]))*1000.0<=MaxActionAgeMs))
                { Indices.Add(I); Rows.Append(Row.GetData(), Row.Num()); }
                else Service->Bridge.DiscardReady(States[I]);
            }
            if (!Indices.IsEmpty())
            {
                ApplyActions(Chunk, Indices, Rows);
                for (int32 I : Indices) Service->Bridge.MarkDelivered(States[I]);
            }
            if (!ActionReadySignal.IsNone() && !Indices.IsEmpty())
            {
                TArray<FMassEntityHandle> Signals;
                for (int32 I : Indices) Signals.Add(Chunk.GetEntity(I));
                Chunk.GetMutableSubsystemChecked<UMassSignalSubsystem>().SignalEntitiesDeferred(Chunk, ActionReadySignal, Signals);
            }
        });
}
UScholaMassScheduleObserver::UScholaMassScheduleObserver() : Query(*this)
{
    ObservedTypes.Add(FScholaInferenceStateFragment::StaticStruct());
    ObservedOperations = EMassObservedOperationFlags::All;
    bRequiresGameThreadExecution = false;
}
void UScholaMassScheduleObserver::ConfigureQueries(const TSharedRef<FMassEntityManager>&)
{
    Query.AddRequirement<FScholaInferenceStateFragment>(EMassFragmentAccess::ReadWrite);
    Query.AddRequirement<FScholaInferenceScheduleFragment>(EMassFragmentAccess::ReadWrite);
    Query.AddConstSharedRequirement<FScholaInferenceBucketFragment>();
    Query.AddSubsystemRequirement<UScholaMassInferenceSubsystem>(EMassFragmentAccess::ReadWrite);
}
void UScholaMassScheduleObserver::Execute(FMassEntityManager&, FMassExecutionContext& Context)
{
    auto* Service = Context.GetWorld() ? Context.GetWorld()->GetSubsystem<UScholaMassInferenceSubsystem>() : nullptr;
    if (!Service) return; // Standalone managers initialize through Bridge explicitly.
    FScopeLock Lock(&Service->Mutex);
    const auto Operation=Context.GetAuxData().Get<FMassObserverExecutionContext>().GetOperationType();
    const bool bRemove=Operation==EMassObservedOperation::RemoveElement || Operation==EMassObservedOperation::DestroyEntity;
    Query.ForEachEntityChunk(Context, [&](FMassExecutionContext& Chunk)
    {
        auto States = Chunk.GetMutableFragmentView<FScholaInferenceStateFragment>();
        auto Schedules = Chunk.GetMutableFragmentView<FScholaInferenceScheduleFragment>();
        const int32 Id = Chunk.GetConstSharedFragment<FScholaInferenceBucketFragment>().BucketId;
        for (int32 I = 0; I < Chunk.GetNumEntities(); ++I)
            if (bRemove) Service->Bridge.ReleaseSlot(States[I]);
            else Service->Bridge.InitializeSlot(Chunk.GetEntity(I), Id, Schedules[I], States[I], FPlatformTime::Seconds());
    });
}

UScholaMassDecideNowProcessor::UScholaMassDecideNowProcessor()
{
    bRequiresGameThreadExecution = false;
    ExecutionOrder.ExecuteBefore.Add(TEXT("ScholaInference.Gather"));
}
void UScholaMassDecideNowProcessor::InitializeInternal(UObject& Owner, const TSharedRef<FMassEntityManager>& Manager)
{
    Super::InitializeInternal(Owner,Manager);
    if (auto* World = Owner.GetWorld())
        if (auto* Signals = World->GetSubsystem<UMassSignalSubsystem>()) SubscribeToSignal(*Signals,TEXT("ScholaDecideNow"));
}
void UScholaMassDecideNowProcessor::ConfigureQueries(const TSharedRef<FMassEntityManager>&)
{
    EntityQuery.AddRequirement<FScholaInferenceScheduleFragment>(EMassFragmentAccess::ReadWrite);
    EntityQuery.AddRequirement<FScholaInferenceStateFragment>(EMassFragmentAccess::ReadWrite);
    EntityQuery.AddConstSharedRequirement<FScholaInferenceBucketFragment>();
    EntityQuery.AddSubsystemRequirement<UScholaMassInferenceSubsystem>(EMassFragmentAccess::ReadWrite);
}
void UScholaMassDecideNowProcessor::SignalEntities(FMassEntityManager&, FMassExecutionContext& Context, FMassSignalNameLookup&)
{
    EntityQuery.ForEachEntityChunk(Context,[&](FMassExecutionContext& Chunk)
    {
        auto& Service = Chunk.GetMutableSubsystemChecked<UScholaMassInferenceSubsystem>();
        FScopeLock Lock(&Service.Mutex);
        const double Now = FPlatformTime::Seconds();
        auto Schedules = Chunk.GetMutableFragmentView<FScholaInferenceScheduleFragment>();
        auto States = Chunk.GetMutableFragmentView<FScholaInferenceStateFragment>();
        const int32 Id = Chunk.GetConstSharedFragment<FScholaInferenceBucketFragment>().BucketId;
        for (int32 I=0; I<Chunk.GetNumEntities(); ++I)
            if (Service.Bridge.InitializeSlot(Chunk.GetEntity(I),Id,Schedules[I],States[I],Now))
            {
                Schedules[I].bUrgent=true;
                Schedules[I].NextDueTime=FMath::Min(Schedules[I].NextDueTime,Now);
            }
    });
}
