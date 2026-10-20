// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "ScholaMassBridgeTestTypes.h"
#if WITH_DEV_AUTOMATION_TESTS
#include "Misc/AutomationTest.h"
#include "ScholaMassInferenceBridge.h"
#include "Inference/DeterministicBatchBackend.h"
#include "MassEntityManager.h"
#include "MassEntityManagerStorage.h"

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FScholaMassBridgeTest, "Schola.Mass.Bridge.LifetimeRoutingAndMigration",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FScholaMassBridgeTest::RunTest(const FString&)
{
    auto Manager = MakeShared<FMassEntityManager>();
    Manager->Initialize(FMassEntityManagerStorageInitParams(TInPlaceType<FMassEntityManager_InitParams_Concurrent>()));
    const UScriptStruct* Types[] = {FScholaInferenceScheduleFragment::StaticStruct(), FScholaInferenceStateFragment::StaticStruct(), FScholaInferenceBucketFragment::StaticStruct()};
    const auto Archetype = Manager->CreateArchetype(MakeArrayView(Types));
    FScholaInferenceBucketFragment Bucket; Bucket.BucketId = 1;
    FMassArchetypeSharedFragmentValues Values;
    Values.Add(Manager->GetOrCreateConstSharedFragment(Bucket)); Values.Sort();
    FDeterministicBatchBackendSettings B; B.ObservationSize = B.ActionSize = 1;
    auto Backend = MakeShared<FDeterministicBatchBackend>(B);
    FNPCBackgroundSchedulerSettings Settings; Settings.WorkerCount = 2; Settings.MaxBatchSize = 2;
    auto Scheduler = FNPCBackgroundScheduler::Create(Settings, Backend);
    FScholaMassInferenceBridge Bridge;
    Bridge.AddBucket(1,7,Scheduler.ToSharedRef(),MakeShared<FNPCCohortScheduler>());
    auto Add = [&]()
    {
        const auto E = Manager->CreateEntity(Archetype, Values);
        Bridge.InitializeSlot(E,1, Manager->GetFragmentDataChecked<FScholaInferenceScheduleFragment>(E),
            Manager->GetFragmentDataChecked<FScholaInferenceStateFragment>(E),0);
        return E;
    };
    const auto A = Add(), Removed = Add(), Migrated = Add();
    Backend->SetGateOpen(false);
    TArray<FMassEntityHandle> Entities = {Migrated,Removed,A};
    TestTrue(TEXT("Submit shuffled handles"), Bridge.Submit(1,*Manager,Entities,[](TArrayView<float> Rows)
        { Rows[0]=30;Rows[1]=20;Rows[2]=10; },0,0) == ENPCBackgroundStatus::Pending);
    TestTrue(TEXT("Held in flight"), Backend->WaitForBlockedCalls(1));
    const int32 OldSlot = Manager->GetFragmentDataChecked<FScholaInferenceStateFragment>(Removed).Slot;
    Bridge.ReleaseSlot(Manager->GetFragmentDataChecked<FScholaInferenceStateFragment>(Removed));
    Manager->DestroyEntity(Removed);
    const auto Replacement = Add();
    TestEqual(TEXT("Mass index reused"), Replacement.Index, Removed.Index);
    TestTrue(TEXT("Reused index changes serial"), Replacement.SerialNumber != Removed.SerialNumber);
    TestEqual(TEXT("Dense slot reused"), Manager->GetFragmentDataChecked<FScholaInferenceStateFragment>(Replacement).Slot, OldSlot);
    Manager->AddTagToEntity(Migrated, FScholaMassMigrationTestTag::StaticStruct());
    Backend->SetGateOpen(true);
    const double Deadline = FPlatformTime::Seconds() + 5;
    while (Scheduler->GetStats().CompletedRows < 3 && FPlatformTime::Seconds() < Deadline) FPlatformProcess::SleepNoStats(.0001f);
    TArray<FMassEntityHandle> Ready;
    Bridge.Drain(*Manager,.01,1000,Ready);
    TestEqual(TEXT("Only live original handles"), Ready.Num(),2);
    for (auto Entity : Ready) Bridge.MarkDelivered(Manager->GetFragmentDataChecked<FScholaInferenceStateFragment>(Entity));
    auto Action = [&](FMassEntityHandle E) { return Bridge.GetHeldAction(Manager->GetFragmentDataChecked<FScholaInferenceStateFragment>(E)); };
    TestTrue(TEXT("Replacement never receives old action"), Action(Replacement).IsEmpty());
    TestTrue(TEXT("Shuffled routing A"), !Action(A).IsEmpty() && Action(A)[0]==20);
    TestTrue(TEXT("Migration preserves routing"), !Action(Migrated).IsEmpty() && Action(Migrated)[0]==60);
    TestTrue(TEXT("Handle serial lifetime"), !Manager->IsEntityValid(Removed));
    // Two live managers may have equal indices, but their pending requests and dense tables are independent.
    auto Manager2 = MakeShared<FMassEntityManager>();
    Manager2->Initialize(FMassEntityManagerStorageInitParams(TInPlaceType<FMassEntityManager_InitParams_Concurrent>()));
    auto Archetype2 = Manager2->CreateArchetype(MakeArrayView(Types));
    FMassArchetypeSharedFragmentValues Values2;
    Values2.Add(Manager2->GetOrCreateConstSharedFragment(Bucket)); Values2.Sort();
    const auto E2 = Manager2->CreateEntity(Archetype2,Values2);
    TestEqual(TEXT("Equal index across worlds"),E2.Index,A.Index);
    FScholaMassInferenceBridge OtherWorld;
    OtherWorld.AddBucket(1,8,Scheduler.ToSharedRef(),MakeShared<FNPCCohortScheduler>());
    auto& State2 = Manager2->GetFragmentDataChecked<FScholaInferenceStateFragment>(E2);
    OtherWorld.InitializeSlot(E2,1,Manager2->GetFragmentDataChecked<FScholaInferenceScheduleFragment>(E2),State2,0);
    TArray<FMassEntityHandle> Entities2 = {E2};
    OtherWorld.Submit(1,*Manager2,Entities2,[](TArrayView<float> Rows){Rows[0]=900;},0,0);
    while (Scheduler->GetStats().CompletedRows < 4 && FPlatformTime::Seconds() < Deadline) FPlatformProcess::SleepNoStats(.0001f);
    OtherWorld.Drain(*Manager2,.01,1000,Ready);
    TestEqual(TEXT("Second world receives its own result"),Ready.Num(),1);
    TestTrue(TEXT("Second-world action"),OtherWorld.GetHeldAction(State2).Num()==1 && OtherWorld.GetHeldAction(State2)[0]==1800);
    TestTrue(TEXT("First-world action unchanged"),!Action(A).IsEmpty() && Action(A)[0]==20);
    OtherWorld.Shutdown(); Manager2->Deinitialize();
    // Newest-only validation even if two generations of observations are both completed.
    Entities = {A};
    Bridge.Submit(1,*Manager,Entities,[](TArrayView<float> Rows){Rows[0]=100;},.02,0);
    Bridge.Submit(1,*Manager,Entities,[](TArrayView<float> Rows){Rows[0]=200;},.03,0);
    while (Scheduler->GetStats().CompletedRows < 6 && FPlatformTime::Seconds() < Deadline) FPlatformProcess::SleepNoStats(.0001f);
    Bridge.Drain(*Manager,.04,1000,Ready);
    TestEqual(TEXT("Newest result only"), Ready.Num(),1);
    for (auto Entity : Ready) Bridge.MarkDelivered(Manager->GetFragmentDataChecked<FScholaInferenceStateFragment>(Entity));
    TestTrue(TEXT("Newest held row"), !Action(A).IsEmpty() && Action(A)[0]==400);
    // Changing a const shared bucket fragment while work is in flight invalidates the old bucket's result.
    Backend->SetGateOpen(false);
    Bridge.Submit(1,*Manager,Entities,[](TArrayView<float> Rows){Rows[0]=300;},.05,0);
    TestTrue(TEXT("Bucket-change call in flight"),Backend->WaitForBlockedCalls(1));
    FScholaInferenceBucketFragment NewBucket; NewBucket.BucketId=2;
    Manager->SwapConstSharedFragmentForEntity(A,Manager->GetOrCreateConstSharedFragment(NewBucket));
    Backend->SetGateOpen(true);
    while (Scheduler->GetStats().CompletedRows < 7 && FPlatformTime::Seconds() < Deadline) FPlatformProcess::SleepNoStats(.0001f);
    Bridge.Drain(*Manager,.06,1000,Ready);
    TestTrue(TEXT("Old bucket result rejected"),Ready.IsEmpty());
    TestTrue(TEXT("Held action remains"),!Action(A).IsEmpty() && Action(A)[0]==400);

    Bridge.Shutdown();
    TestTrue(TEXT("Teardown clears slots"), Action(A).IsEmpty());
    Manager->Deinitialize();
    return true;
}
#endif

#include "MassExecutionContext.h"
void UScholaMassTestGather::WriteObservations(FMassExecutionContext& Context,TConstArrayView<int32> Indices,TArrayView<float> Rows)
{ for(int32 I=0;I<Indices.Num();++I)Rows[I]=float(Context.GetEntity(Indices[I]).Index); }
#if WITH_DEV_AUTOMATION_TESTS
#include "ScholaMassInferenceSubsystem.h"
#include "MassEntitySubsystem.h"
#include "MassExecutor.h"
#include "MassProcessingContext.h"
#include "Spaces/BoxSpace.h"
#include "Engine/World.h"
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FScholaMassParallelGatherTest,"Schola.Mass.Gather.SequentialEqualsParallel",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FScholaMassParallelGatherTest::RunTest(const FString&)
{
    auto* World=UWorld::CreateWorld(EWorldType::Game,false);
    auto* Service=World->GetSubsystem<UScholaMassInferenceSubsystem>();
    auto* Provider=NewObject<UScholaMassTestRowProvider>(World);
    auto Space=TInstancedStruct<FSpace>::Make<FBoxSpace>(TArray<float>{-100000},TArray<float>{100000});
    FString Error;FNPCBackgroundSchedulerSettings Settings;
    const int32 Id=Service->AcquireBucket(*Provider,FInteractionDefinition(Space,Space),Settings,Error);
    auto& Manager=World->GetSubsystem<UMassEntitySubsystem>()->GetMutableEntityManager();
    const UScriptStruct* Types[]={FScholaInferenceScheduleFragment::StaticStruct(),FScholaInferenceStateFragment::StaticStruct()};
    auto Archetype=Manager.CreateArchetype(MakeArrayView(Types));
    FScholaInferenceBucketFragment Bucket;Bucket.BucketId=Id;
    FMassArchetypeSharedFragmentValues Values;Values.Add(Manager.GetOrCreateConstSharedFragment(Bucket));Values.Sort();
    TArray<FMassEntityHandle> Entities;
    {auto Creation=Manager.BatchCreateEntities(Archetype,Values,3000,Entities);}
    TestFalse(TEXT("Test gather is never installed in game processing phases"),GetDefault<UScholaMassTestGather>()->ShouldAutoAddToGlobalList());
    auto* Gather=NewObject<UScholaMassTestGather>(World);Gather->MaxActionAgeMs=0;
    Gather->CallInitialize(World,Manager.AsShared());
    auto Scheduler=Service->Bridge.GetScheduler(Id);
    int64 ExpectedCompleted=0;
    for(auto Layout:{ENPCPhaseLayout::ModelAware,ENPCPhaseLayout::PerEntityUniform})
    for(bool bParallel:{false,true})
    {
        Gather->PhaseLayout=Layout;
        for(auto Entity:Entities)
        {
            auto& Schedule=Manager.GetFragmentDataChecked<FScholaInferenceScheduleFragment>(Entity);
            auto& State=Manager.GetFragmentDataChecked<FScholaInferenceStateFragment>(Entity);
            Service->Bridge.InitializeSlot(Entity,Id,Schedule,State,FPlatformTime::Seconds(),Layout);
            Schedule.NextDueTime=0;
        }
        Gather->ParallelGatherMinEntities=bParallel?1:MAX_int32;
        UE::Mass::FProcessingContext Context(Manager,1.f/60);
        UE::Mass::Executor::Run(*Gather,Context);
        ExpectedCompleted+=Entities.Num();
        const double Deadline=FPlatformTime::Seconds()+5;
        while(Scheduler->GetStats().CompletedRows<ExpectedCompleted && FPlatformTime::Seconds()<Deadline)FPlatformProcess::SleepNoStats(.0001f);
        TArray<FMassEntityHandle> Ready;
        Service->Bridge.Drain(Manager,FPlatformTime::Seconds(),0,Ready);
        TestEqual(TEXT("Every gathered row completed"),Ready.Num(),Entities.Num());
        for (auto Entity : Ready) Service->Bridge.MarkDelivered(Manager.GetFragmentDataChecked<FScholaInferenceStateFragment>(Entity));
        for(auto Entity:Entities)
        {
            const auto Row=Service->Bridge.GetHeldAction(Manager.GetFragmentDataChecked<FScholaInferenceStateFragment>(Entity));
            if(!TestTrue(TEXT("Prefix gather preserves identity"),Row.Num()==1 && Row[0]==2.f*Entity.Index))break;
        }
    }
    World->DestroyWorld(false);
    return true;
}
#endif

#if WITH_DEV_AUTOMATION_TESTS
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FScholaMassRetainLayoutTest,"Schola.Mass.Bridge.UrgentInitializationPreservesUniformLayout",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FScholaMassRetainLayoutTest::RunTest(const FString&)
{
    auto Manager=MakeShared<FMassEntityManager>();
    Manager->Initialize(FMassEntityManagerStorageInitParams(TInPlaceType<FMassEntityManager_InitParams_Concurrent>()));
    const UScriptStruct* Types[]={FScholaInferenceScheduleFragment::StaticStruct(),FScholaInferenceStateFragment::StaticStruct()};
    auto Archetype=Manager->CreateArchetype(MakeArrayView(Types));
    FScholaInferenceBucketFragment Bucket;Bucket.BucketId=1;
    FMassArchetypeSharedFragmentValues Values;Values.Add(Manager->GetOrCreateConstSharedFragment(Bucket));Values.Sort();
    auto Entity=Manager->CreateEntity(Archetype,Values);
    auto& Schedule=Manager->GetFragmentDataChecked<FScholaInferenceScheduleFragment>(Entity);
    auto& State=Manager->GetFragmentDataChecked<FScholaInferenceStateFragment>(Entity);
    FDeterministicBatchBackendSettings BackendSettings;BackendSettings.ObservationSize=BackendSettings.ActionSize=1;
    auto Backend=MakeShared<FDeterministicBatchBackend>(BackendSettings);Backend->SetGateOpen(false);
    auto Scheduler=FNPCBackgroundScheduler::Create(FNPCBackgroundSchedulerSettings(),Backend);
    FScholaMassInferenceBridge Bridge;Bridge.AddBucket(1,1,Scheduler.ToSharedRef(),MakeShared<FNPCCohortScheduler>());
    Bridge.InitializeSlot(Entity,1,Schedule,State,0,ENPCPhaseLayout::PerEntityUniform);
    const uint32 Serial=State.SlotSerial;
    TArray<FMassEntityHandle> Entities{Entity};
    TestTrue(TEXT("Submit uniform-phased request"),Bridge.Submit(1,*Manager,Entities,[](TArrayView<float> Rows){Rows[0]=7;},0,0)==ENPCBackgroundStatus::Pending);
    TestTrue(TEXT("Backend request is in flight"),Backend->WaitForBlockedCalls(1));
    const uint32 Submitted=State.SubmittedSerial;
    // Exactly the call used by the explicit urgent signal processor before changing NextDueTime.
    Bridge.InitializeSlot(Entity,1,Schedule,State,.01);
    TestEqual(TEXT("Urgent initialization retains slot lifetime"),State.SlotSerial,Serial);
    TestEqual(TEXT("Urgent initialization retains pending observation"),State.SubmittedSerial,Submitted);
    Backend->SetGateOpen(true);
    const double Deadline=FPlatformTime::Seconds()+5;
    while(Scheduler->GetStats().CompletedRows<1 && FPlatformTime::Seconds()<Deadline)FPlatformProcess::SleepNoStats(.0001f);
    TArray<FMassEntityHandle> Ready;Bridge.Drain(*Manager,.02,1000,Ready);
    TestEqual(TEXT("Existing valid row is still routed"),Ready.Num(),1);
    Bridge.MarkDelivered(State);
    Bridge.InitializeSlot(Entity,1,Schedule,State,.03);
    const auto Action=Bridge.GetHeldAction(State);
    TestTrue(TEXT("Held delivery survives urgent initialization"),Bridge.IsActionDelivered(State) && Action.Num()==1 && Action[0]==14);
    Schedule.PeriodSeconds*=2;
    Bridge.InitializeSlot(Entity,1,Schedule,State,.04);
    const uint32 NewSerial=State.SlotSerial;
    TestEqual(TEXT("Rate change preserves entity lifetime"),NewSerial,Serial);
    TestTrue(TEXT("Rate change preserves held delivery"),Bridge.IsActionDelivered(State));
    Bridge.InitializeSlot(Entity,1,Schedule,State,.04,ENPCPhaseLayout::PerEntityUniform);
    TestEqual(TEXT("Rate change retained uniform layout"),State.SlotSerial,NewSerial);
    Bridge.InitializeSlot(Entity,1,Schedule,State,.05,ENPCPhaseLayout::ModelAware);
    TestEqual(TEXT("Layout change preserves entity lifetime"),State.SlotSerial,Serial);
    TestTrue(TEXT("Layout change preserves held delivery"),Bridge.IsActionDelivered(State));
    Bridge.Shutdown();Manager->Deinitialize();return true;
}
#endif

#if WITH_DEV_AUTOMATION_TESTS
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FScholaMassGlobalOldestTest,"Schola.Mass.Gather.OldestAdmissionAcrossBuckets",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FScholaMassGlobalOldestTest::RunTest(const FString&)
{
    auto* World=UWorld::CreateWorld(EWorldType::Game,false);
    auto* Service=World->GetSubsystem<UScholaMassInferenceSubsystem>();
    auto& Manager=World->GetSubsystem<UMassEntitySubsystem>()->GetMutableEntityManager();
    FDeterministicBatchBackendSettings BackendSettings;BackendSettings.ObservationSize=BackendSettings.ActionSize=1;
    auto BackendA=MakeShared<FDeterministicBatchBackend>(BackendSettings),BackendB=MakeShared<FDeterministicBatchBackend>(BackendSettings);
    auto A=FNPCBackgroundScheduler::Create(FNPCBackgroundSchedulerSettings(),BackendA);
    auto B=FNPCBackgroundScheduler::Create(FNPCBackgroundSchedulerSettings(),BackendB);
    Service->Bridge.AddBucket(1,1,A.ToSharedRef(),MakeShared<FNPCCohortScheduler>());
    Service->Bridge.AddBucket(2,1,B.ToSharedRef(),MakeShared<FNPCCohortScheduler>());
    const UScriptStruct* Types[]={FScholaInferenceScheduleFragment::StaticStruct(),FScholaInferenceStateFragment::StaticStruct()};
    const auto Archetype=Manager.CreateArchetype(MakeArrayView(Types));
    TArray<FMassEntityHandle> Entities;
    for(int32 BucketId:{1,2})
    {
        FScholaInferenceBucketFragment Bucket;Bucket.BucketId=BucketId;
        FMassArchetypeSharedFragmentValues Values;Values.Add(Manager.GetOrCreateConstSharedFragment(Bucket));Values.Sort();
        auto Entity=Manager.CreateEntity(Archetype,Values);Entities.Add(Entity);
        auto& Schedule=Manager.GetFragmentDataChecked<FScholaInferenceScheduleFragment>(Entity);
        Service->Bridge.InitializeSlot(Entity,BucketId,Schedule,Manager.GetFragmentDataChecked<FScholaInferenceStateFragment>(Entity),FPlatformTime::Seconds());
        Schedule.NextDueTime=FPlatformTime::Seconds()-(BucketId==1?1:2);
    }
    auto* Gather=NewObject<UScholaMassTestGather>(World);Gather->MaxRowsPerFrame=1;Gather->MaxActionAgeMs=0;
    Gather->CallInitialize(World,Manager.AsShared());
    // Reverse age order as well: this catches bucket-order admission regardless
    // of which shared-value chunk happens to be enumerated first by Mass.
    for(int32 OlderId:{1,2})
    {
        A->ResetStats();B->ResetStats();
        const int64 DeferredBefore=Service->Bridge.DeferredRows;
        for(int32 I=0;I<Entities.Num();++I)
            Manager.GetFragmentDataChecked<FScholaInferenceScheduleFragment>(Entities[I]).NextDueTime=
                FPlatformTime::Seconds()-(I+1==OlderId?2:1);
        {UE::Mass::FProcessingContext Context(Manager,1.f/60);UE::Mass::Executor::Run(*Gather,Context);}
        const double Deadline=FPlatformTime::Seconds()+5;
        while(A->GetStats().CompletedRows+B->GetStats().CompletedRows<1 && FPlatformTime::Seconds()<Deadline)FPlatformProcess::SleepNoStats(.0001f);
        auto Older=OlderId==1?A:B;auto Newer=OlderId==1?B:A;
        TestEqual(TEXT("Newer bucket cannot spend the shared frame budget"),Newer->GetStats().AcceptedRows,int64(0));
        TestEqual(TEXT("Globally oldest entity is admitted first"),Older->GetStats().AcceptedRows,int64(1));
        TestEqual(TEXT("One deferred row is accounted"),Service->Bridge.DeferredRows,DeferredBefore+1);
        TArray<FMassEntityHandle> Ready;Service->Bridge.Drain(Manager,FPlatformTime::Seconds(),0,Ready);
        TestTrue(TEXT("Oldest bucket receives its own row"),Ready.Num()==1 && Ready[0]==Entities[OlderId-1]);
        for(auto Entity:Ready)Service->Bridge.MarkDelivered(Manager.GetFragmentDataChecked<FScholaInferenceStateFragment>(Entity));
    }
    World->DestroyWorld(false);return true;
}
#endif

#if WITH_DEV_AUTOMATION_TESTS
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FScholaMassUrgentPhaseTest,"Schola.Mass.Bridge.UrgentDecisionResumesPeriodicPhase",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FScholaMassUrgentPhaseTest::RunTest(const FString&)
{
    auto Manager=MakeShared<FMassEntityManager>();
    Manager->Initialize(FMassEntityManagerStorageInitParams(TInPlaceType<FMassEntityManager_InitParams_Concurrent>()));
    const UScriptStruct* Types[]={FScholaInferenceScheduleFragment::StaticStruct(),FScholaInferenceStateFragment::StaticStruct()};
    auto Archetype=Manager->CreateArchetype(MakeArrayView(Types));
    FScholaInferenceBucketFragment Bucket;Bucket.BucketId=1;
    FMassArchetypeSharedFragmentValues Values;Values.Add(Manager->GetOrCreateConstSharedFragment(Bucket));Values.Sort();
    auto Entity=Manager->CreateEntity(Archetype,Values);
    FDeterministicBatchBackendSettings BackendSettings;BackendSettings.ObservationSize=BackendSettings.ActionSize=1;
    auto Backend=MakeShared<FDeterministicBatchBackend>(BackendSettings);
    auto Scheduler=FNPCBackgroundScheduler::Create(FNPCBackgroundSchedulerSettings(),Backend);
    FScholaMassInferenceBridge Bridge;Bridge.AddBucket(1,1,Scheduler.ToSharedRef(),MakeShared<FNPCCohortScheduler>(17));
    auto& Schedule=Manager->GetFragmentDataChecked<FScholaInferenceScheduleFragment>(Entity);
    auto& State=Manager->GetFragmentDataChecked<FScholaInferenceStateFragment>(Entity);
    Bridge.InitializeSlot(Entity,1,Schedule,State,0);
    const double PeriodicDue=Schedule.NextDueTime;
    TestTrue(TEXT("Original periodic phase is in the future"),PeriodicDue>0);
    Schedule.bUrgent=true;Schedule.NextDueTime=0;
    TArray<FMassEntityHandle> Entities{Entity};
    TestTrue(TEXT("Urgent row admitted"),Bridge.Submit(1,*Manager,Entities,
        [](TArrayView<float> Rows){Rows[0]=3;},0,0)==ENPCBackgroundStatus::Pending);
    TestTrue(TEXT("Urgency resumes the original balanced phase"),FMath::Abs(Schedule.NextDueTime-PeriodicDue)<1e-6);
    TestFalse(TEXT("Urgency clears only after admission"),Schedule.bUrgent);
    const double Deadline=FPlatformTime::Seconds()+5;
    while(Scheduler->GetStats().CompletedRows<1 && FPlatformTime::Seconds()<Deadline)FPlatformProcess::SleepNoStats(.0001f);
    TArray<FMassEntityHandle> Ready;Bridge.Drain(*Manager,.01,1000,Ready);
    TestEqual(TEXT("Urgent result routes normally"),Ready.Num(),1);
    Bridge.MarkDelivered(State);
    TestTrue(TEXT("Urgent action delivered"),Bridge.IsActionDelivered(State));
    Bridge.Shutdown();Manager->Deinitialize();return true;
}
#endif

#if WITH_DEV_AUTOMATION_TESTS
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FScholaMassObserverLifetimeTest,"Schola.Mass.Observer.DestroyAndFragmentRemovalReleaseSlots",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FScholaMassObserverLifetimeTest::RunTest(const FString&)
{
    auto* World=UWorld::CreateWorld(EWorldType::Game,false);
    auto* Service=World->GetSubsystem<UScholaMassInferenceSubsystem>();
    auto& Manager=World->GetSubsystem<UMassEntitySubsystem>()->GetMutableEntityManager();
    FDeterministicBatchBackendSettings BackendSettings;BackendSettings.ObservationSize=BackendSettings.ActionSize=1;
    auto Backend=MakeShared<FDeterministicBatchBackend>(BackendSettings);
    auto Scheduler=FNPCBackgroundScheduler::Create(FNPCBackgroundSchedulerSettings(),Backend);
    Service->Bridge.AddBucket(44,1,Scheduler.ToSharedRef(),MakeShared<FNPCCohortScheduler>());
    const UScriptStruct* Types[]={FScholaInferenceScheduleFragment::StaticStruct(),FScholaInferenceStateFragment::StaticStruct()};
    const auto Archetype=Manager.CreateArchetype(MakeArrayView(Types));
    FScholaInferenceBucketFragment Bucket;Bucket.BucketId=44;
    FMassArchetypeSharedFragmentValues Values;Values.Add(Manager.GetOrCreateConstSharedFragment(Bucket));Values.Sort();
    auto Entity=Manager.CreateEntity(Archetype,Values);
    auto State=Manager.GetFragmentDataChecked<FScholaInferenceStateFragment>(Entity);
    TestTrue(TEXT("Real creation observer initialized the slot"),State.Slot!=INDEX_NONE);
    int64 Expected=0;
    auto Deliver=[&](FMassEntityHandle Target)
    {
        TArray<FMassEntityHandle> Entities{Target};
        TestTrue(TEXT("Admit observer-bound entity"),Service->Bridge.Submit(44,Manager,Entities,
            [](TArrayView<float> Row){Row[0]=3;},FPlatformTime::Seconds(),0)==ENPCBackgroundStatus::Pending);
        ++Expected;const double Deadline=FPlatformTime::Seconds()+5;
        while(Scheduler->GetStats().CompletedRows<Expected && FPlatformTime::Seconds()<Deadline)FPlatformProcess::SleepNoStats(.0001f);
        TArray<FMassEntityHandle> Ready;Service->Bridge.Drain(Manager,FPlatformTime::Seconds(),0,Ready);
        for(auto ReadyEntity:Ready)Service->Bridge.MarkDelivered(Manager.GetFragmentDataChecked<FScholaInferenceStateFragment>(ReadyEntity));
        return Manager.GetFragmentDataChecked<FScholaInferenceStateFragment>(Target);
    };
    State=Deliver(Entity);
    TestTrue(TEXT("Held row exists before destruction"),!Service->Bridge.GetHeldAction(State).IsEmpty());
    Manager.DestroyEntity(Entity);
    TestTrue(TEXT("DestroyEntity observer releases the held slot"),Service->Bridge.GetHeldAction(State).IsEmpty());
    auto Replacement=Manager.CreateEntity(Archetype,Values);
    auto ReplacementState=Manager.GetFragmentDataChecked<FScholaInferenceStateFragment>(Replacement);
    TestEqual(TEXT("Destruction makes the dense slot reusable"),ReplacementState.Slot,State.Slot);
    TestTrue(TEXT("Reused slot has a new lifetime"),ReplacementState.SlotSerial!=State.SlotSerial);
    ReplacementState=Deliver(Replacement);
    TestTrue(TEXT("Replacement has a held row"),!Service->Bridge.GetHeldAction(ReplacementState).IsEmpty());
    Manager.RemoveFragmentFromEntity(Replacement,FScholaInferenceStateFragment::StaticStruct());
    TestTrue(TEXT("RemoveElement observer releases the held slot"),Service->Bridge.GetHeldAction(ReplacementState).IsEmpty());
    Manager.AddFragmentToEntity(Replacement,FScholaInferenceStateFragment::StaticStruct());
    const auto Restored=Manager.GetFragmentDataChecked<FScholaInferenceStateFragment>(Replacement);
    TestEqual(TEXT("Fragment re-add reuses the released slot"),Restored.Slot,ReplacementState.Slot);
    TestTrue(TEXT("Fragment re-add creates a new slot lifetime"),Restored.SlotSerial!=ReplacementState.SlotSerial);
    World->DestroyWorld(false);return true;
}
#endif

#if WITH_DEV_AUTOMATION_TESTS
#include "MassSignalSubsystem.h"
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FScholaMassDecideNowSignalTest,"Schola.Mass.Signal.DecideNowMarksOnlySignalledEntities",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FScholaMassDecideNowSignalTest::RunTest(const FString&)
{
    auto* World=UWorld::CreateWorld(EWorldType::Game,false);
    auto* Service=World->GetSubsystem<UScholaMassInferenceSubsystem>();
    auto* Signals=World->GetSubsystem<UMassSignalSubsystem>();
    if(!TestNotNull(TEXT("Signal subsystem"),Signals)){World->DestroyWorld(false);return false;}
    auto& Manager=World->GetSubsystem<UMassEntitySubsystem>()->GetMutableEntityManager();
    FDeterministicBatchBackendSettings BackendSettings;BackendSettings.ObservationSize=BackendSettings.ActionSize=1;
    auto Scheduler=FNPCBackgroundScheduler::Create(FNPCBackgroundSchedulerSettings(),MakeShared<FDeterministicBatchBackend>(BackendSettings));
    Service->Bridge.AddBucket(45,1,Scheduler.ToSharedRef(),MakeShared<FNPCCohortScheduler>());
    const UScriptStruct* Types[]={FScholaInferenceScheduleFragment::StaticStruct(),FScholaInferenceStateFragment::StaticStruct()};
    const auto Archetype=Manager.CreateArchetype(MakeArrayView(Types));
    FScholaInferenceBucketFragment Bucket;Bucket.BucketId=45;
    FMassArchetypeSharedFragmentValues Values;Values.Add(Manager.GetOrCreateConstSharedFragment(Bucket));Values.Sort();
    const auto Signalled=Manager.CreateEntity(Archetype,Values),Quiet=Manager.CreateEntity(Archetype,Values);
    for(auto Entity:{Signalled,Quiet})
    {
        auto& Schedule=Manager.GetFragmentDataChecked<FScholaInferenceScheduleFragment>(Entity);
        Schedule.bUrgent=false;Schedule.NextDueTime=1e9;
    }
    const uint32 Serial=Manager.GetFragmentDataChecked<FScholaInferenceStateFragment>(Signalled).SlotSerial;
    auto* Processor=NewObject<UScholaMassDecideNowProcessor>(World);
    Processor->CallInitialize(World,Manager.AsShared());
    auto Run=[&]{UE::Mass::FProcessingContext Context(Manager,1.f/60);UE::Mass::Executor::Run(*Processor,Context);};
    Run();
    TestFalse(TEXT("No signal, no urgency"),Manager.GetFragmentDataChecked<FScholaInferenceScheduleFragment>(Signalled).bUrgent);
    Signals->SignalEntity(TEXT("ScholaDecideNow"),Signalled);
    const double SignalTime=FPlatformTime::Seconds();
    Run();
    const auto& Urgent=Manager.GetFragmentDataChecked<FScholaInferenceScheduleFragment>(Signalled);
    TestTrue(TEXT("Signalled entity marked urgent"),Urgent.bUrgent);
    TestTrue(TEXT("Signalled entity is due now"),Urgent.NextDueTime<=FPlatformTime::Seconds() && Urgent.NextDueTime>=SignalTime-1);
    TestEqual(TEXT("Signal keeps the slot lifetime"),Manager.GetFragmentDataChecked<FScholaInferenceStateFragment>(Signalled).SlotSerial,Serial);
    const auto& Other=Manager.GetFragmentDataChecked<FScholaInferenceScheduleFragment>(Quiet);
    TestTrue(TEXT("Unsignalled entity untouched"),!Other.bUrgent && Other.NextDueTime==1e9);
    World->DestroyWorld(false);
    return true;
}
#endif
