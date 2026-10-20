// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#if WITH_DEV_AUTOMATION_TESTS
#include "Misc/AutomationTest.h"
#include "BackgroundAdapterTestTypes.h"
#include "Steppers/BackgroundStepper.h"
#include "../Steppers/GatedStepperTestTypes.h"
#include "Spaces/BoxSpace.h"
#include "Points/BoxPoint.h"
#include "Async/Async.h"
#include "Engine/World.h"

namespace
{
FInteractionDefinition AdapterDefinition()
{
    auto S = TInstancedStruct<FSpace>::Make<FBoxSpace>(TArray<float>{-10000}, TArray<float>{10000});
    return FInteractionDefinition(S,S);
}
FNPCInferenceConfig AdapterConfig(UBackgroundAdapterTestProvider*& Provider, float Failure = 0.f)
{
    Provider = NewObject<UBackgroundAdapterTestProvider>();
    FDeterministicBatchBackendSettings B;
    B.ObservationSize = B.ActionSize = 1; B.FailureProbability = Failure;
    Provider->Backend = MakeShared<FDeterministicBatchBackend>(B);
    FNPCInferenceConfig C;
    C.Paradigm = ENPCInferenceParadigm::Background;
    C.BackgroundProvider = Provider; C.FanOut = ENPCInferenceFanOut::SharedQueue;
    return C;
}
TArray<TInstancedStruct<FPoint>> AdapterPoints(float V)
{ return {TInstancedStruct<FPoint>::Make<FBoxPoint>(TArray<float>{V})}; }
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundScheduledDeadlineTest, "Schola.Inference.Background.Stepper.ScheduledDeadlineIncludesBusyDeferral",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FBackgroundScheduledDeadlineTest::RunTest(const FString&)
{
    UBackgroundAdapterTestProvider* Provider;auto Config=AdapterConfig(Provider);
    Config.BackgroundSettings.MaxPendingRows=1;
    auto* Policy=NewObject<UNPCInferencePolicy>();Policy->Configure(Config);Policy->Init(AdapterDefinition());
    auto* Agent=NewObject<USequenceTestAgent>();auto* Stepper=NewObject<UBackgroundStepper>();
    Stepper->MaxActionAgeMs=10000;
    Stepper->Init(Policy,{TScriptInterface<IAgent>(Agent)},{1.f});
    Provider->Backend->SetGateOpen(false);
    Stepper->Step(0);
    TestTrue(TEXT("First observation is in flight"),Provider->Backend->WaitForBlockedCalls(1));
    TestEqual(TEXT("Next periodic deadline"),Stepper->GetNextDecisionTime(0),1.0);
    Stepper->Step(1.1);
    TestEqual(TEXT("Busy preserves the scheduled deadline instead of resetting it to submission time"),Stepper->GetNextDecisionTime(0),1.0);
    TestTrue(TEXT("Admission was deferred"),Stepper->DeferredRows>0);
    Provider->Backend->SetGateOpen(true);
    const double Deadline=FPlatformTime::Seconds()+5;
    while(Policy->GetBackgroundStats().CompletedRows<1 && FPlatformTime::Seconds()<Deadline)FPlatformProcess::SleepNoStats(.0001f);
    Stepper->Step(1.2);
    TestEqual(TEXT("Accepted work advances the periodic deadline"),Stepper->GetNextDecisionTime(0),2.0);
    Stepper->RemoveAgent(0);
    TestEqual(TEXT("Removed slots expose no deadline"),Stepper->GetNextDecisionTime(0),0.0);
    Stepper->Reset();
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundResetCallbackTest, "Schola.Inference.Background.Stepper.ResetInsideAgentCallback",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FBackgroundResetCallbackTest::RunTest(const FString&)
{
    for (bool bRaw : {false, true})
    for (bool bObserve : {true, false})
    {
        UBackgroundAdapterTestProvider* Provider;auto Config=AdapterConfig(Provider);
        auto* Policy=NewObject<UNPCInferencePolicy>();Policy->Configure(Config);Policy->Init(AdapterDefinition());
        auto* A=bRaw?static_cast<USequenceTestAgent*>(NewObject<UBackgroundCallbackRowAgent>()):NewObject<USequenceTestAgent>();
        auto* B=bRaw?static_cast<USequenceTestAgent*>(NewObject<UBackgroundCallbackRowAgent>()):NewObject<USequenceTestAgent>();
        auto* Stepper=NewObject<UBackgroundStepper>();
        Stepper->Init(Policy,{TScriptInterface<IAgent>(A),TScriptInterface<IAgent>(B)},{1.f,1.f});
        Stepper->RequestDecisionNow(0);Stepper->RequestDecisionNow(1);
        if(bObserve)A->OnObserve=[Stepper]{Stepper->Reset();CollectGarbage(RF_NoFlags);};
        else {A->OnAct=[Stepper]{Stepper->Reset();};B->OnAct=[Stepper]{Stepper->Reset();};}
        Stepper->Step(0);
        if(!bObserve)
        {
            const double Deadline=FPlatformTime::Seconds()+5;
            while(Policy->GetBackgroundStats().CompletedRows<2 && FPlatformTime::Seconds()<Deadline)FPlatformProcess::SleepNoStats(.0001f);
            Stepper->Step(.01);
            TestEqual(TEXT("No further actions after the first callback resets delivery"),A->Applied.Num()+B->Applied.Num(),1);
        }
        TestEqual(TEXT("Callback reset leaves no active submission state"),Stepper->SubmittedRows,int64(0));
        Stepper->Step(.02);
        TestEqual(TEXT("Reset prevents later delivery"),A->Applied.Num()+B->Applied.Num(),bObserve?0:1);
    }
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundMutationCallbackTest, "Schola.Inference.Background.Stepper.MutationInsideAgentCallback",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FBackgroundMutationCallbackTest::RunTest(const FString&)
{
    for (bool bRaw : {false, true})
    for (bool bRemove : {true, false})
    {
        UBackgroundAdapterTestProvider* Provider;auto Config=AdapterConfig(Provider);
        auto* Policy=NewObject<UNPCInferencePolicy>();Policy->Configure(Config);Policy->Init(AdapterDefinition());
        auto* Agent=bRaw?static_cast<USequenceTestAgent*>(NewObject<UBackgroundCallbackRowAgent>()):NewObject<USequenceTestAgent>();
        auto* Stepper=NewObject<UBackgroundStepper>();
        Stepper->Init(Policy,{TScriptInterface<IAgent>(Agent)},{1.f});
        Stepper->RequestDecisionNow(0);
        Agent->OnObserve=[&,Stepper,Agent,Policy]
        {
            Stepper->Step(0); // Nested stepping cannot recurse into the callback.
            TestEqual(TEXT("Callback cannot relocate slot storage"),Stepper->AddAgent(TScriptInterface<IAgent>(Agent),1.f),INDEX_NONE);
            if(bRemove)Stepper->RemoveAgent(0);
            else {Policy->Configure(Config);TestTrue(TEXT("Callback reconfiguration"),Policy->Init(AdapterDefinition()));}
        };
        Stepper->Step(0);
        TestEqual(TEXT("Callback completes with its observation storage alive"),Agent->ObservationCounter,1);
        TestEqual(TEXT("Removed or rebound observation is never submitted for delivery"),Stepper->SubmittedRows,int64(0));
        TestEqual(TEXT("Removed or reset slot has no deadline"),Stepper->GetNextDecisionTime(0),0.0);
        Stepper->Step(.01);
        TestEqual(TEXT("No later callback after removal or rebind"),Agent->ObservationCounter,1);
        TestEqual(TEXT("No stale actions"),Agent->Applied.Num(),0);
        Stepper->Reset();
    }
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundPolicyStatuses, "Schola.Inference.Background.Policy.StatusesAndTransaction",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FBackgroundPolicyStatuses::RunTest(const FString&)
{
    auto* Policy = NewObject<UNPCInferencePolicy>();
    UBackgroundAdapterTestProvider* Provider;
    auto Config = AdapterConfig(Provider);
    Config.BackgroundSettings.MaxPendingRows = 1;
    Policy->Configure(Config); TestTrue(TEXT("Init"), Policy->Init(AdapterDefinition()));
    Provider->Backend->SetGateOpen(false);
    FNPCBackgroundRequest Request, Other;
    auto Out = AdapterPoints(99), Sentinel = Out;
    TestTrue(TEXT("Invalid"), Policy->SubmitBackground({}, {}, Other) == ENPCBackgroundStatus::Invalid);
    TestTrue(TEXT("Pending submit"), Policy->SubmitBackground(AdapterPoints(3), {}, Request) == ENPCBackgroundStatus::Pending);
    TestTrue(TEXT("Backend blocked"), Provider->Backend->WaitForBlockedCalls(1));
    TestTrue(TEXT("Pending status"), Policy->GetBackgroundStatus(Request) == ENPCBackgroundStatus::Pending);
    TestTrue(TEXT("Pending consume"), Policy->ConsumeBackground(Request, Out) == ENPCBackgroundStatus::Pending);
    TestTrue(TEXT("Pending preserves output"), Out == Sentinel);
    TestTrue(TEXT("Busy"), Policy->SubmitBackground(AdapterPoints(4), {}, Other) == ENPCBackgroundStatus::Busy);
    TestTrue(TEXT("Busy reflects capacity"), Policy->IsInferenceBusy());
    TestTrue(TEXT("Cancel"), Policy->CancelBackground(Request));
    TestTrue(TEXT("Cancelled status"), Policy->GetBackgroundStatus(Request) == ENPCBackgroundStatus::Cancelled);
    Provider->Backend->SetGateOpen(true);
    const double CancelDeadline=FPlatformTime::Seconds()+5;
    while(Policy->GetBackgroundStats().CancelledRows<1 && FPlatformTime::Seconds()<CancelDeadline)FPlatformProcess::SleepNoStats(.0001f);
    TestTrue(TEXT("Unconsumed cancelled ticket still fills policy admission"),Policy->IsInferenceBusy());
    TestTrue(TEXT("Cancelled consume"), Policy->ConsumeBackground(Request, Out) == ENPCBackgroundStatus::Cancelled);
    TestTrue(TEXT("Cancel preserves output"), Out == Sentinel);
    TestFalse(TEXT("Consuming cancellation releases ticket admission"),Policy->IsInferenceBusy());
    TestTrue(TEXT("Retired invalid"), Policy->GetBackgroundStatus(Request) == ENPCBackgroundStatus::Invalid);
    Config.BackgroundSettings.MaxPendingRows = 10;
    Policy->Configure(Config); Policy->Init(AdapterDefinition());
    TSharedPtr<FNPCBackgroundScheduler, ESPMode::ThreadSafe> S;
    TSharedPtr<const FNPCRowCodec, ESPMode::ThreadSafe> Codec; uint32 Generation;
    Policy->GetBackgroundBinding(S, Codec, Generation);
    Policy->SubmitBackground(AdapterPoints(3), {}, Request);
    // Wait is deliberately confined to test code; public async methods never wait.
    const double Deadline = FPlatformTime::Seconds() + 5;
    while (Policy->GetBackgroundStatus(Request) == ENPCBackgroundStatus::Pending && FPlatformTime::Seconds() < Deadline) FPlatformProcess::SleepNoStats(0.0001f);
    TestTrue(TEXT("Completed"), Policy->ConsumeBackground(Request, Out) == ENPCBackgroundStatus::Completed);
    TestEqual(TEXT("Real backend action"), Out[0].Get<FBoxPoint>().Values[0], 6.f);
    auto Old = Request;
    Policy->Configure(Config); Policy->Init(AdapterDefinition());
    Policy->SubmitBackground(AdapterPoints(1), {}, Request);
    TestTrue(TEXT("No handle alias after Configure"), Request.Id != Old.Id && Policy->GetBackgroundStatus(Old) == ENPCBackgroundStatus::Invalid);
    Policy->CancelBackground(Request); Policy->ConsumeBackground(Request, Out);
    // Parallel pipelined callers use independent output containers.
    auto Work = [Policy]() { TArray<TInstancedStruct<FPoint>> A; return Policy->BatchedThink(AdapterPoints(4), A) && A[0].Get<FBoxPoint>().Values[0] == 8.f; };
    auto A = Async(EAsyncExecution::ThreadPool, Work); auto B = Async(EAsyncExecution::ThreadPool, Work);
    TestTrue(TEXT("Concurrent callers"), A.Get() && B.Get());
    Config = AdapterConfig(Provider, 1.f);
    Policy->Configure(Config); Policy->Init(AdapterDefinition());
    Out = Sentinel;
    Policy->SubmitBackground(AdapterPoints(3), {}, Request);
    while (Policy->GetBackgroundStatus(Request) == ENPCBackgroundStatus::Pending) FPlatformProcess::SleepNoStats(0.0001f);
    TestTrue(TEXT("Failed"), Policy->ConsumeBackground(Request, Out) == ENPCBackgroundStatus::Failed);
    TestTrue(TEXT("Failure preserves output"), Out == Sentinel);
    Config = AdapterConfig(Provider);
    Config.WorkerCount = 1; Config.MaxBatchSize = 1;
    Config.BackgroundSettings.MaxInFlightBatchesPerWorker = 1;
    Config.BackgroundSettings.DefaultMaxAgeMs = .1f;
    Policy->Configure(Config); Policy->Init(AdapterDefinition());
    Provider->Backend->SetGateOpen(false);
    Policy->SubmitBackground(AdapterPoints(1), {}, Other);
    TestTrue(TEXT("Expiry blocker"), Provider->Backend->WaitForBlockedCalls(1));
    Policy->SubmitBackground(AdapterPoints(2), {}, Request);
    FPlatformProcess::SleepNoStats(.002f);
    Provider->Backend->SetGateOpen(true);
    const double ExpiryDeadline = FPlatformTime::Seconds() + 5;
    while (Policy->GetBackgroundStatus(Request) == ENPCBackgroundStatus::Pending && FPlatformTime::Seconds() < ExpiryDeadline) FPlatformProcess::SleepNoStats(.0001f);
    Out = Sentinel;
    TestTrue(TEXT("Expired"), Policy->ConsumeBackground(Request, Out) == ENPCBackgroundStatus::Expired);
    TestTrue(TEXT("Expiry preserves output"), Out == Sentinel);
    Policy->CancelBackground(Other);

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundBucketSharing, "Schola.Inference.Background.Policy.SharedBuckets",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FBackgroundBucketSharing::RunTest(const FString&)
{
    UBackgroundAdapterTestProvider* Provider; auto Config = AdapterConfig(Provider);
    UWorld* World = UWorld::CreateWorld(EWorldType::Game, false);
    auto* Registry = World->GetSubsystem<UNPCBackgroundInferenceSubsystem>();
    if (!TestNotNull(TEXT("Registry"), Registry)) { World->DestroyWorld(false); return false; }
    FString Error;
    auto H1 = Registry->AcquireBucket(*Provider, AdapterDefinition(), Config.BackgroundSettings, Error);
    auto H2 = Registry->AcquireBucket(*Provider, AdapterDefinition(), Config.BackgroundSettings, Error);
    TestTrue(TEXT("Shared identity"), H1.Id == H2.Id && H1.Generation == H2.Generation);
    auto S = Registry->ShareScheduler(H1);
    TestTrue(TEXT("Same scheduler"), S == Registry->ShareScheduler(H2));
    Registry->ReleaseBucket(H1);
    TestNotNull(TEXT("One reference remains"), Registry->GetScheduler(H2));
    Registry->ReleaseBucket(H2);
    TestNull(TEXT("Last release removes bucket"), Registry->GetScheduler(H2));
    TestFalse(TEXT("Last release shuts workers down"), S->CanAccept(1));
    auto H3 = Registry->AcquireBucket(*Provider, AdapterDefinition(), Config.BackgroundSettings, Error);
    TestTrue(TEXT("Generation advances"), H3.Generation != H1.Generation);
    S = Registry->ShareScheduler(H3);
    World->DestroyWorld(false);
    TestFalse(TEXT("World teardown shuts workers down"), S->CanAccept(1));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundStepperLifecycle, "Schola.Inference.Background.Stepper.LifetimeAndAge",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FBackgroundStepperLifecycle::RunTest(const FString&)
{
    UBackgroundAdapterTestProvider* Provider; auto Config = AdapterConfig(Provider);
    auto* Policy = NewObject<UNPCInferencePolicy>(); Policy->Configure(Config); Policy->Init(AdapterDefinition());
    auto* Agent = NewObject<USequenceTestAgent>();
    TScriptInterface<IAgent> Interface(Agent);
    auto* Stepper = NewObject<UBackgroundStepper>();
    Stepper->MaxActionAgeMs = 1000;
    TestTrue(TEXT("Init"), Stepper->Init(Policy, {Interface}, {10.f}));
    Provider->Backend->SetGateOpen(false);
    Stepper->Step(0);
    TestTrue(TEXT("In flight"), Provider->Backend->WaitForBlockedCalls(1));
    Stepper->RemoveAgent(0);
    auto* Replacement = NewObject<USequenceTestAgent>();
    TestEqual(TEXT("Reuse slot"), Stepper->AddAgent(TScriptInterface<IAgent>(Replacement), 10.f), 0);
    Provider->Backend->SetGateOpen(true);
    const double Deadline = FPlatformTime::Seconds()+5;
    while (Policy->GetBackgroundStats().CompletedRows < 1 && FPlatformTime::Seconds() < Deadline) FPlatformProcess::SleepNoStats(0.0001f);
    Stepper->Step(.01);
    TestTrue(TEXT("Removed agent receives nothing"), Agent->Applied.IsEmpty());
    TestTrue(TEXT("Replacement rejects old lifetime"), Replacement->Applied.IsEmpty());
    while (Policy->GetBackgroundStats().CompletedRows < 2 && FPlatformTime::Seconds() < Deadline) FPlatformProcess::SleepNoStats(0.0001f);
    Stepper->Step(.02);
    TestEqual(TEXT("New lifetime receives its result"), Replacement->Applied.Num(), 1);
    Stepper->Reset();
    TestTrue(TEXT("Reinit"), Stepper->Init(Policy, {Interface}, {10.f}));
    Stepper->MaxActionAgeMs = 1;
    Stepper->Step(1);
    while (Policy->GetBackgroundStats().CompletedRows < 3 && FPlatformTime::Seconds() < Deadline) FPlatformProcess::SleepNoStats(0.0001f);
    Stepper->Step(1.02);
    TestTrue(TEXT("Over-age action held"), Agent->Applied.IsEmpty());
    Stepper->Reset();
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundCohortAccounting, "Schola.Inference.Background.Cohorts.ReferenceMix",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FBackgroundCohortAccounting::RunTest(const FString&)
{
    FNPCCohortScheduler Cohorts;
    TArray<int32> Due; Due.Init(0,600);
    for (int32 I = 0; I < 10000; ++I)
    {
        const int32 Every = I < 1000 ? 6 : I < 4000 ? 12 : 60;
        const float Period = float(Every) / 60.f;
        const int32 Phase = Cohorts.Acquire(Period, ENPCPhaseLayout::ModelAware, I);
        for (int32 Frame = Phase; Frame < 600; Frame += Every) ++Due[Frame];
    }
    int32 Total = 0;
    for (int32 Count : Due) { Total += Count; TestTrue(TEXT("517 +/- 1 due per frame"), Count >= 516 && Count <= 518); }
    TestEqual(TEXT("Ten-second decisions"), Total,310000);
    return true;
}
#endif

#if WITH_DEV_AUTOMATION_TESTS
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundStepperDelays, "Schola.Inference.Background.Stepper.DelayedFrames",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FBackgroundStepperDelays::RunTest(const FString&)
{
    UBackgroundAdapterTestProvider* Provider; auto Config = AdapterConfig(Provider);
    auto* Policy = NewObject<UNPCInferencePolicy>(); Policy->Configure(Config); Policy->Init(AdapterDefinition());
    auto* Stepper = NewObject<UBackgroundStepper>(); Stepper->MaxActionAgeMs = 1000;
    for (int32 Frames = 1; Frames <= 10; ++Frames)
    {
        auto* Agent = NewObject<USequenceTestAgent>();
        Stepper->Init(Policy, {TScriptInterface<IAgent>(Agent)}, {1.f});
        const int64 Before = Policy->GetBackgroundStats().CompletedRows;
        Provider->Backend->SetGateOpen(false);
        Stepper->Step(0);
        TestTrue(TEXT("Delayed call running"), Provider->Backend->WaitForBlockedCalls(1));
        for (int32 Frame = 1; Frame <= Frames; ++Frame) Stepper->Step(double(Frame)/60.0);
        TestTrue(TEXT("No application before completion"), Agent->Applied.IsEmpty());
        Provider->Backend->SetGateOpen(true);
        const double Deadline = FPlatformTime::Seconds()+5;
        while (Policy->GetBackgroundStats().CompletedRows < Before+1 && FPlatformTime::Seconds()<Deadline) FPlatformProcess::SleepNoStats(.0001f);
        Stepper->Step(double(Frames)/60.0+.001);
        TestEqual(TEXT("One valid delayed application"), Agent->Applied.Num(),1);
        Stepper->Reset();
    }
    return true;
}
#endif

#if WITH_DEV_AUTOMATION_TESTS
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundAutoTickTest, "Schola.Inference.Background.Stepper.AutoTickAndTeardown",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FBackgroundAutoTickTest::RunTest(const FString&)
{
    UWorld* World = UWorld::CreateWorld(EWorldType::Game,false);
    UBackgroundAdapterTestProvider* Provider; auto Config = AdapterConfig(Provider);
    auto* Policy = NewObject<UNPCInferencePolicy>(World); Policy->Configure(Config); Policy->Init(AdapterDefinition());
    auto* Agent = NewObject<USequenceTestAgent>(World);
    auto* Stepper = NewObject<UBackgroundStepper>(World); Stepper->bAutoTick=true;
    TestTrue(TEXT("Init auto tick"),Stepper->Init(Policy,{TScriptInterface<IAgent>(Agent)},{1.f}));
    auto* Registry = World->GetSubsystem<UNPCBackgroundInferenceSubsystem>();
    Registry->Tick(1.f/60);
    const double Deadline = FPlatformTime::Seconds()+5;
    while(Policy->GetBackgroundStats().CompletedRows<1 && FPlatformTime::Seconds()<Deadline)FPlatformProcess::SleepNoStats(.0001f);
    Registry->Tick(1.f/60);
    TestEqual(TEXT("Auto tick delivers"),Agent->Applied.Num(),1);
    Stepper->RequestDecisionNow(0); Registry->Tick(1.f/60);
    World->DestroyWorld(false);
    const int32 Before = Agent->Applied.Num();
    Stepper->Step(FPlatformTime::Seconds());
    TestEqual(TEXT("No application after world teardown"),Agent->Applied.Num(),Before);
    return true;
}
#endif

#if WITH_DEV_AUTOMATION_TESTS
namespace
{
class FReverseAdapterBackend;
class FReverseAdapterContext : public IBatchInferenceContext
{
public:
    explicit FReverseAdapterContext(TSharedRef<FReverseAdapterBackend> InDriver) : Driver(InDriver) {}
    virtual bool RunBatch(const float* Observation,int32 Rows,float* Action) override;
    TSharedRef<FReverseAdapterBackend> Driver;
};
class FReverseAdapterBackend : public IBatchInferenceBackend, public TSharedFromThis<FReverseAdapterBackend>
{
public:
    FReverseAdapterBackend() { for(auto& Gate:Gates)Gate=FPlatformProcess::GetSynchEventFromPool(true); }
    ~FReverseAdapterBackend() { for(auto* Gate:Gates)FPlatformProcess::ReturnSynchEventToPool(Gate); }
    FEvent* Gates[2]; std::atomic<uint32> Started{0};
    virtual int32 GetObservationSize() const override { return 1; }
    virtual int32 GetActionSize() const override { return 1; }
    virtual int32 GetMaxBatchSize() const override { return 1; }
    virtual FString GetName() const override { return TEXT("ReverseAdapter"); }
    virtual TUniquePtr<IBatchInferenceContext> CreateContext() override { return MakeUnique<FReverseAdapterContext>(AsShared()); }
    bool WaitStarted(uint32 Mask)
    {
        const double Deadline=FPlatformTime::Seconds()+5;
        while((Started.load()&Mask)!=Mask && FPlatformTime::Seconds()<Deadline)FPlatformProcess::SleepNoStats(.0001f);
        return (Started.load()&Mask)==Mask;
    }
};
bool FReverseAdapterContext::RunBatch(const float* Observation,int32 Rows,float* Action)
{
    const int32 Index=int32(Observation[0])-1;
    if(Rows!=1 || Index<0 || Index>1)return false;
    Driver->Started.fetch_or(1u<<Index);Driver->Gates[Index]->Wait();Action[0]=2*Observation[0];return true;
}
}
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundReverseCompletionTest,"Schola.Inference.Background.Stepper.ReversedCompletion",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FBackgroundReverseCompletionTest::RunTest(const FString&)
{
    UBackgroundAdapterTestProvider* Provider;auto Config=AdapterConfig(Provider);
    auto Backend=MakeShared<FReverseAdapterBackend>();Provider->GenericBackend=Backend;
    Config.WorkerCount=2;Config.MaxBatchSize=1;
    auto* Policy=NewObject<UNPCInferencePolicy>();Policy->Configure(Config);Policy->Init(AdapterDefinition());
    auto* Agent=NewObject<USequenceTestAgent>();auto* Stepper=NewObject<UBackgroundStepper>();Stepper->MaxActionAgeMs=1000;
    Stepper->Init(Policy,{TScriptInterface<IAgent>(Agent)},{1.f});
    Stepper->Step(0);TestTrue(TEXT("First observation executing"),Backend->WaitStarted(1));
    Stepper->RequestDecisionNow(0);Stepper->Step(.01);TestTrue(TEXT("Second observation executing"),Backend->WaitStarted(3));
    Backend->Gates[1]->Trigger();
    double Deadline=FPlatformTime::Seconds()+5;
    while(Policy->GetBackgroundStats().CompletedRows<1 && FPlatformTime::Seconds()<Deadline)FPlatformProcess::SleepNoStats(.0001f);
    Stepper->Step(.02);
    TestTrue(TEXT("Newer completion applied first"),Agent->Applied==TArray<float>{4});
    Backend->Gates[0]->Trigger();
    while(Policy->GetBackgroundStats().CompletedRows<2 && FPlatformTime::Seconds()<Deadline)FPlatformProcess::SleepNoStats(.0001f);
    Stepper->Step(.03);
    TestTrue(TEXT("Older completion cannot overwrite newest"),Agent->Applied==TArray<float>{4});
    Stepper->Reset();
    return true;
}
#endif

#if WITH_DEV_AUTOMATION_TESTS
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundConcurrentAsyncTest,"Schola.Inference.Background.Policy.AsyncDoesNotWaitForSync",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FBackgroundConcurrentAsyncTest::RunTest(const FString&)
{
    UBackgroundAdapterTestProvider* Provider;auto Config=AdapterConfig(Provider);
    auto* Policy=NewObject<UNPCInferencePolicy>();Policy->Configure(Config);Policy->Init(AdapterDefinition());
    Provider->Backend->SetGateOpen(false);
    auto Sync=Async(EAsyncExecution::ThreadPool,[Policy]
    {
        auto Out=AdapterPoints(99);
        return Policy->BatchedThink(AdapterPoints(3),Out) && Out[0].Get<FBoxPoint>().Values[0]==6;
    });
    TestTrue(TEXT("Synchronous worker is waiting"),Provider->Backend->WaitForBlockedCalls(1));
    auto Release=Async(EAsyncExecution::ThreadPool,[Backend=Provider->Backend]
    { FPlatformProcess::SleepNoStats(.1f);Backend->SetGateOpen(true); });
    FNPCBackgroundRequest Request;
    const double Start=FPlatformTime::Seconds();
    const auto Status=Policy->SubmitBackground(AdapterPoints(4),{},Request);
    TestTrue(TEXT("Async admission does not wait for backend"),FPlatformTime::Seconds()-Start<.05);
    TestTrue(TEXT("Concurrent async request admitted"),Status==ENPCBackgroundStatus::Pending);
    Release.Get();TestTrue(TEXT("Synchronous caller completed"),Sync.Get());
    const double Deadline=FPlatformTime::Seconds()+5;
    while(Policy->GetBackgroundStatus(Request)==ENPCBackgroundStatus::Pending && FPlatformTime::Seconds()<Deadline)FPlatformProcess::SleepNoStats(.0001f);
    TArray<TInstancedStruct<FPoint>> Out;
    TestTrue(TEXT("Async completed"),Policy->ConsumeBackground(Request,Out)==ENPCBackgroundStatus::Completed);
    TestEqual(TEXT("Concurrent dispatch counter counts each backend call once"),int64(Policy->GetStats().BatchDispatches),Provider->Backend->GetCalls());
    return true;
}
#endif

#if WITH_DEV_AUTOMATION_TESTS
#include "UObject/StrongObjectPtr.h"
#include "UObject/GarbageCollection.h"
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundReconfigureCycles,"Schola.Inference.Background.Policy.Reconfigure100WithInFlightWork",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FBackgroundReconfigureCycles::RunTest(const FString&)
{
    UBackgroundAdapterTestProvider* Provider;auto Config=AdapterConfig(Provider);
    TStrongObjectPtr<UNPCInferencePolicy> Policy(NewObject<UNPCInferencePolicy>());
    TStrongObjectPtr<UBackgroundStepper> Stepper(NewObject<UBackgroundStepper>());
    TStrongObjectPtr<USequenceTestAgent> Agent(NewObject<USequenceTestAgent>());
    Policy->Configure(Config);Policy->Init(AdapterDefinition());
    for(int32 Cycle=0;Cycle<100;++Cycle)
    {
        TestTrue(TEXT("Initialize stepper"),Stepper->Init(Policy.Get(),{TScriptInterface<IAgent>(Agent.Get())},{1.f}));
        Provider->Backend->SetGateOpen(false);
        FNPCBackgroundRequest Old;
        TestTrue(TEXT("Submit before reconfigure"),Policy->SubmitBackground(AdapterPoints(1),{},Old)==ENPCBackgroundStatus::Pending);
        Stepper->Step(0);
        TestTrue(TEXT("Work in flight"),Provider->Backend->WaitForBlockedCalls(1));
        auto Release=Async(EAsyncExecution::ThreadPool,[Backend=Provider->Backend]
        { FPlatformProcess::SleepNoStats(.001f);Backend->SetGateOpen(true); });
        Policy->Configure(Config);Release.Get();
        TestTrue(TEXT("Old ticket invalidated"),Policy->GetBackgroundStatus(Old)==ENPCBackgroundStatus::Invalid);
        const int32 Before=Agent->Applied.Num();Stepper->Step(.01);
        TestEqual(TEXT("Old stepper binding cannot apply after Configure"),Agent->Applied.Num(),Before);
        Stepper->Reset();
        TestTrue(TEXT("Reinitialize policy"),Policy->Init(AdapterDefinition()));
        if(Cycle%10==9)CollectGarbage(RF_NoFlags);
    }
    Stepper->Reset();return true;
}
#endif

#if WITH_DEV_AUTOMATION_TESTS
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundSharedReconfigureTest,"Schola.Inference.Background.Policy.ReconfigureWhileBucketIsShared",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FBackgroundSharedReconfigureTest::RunTest(const FString&)
{
    auto* World=UWorld::CreateWorld(EWorldType::Game,false);
    UBackgroundAdapterTestProvider* Provider;auto Config=AdapterConfig(Provider);
    auto* Keeper=NewObject<UNPCInferencePolicy>(World);Keeper->Configure(Config);Keeper->Init(AdapterDefinition());
    auto* Policy=NewObject<UNPCInferencePolicy>(World);Policy->Configure(Config);Policy->Init(AdapterDefinition());
    TSharedPtr<FNPCBackgroundScheduler> Before,After;TSharedPtr<const FNPCRowCodec> Codec;uint32 OldEpoch=0,NewEpoch=0;
    Policy->GetBackgroundBinding(Before,Codec,OldEpoch);
    auto* Agent=NewObject<USequenceTestAgent>(World);auto* Stepper=NewObject<UBackgroundStepper>(World);
    TestTrue(TEXT("Initialize old stepper"),Stepper->Init(Policy,{TScriptInterface<IAgent>(Agent)},{1.f}));
    Provider->Backend->SetGateOpen(false);Stepper->Step(0);
    TestTrue(TEXT("Old stepper work in flight"),Provider->Backend->WaitForBlockedCalls(1));
    FNPCBackgroundRequest Ticket;Policy->SubmitBackground(AdapterPoints(8),{},Ticket);
    TestTrue(TEXT("Configure shared policy"),Policy->Configure(Config));
    TestTrue(TEXT("Reinitialize shared policy"),Policy->Init(AdapterDefinition()));
    Policy->GetBackgroundBinding(After,Codec,NewEpoch);
    TestTrue(TEXT("Keeper preserves shared scheduler"),Before==After);
    TestTrue(TEXT("Policy epoch still advances"),NewEpoch!=OldEpoch);
    TestTrue(TEXT("Old policy ticket invalid"),Policy->GetBackgroundStatus(Ticket)==ENPCBackgroundStatus::Invalid);
    Stepper->Step(.01);
    Provider->Backend->SetGateOpen(true);
    TArray<TInstancedStruct<FPoint>> Action;
    TestTrue(TEXT("Other shared policy still works"),Keeper->BatchedThink(AdapterPoints(2),Action));
    Stepper->Step(.02);
    TestTrue(TEXT("Old stepper cannot apply across shared reconfigure"),Agent->Applied.IsEmpty());
    TestTrue(TEXT("New stepper binding works"),Stepper->Init(Policy,{TScriptInterface<IAgent>(Agent)},{1.f}));
    Stepper->Step(1);
    const double Deadline=FPlatformTime::Seconds()+5;
    while(Policy->GetBackgroundStats().CompletedRows<2 && FPlatformTime::Seconds()<Deadline)FPlatformProcess::SleepNoStats(.0001f);
    Stepper->Step(1.01);
    TestTrue(TEXT("Only new observation is delivered"),Agent->Applied==TArray<float>{4});
    Stepper->Reset();World->DestroyWorld(false);return true;
}
#endif

#if WITH_DEV_AUTOMATION_TESTS
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundUrgentStartupTest,"Schola.Inference.Background.Stepper.UrgentBeforeFirstTickPreservesPhase",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FBackgroundUrgentStartupTest::RunTest(const FString&)
{
    UBackgroundAdapterTestProvider* Provider;auto Config=AdapterConfig(Provider);
    auto* Policy=NewObject<UNPCInferencePolicy>();Policy->Configure(Config);Policy->Init(AdapterDefinition());
    auto* A=NewObject<USequenceTestAgent>();auto* B=NewObject<USequenceTestAgent>();
    auto* Stepper=NewObject<UBackgroundStepper>();
    TestTrue(TEXT("Initialize two phased agents"),Stepper->Init(Policy,{TScriptInterface<IAgent>(A),TScriptInterface<IAgent>(B)},{1.f,1.f}));
    Stepper->RequestDecisionNow(1);Stepper->Step(0);
    TestEqual(TEXT("Pre-tick urgency is not overwritten by initial phasing"),Stepper->SubmittedRows,int64(2));
    auto WaitRows=[&](int64 Rows)
    {
        const double Deadline=FPlatformTime::Seconds()+5;
        while(Policy->GetBackgroundStats().CompletedRows<Rows && FPlatformTime::Seconds()<Deadline)FPlatformProcess::SleepNoStats(.0001f);
    };
    WaitRows(2);Stepper->Step(.01);
    TestTrue(TEXT("Urgent action applied"),B->Applied==TArray<float>{2});
    Stepper->Step(.02);
    TestEqual(TEXT("Urgency retains the original next periodic phase"),Stepper->SubmittedRows,int64(3));
    WaitRows(3);Stepper->Step(.03);
    TestTrue(TEXT("Periodic observation follows the urgent action"),B->Applied==TArray<float>({2,4}));
    Stepper->Reset();
    auto* FreshPolicy=NewObject<UNPCInferencePolicy>();FreshPolicy->Configure(Config);FreshPolicy->Init(AdapterDefinition());
    Stepper->Init(FreshPolicy,{TScriptInterface<IAgent>(A)},{1.f});
    Stepper->RequestDecisionNow(0);Stepper->Step(10);Stepper->Step(10.01);
    TestEqual(TEXT("All-urgent startup still primes the periodic epoch"),Stepper->SubmittedRows,int64(1));
    Stepper->Reset();return true;
}
#endif

#if WITH_DEV_AUTOMATION_TESTS
namespace
{
TArray<TInstancedStruct<FPoint>> AdapterRows(int32 Rows)
{
    TArray<TInstancedStruct<FPoint>> Out;
    for (int32 I = 0; I < Rows; ++I) Out.Add(TInstancedStruct<FPoint>::Make<FBoxPoint>(TArray<float>{float(I)}));
    return Out;
}
bool WaitForPolicy(UNPCInferencePolicy* Policy, FNPCBackgroundRequest Request)
{
    const double Deadline = FPlatformTime::Seconds() + 5;
    while (Policy->GetBackgroundStatus(Request) == ENPCBackgroundStatus::Pending && FPlatformTime::Seconds() < Deadline) FPlatformProcess::SleepNoStats(.0001f);
    return Policy->GetBackgroundStatus(Request) != ENPCBackgroundStatus::Pending;
}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundPendingRowCapTest,"Schola.Inference.Background.Policy.PendingRowCapCountsRows",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FBackgroundPendingRowCapTest::RunTest(const FString&)
{
    UBackgroundAdapterTestProvider* Provider; auto Config = AdapterConfig(Provider);
    Config.BackgroundSettings.MaxPendingRows = 4;
    auto* Policy = NewObject<UNPCInferencePolicy>(); Policy->Configure(Config);
    if (!TestTrue(TEXT("Init"), Policy->Init(AdapterDefinition()))) return false;
    TSharedPtr<FNPCBackgroundScheduler, ESPMode::ThreadSafe> Scheduler;
    TSharedPtr<const FNPCRowCodec, ESPMode::ThreadSafe> Codec; uint32 Generation;
    Policy->GetBackgroundBinding(Scheduler, Codec, Generation);
    TArray<TInstancedStruct<FPoint>> Out;
    FNPCBackgroundRequest Three, Two, One, Four;
    Provider->Backend->SetGateOpen(false);
    TestTrue(TEXT("Three rows admitted"), Policy->SubmitBackground(AdapterRows(3), {}, Three) == ENPCBackgroundStatus::Pending);
    TestTrue(TEXT("Backend blocked"), Provider->Backend->WaitForBlockedCalls(1));
    TestTrue(TEXT("Cancel"), Policy->CancelBackground(Three));
    Provider->Backend->SetGateOpen(true);
    const double Deadline = FPlatformTime::Seconds() + 5;
    while (Policy->GetBackgroundStats().PendingRows > 0 && FPlatformTime::Seconds() < Deadline) FPlatformProcess::SleepNoStats(.0001f);
    TestTrue(TEXT("Scheduler released the cancelled rows"), Scheduler->CanAccept(2));
    TestTrue(TEXT("Unconsumed cancelled rows still count against MaxPendingRows"), Policy->SubmitBackground(AdapterRows(2), {}, Two) == ENPCBackgroundStatus::Busy);
    TestTrue(TEXT("Busy reflects the rejected row count"), Policy->IsInferenceBusy());
    TestTrue(TEXT("The cap is in rows, not requests"), Policy->SubmitBackground(AdapterRows(1), {}, One) == ENPCBackgroundStatus::Pending);
    TestTrue(TEXT("One completes"), WaitForPolicy(Policy, One));
    TestTrue(TEXT("Consume one"), Policy->ConsumeBackground(One, Out) == ENPCBackgroundStatus::Completed);
    TestTrue(TEXT("Consume cancellation"), Policy->ConsumeBackground(Three, Out) == ENPCBackgroundStatus::Cancelled);
    TestFalse(TEXT("Consuming releases held rows"), Policy->IsInferenceBusy());
    TestTrue(TEXT("A full-cap request fits once rows are released"), Policy->SubmitBackground(AdapterRows(4), {}, Four) == ENPCBackgroundStatus::Pending);
    TestTrue(TEXT("Four completes"), WaitForPolicy(Policy, Four));
    TestTrue(TEXT("Consume four"), Policy->ConsumeBackground(Four, Out) == ENPCBackgroundStatus::Completed && Out.Num() == 4);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundSettingsPrecedenceTest,"Schola.Inference.Background.Policy.LegacyFieldsOverrideBackgroundSettings",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FBackgroundSettingsPrecedenceTest::RunTest(const FString&)
{
    // The conflict warning is logged once per process, so it may already have fired.
    AddExpectedMessagePlain(TEXT("take precedence"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, -1);
    UBackgroundAdapterTestProvider* Provider; auto Config = AdapterConfig(Provider);
    Config.WorkerCount = 1; Config.MaxBatchSize = 2;
    Config.BackgroundSettings.WorkerCount = 4; Config.BackgroundSettings.MaxBatchSize = 16;
    Config.BackgroundSettings.Dispatch = ENPCBackgroundDispatch::LeastLoaded;
    auto* Policy = NewObject<UNPCInferencePolicy>(); Policy->Configure(Config);
    if (!TestTrue(TEXT("Init"), Policy->Init(AdapterDefinition()))) return false;
    TestEqual(TEXT("WorkerCount comes from the legacy field"), Policy->GetBackgroundStats().WorkerBatches.Num(), 1);
    FNPCBackgroundRequest Request; TArray<TInstancedStruct<FPoint>> Out;
    TestTrue(TEXT("Submit"), Policy->SubmitBackground(AdapterRows(8), {}, Request) == ENPCBackgroundStatus::Pending);
    TestTrue(TEXT("Complete"), WaitForPolicy(Policy, Request));
    TestTrue(TEXT("Consume"), Policy->ConsumeBackground(Request, Out) == ENPCBackgroundStatus::Completed);
    TestEqual(TEXT("MaxBatchSize comes from the legacy field"), Provider->Backend->GetLargestCallRows(), 2);
    TestEqual(TEXT("Requested settings are left untouched"), Policy->GetConfig().BackgroundSettings.WorkerCount, 4);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundUnsharedBucketTest,"Schola.Inference.Background.Policy.UnsharedBucketWithProvider",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FBackgroundUnsharedBucketTest::RunTest(const FString&)
{
    UWorld* World = UWorld::CreateWorld(EWorldType::Game, false);
    auto* Registry = World->GetSubsystem<UNPCBackgroundInferenceSubsystem>();
    UBackgroundAdapterTestProvider* Provider; auto Config = AdapterConfig(Provider);
    auto* A = NewObject<UNPCInferencePolicy>(World); auto* B = NewObject<UNPCInferencePolicy>(World);
    auto Bind = [](UNPCInferencePolicy* Policy)
    {
        TSharedPtr<FNPCBackgroundScheduler, ESPMode::ThreadSafe> Scheduler;
        TSharedPtr<const FNPCRowCodec, ESPMode::ThreadSafe> Codec; uint32 Generation;
        Policy->GetBackgroundBinding(Scheduler, Codec, Generation);
        return Scheduler;
    };
    auto Roundtrip = [&](UNPCInferencePolicy* Policy)
    {
        FNPCBackgroundRequest Request; TArray<TInstancedStruct<FPoint>> Out;
        return Policy->SubmitBackground(AdapterPoints(5), {}, Request) == ENPCBackgroundStatus::Pending && WaitForPolicy(Policy, Request)
            && Policy->ConsumeBackground(Request, Out) == ENPCBackgroundStatus::Completed && Out[0].Get<FBoxPoint>().Values[0] == 10.f;
    };
    for (const bool bShare : {false, true})
    {
        Config.bShareBackgroundBucket = bShare;
        A->Configure(Config); B->Configure(Config);
        TestTrue(TEXT("Init A"), A->Init(AdapterDefinition()));
        TestTrue(TEXT("Init B"), B->Init(AdapterDefinition()));
        const FString Mode = bShare ? TEXT("Shared: ") : TEXT("Unshared: ");
        TestTrue(Mode + TEXT("scheduler identity"), (Bind(A) == Bind(B)) == bShare);
        TestEqual(Mode + TEXT("registered buckets"), Registry->GetAllBucketStats().Num(), bShare ? 1 : 0);
        TestTrue(Mode + TEXT("A completes"), Roundtrip(A));
        TestTrue(Mode + TEXT("B completes"), Roundtrip(B));
        if (bShare) TestEqual(TEXT("Bucket stats aggregate both policies"), Registry->GetAllBucketStats()[0].CompletedRows, int64(2));
    }
    World->DestroyWorld(false);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBackgroundReapplyHeldTest,"Schola.Inference.Background.Stepper.ReapplyHeldAction",
    EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
bool FBackgroundReapplyHeldTest::RunTest(const FString&)
{
    UBackgroundAdapterTestProvider* Provider; auto Config = AdapterConfig(Provider);
    auto* Policy = NewObject<UNPCInferencePolicy>(); Policy->Configure(Config); Policy->Init(AdapterDefinition());
    for (const bool bReapply : {false, true})
    {
        auto* Agent = NewObject<USequenceTestAgent>();
        auto* Stepper = NewObject<UBackgroundStepper>();
        Stepper->MaxActionAgeMs = 1000; Stepper->bReapplyHeldAction = bReapply;
        TestTrue(TEXT("Init"), Stepper->Init(Policy, {TScriptInterface<IAgent>(Agent)}, {1.f}));
        const int64 Before = Policy->GetBackgroundStats().CompletedRows;
        Stepper->Step(0);
        const double Deadline = FPlatformTime::Seconds() + 5;
        while (Policy->GetBackgroundStats().CompletedRows < Before + 1 && FPlatformTime::Seconds() < Deadline) FPlatformProcess::SleepNoStats(.0001f);
        Stepper->Step(.01);
        TestEqual(TEXT("Fresh result applied once"), Agent->Applied.Num(), 1);
        for (int32 Frame = 2; Frame <= 4; ++Frame) Stepper->Step(Frame * .01);
        TestEqual(bReapply ? TEXT("Held action re-applied every step") : TEXT("Held action not re-applied"), Agent->Applied.Num(), bReapply ? 4 : 1);
        TestTrue(TEXT("Re-applied value is the held action"), Agent->Applied.Last() == Agent->Applied[0]);
        Stepper->Reset();
    }
    return true;
}
#endif
