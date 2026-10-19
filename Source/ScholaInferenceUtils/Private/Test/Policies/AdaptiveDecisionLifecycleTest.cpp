// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "Misc/AutomationTest.h"
#include "AdaptiveDecisionTestPolicy.h"
#include "Common/InstancedStructUtils.h"
#include "Points/BoxPoint.h"
#include "Spaces/BoxSpace.h"
#include "UObject/UnrealType.h"
#include <limits>

namespace
{
FInteractionDefinition LifecycleDefinition()
{
    FInteractionDefinition Definition;
    Definition.ObsSpaceDefn.InitializeAs<FBoxSpace>(TArray<float>{-1, -1}, TArray<float>{1, 1});
    Definition.ActionSpaceDefn.InitializeAs<FBoxSpace>(TArray<float>{-1}, TArray<float>{1});
    return Definition;
}

TInstancedStruct<FPoint> LifecyclePoint(TArray<float> Values)
{
    TInstancedStruct<FPoint> Point;
    Point.InitializeAs<FBoxPoint>(Values);
    return Point;
}

UAdaptiveDecisionTestPolicy* ReadyPolicy()
{
    auto* Policy = NewObject<UAdaptiveDecisionTestPolicy>();
    Policy->Init(LifecycleDefinition());
    Policy->Start(FGuid::NewGuid());
    return Policy;
}

FAdaptiveDecision ResultFor(const FAdaptiveDecisionRequest& Request, int64 Version = 0)
{
    FAdaptiveDecision Result;
    Result.SessionId = Request.SessionId;
    Result.AgentId = Request.AgentId;
    Result.EpisodeId = Request.EpisodeId;
    Result.DecisionId = Request.DecisionId;
    Result.PolicyVersion = Version;
    Result.Action = LifecyclePoint({0.5f});
    return Result;
}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FAdaptiveDecisionRoundTripTest, "Schola.Inference.Adaptive.DecisionRoundTrip",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FAdaptiveDecisionRoundTripTest::RunTest(const FString& Parameters)
{
    auto* Policy = ReadyPolicy();
    auto Observation = LifecyclePoint({0.25f, -0.25f});
    int64 Ticket = -42;
    TestTrue(TEXT("Request dispatch returns immediately without a response"), Policy->RequestDecision(Observation, Ticket, 2.0f));
    TestEqual(TEXT("Zero is a valid first ticket"), Ticket, int64(0));
    TestTrue(TEXT("Pending inference is busy"), Policy->IsInferenceBusy());
    Observation.GetMutable<FBoxPoint>().Values[0] = 0.9f;
    TestEqual(TEXT("Dispatched observation is an owned copy"), Policy->Sent.Observation.Get<FBoxPoint>().Values[0], 0.25f);
    FAdaptiveDecision Output;
    Output.DecisionId = 999;
    TestTrue(TEXT("Pending poll returns promptly"), Policy->TryGetDecision(Ticket, Output) == EAdaptiveRequestStatus::Pending);
    TestEqual(TEXT("Pending poll preserves output"), Output.DecisionId, int64(999));
    int64 OtherTicket = -7;
    TestFalse(TEXT("Only one pending request allowed"), Policy->RequestDecision(Observation, OtherTicket));
    TestEqual(TEXT("Rejected request preserves caller ticket"), OtherTicket, int64(-7));
    TestEqual(TEXT("No extra dispatch on capacity rejection"), Policy->DispatchCount, 1);
    TestTrue(TEXT("Unknown ticket is distinct"), Policy->TryGetDecision(123, Output) == EAdaptiveRequestStatus::Unknown);

    auto Result = ResultFor(Policy->Sent, 7);
    Policy->Now += 0.5;
    TestTrue(TEXT("Delayed action accepted"), Policy->Deliver(Result));
    TestFalse(TEXT("Completed inference is no longer busy"), Policy->IsInferenceBusy());
    Result.Action.GetMutable<FBoxPoint>().Values[0] = -0.5f;
    TestTrue(TEXT("Ready result available"), Policy->TryGetDecision(Ticket, Output) == EAdaptiveRequestStatus::Ready);
    TestEqual(TEXT("Result is copied from completion"), Output.Action.Get<FBoxPoint>().Values[0], 0.5f);
    Output.Action.GetMutable<FBoxPoint>().Values[0] = -0.9f;
    Policy->TryGetDecision(Ticket, Output);
    TestEqual(TEXT("Caller mutation cannot modify retained result"), Output.Action.Get<FBoxPoint>().Values[0], 0.5f);
    TestFalse(TEXT("Duplicate completion rejected"), Policy->Deliver(Result));
    TestTrue(TEXT("Next request retires the previous ready result"), Policy->RequestDecision(Observation, OtherTicket));
    TestTrue(TEXT("Retired ticket is unknown"), Policy->TryGetDecision(Ticket, Output) == EAdaptiveRequestStatus::Unknown);
    TestTrue(TEXT("Policy version may go backwards across valid decisions"), Policy->Deliver(ResultFor(Policy->Sent, 0)));
    Policy->TryGetDecision(OtherTicket, Output);
    TestEqual(TEXT("Original policy version retained"), Output.PolicyVersion, int64(0));

    Policy->bCompleteImmediately = true;
    Policy->ImmediateResult = ResultFor(Policy->Sent);
    TestTrue(TEXT("Immediate completion during dispatch supported"), Policy->RequestDecision(Observation, Ticket));
    TestTrue(TEXT("Immediate answer is retrievable"), Policy->TryGetDecision(Ticket, Output) == EAdaptiveRequestStatus::Ready);
    // The IPolicy contract never turns into a blocking network call, even for a ready session.
    auto Action = LifecyclePoint({0.9f});
    TArray<TInstancedStruct<FPoint>> Actions = {Action};
    TestFalse(TEXT("Think remains unavailable"), Policy->Think(Observation, Action));
    TestFalse(TEXT("BatchedThink remains unavailable"), Policy->BatchedThink({Observation}, Actions));
    TestTrue(TEXT("Immediate outputs preserved"), Action == LifecyclePoint({0.9f}) && Actions.Num() == 1 && Actions[0] == Action);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FAdaptiveDecisionValidationTest, "Schola.Inference.Adaptive.DecisionValidation",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FAdaptiveDecisionValidationTest::RunTest(const FString& Parameters)
{
    auto* Policy = NewObject<UAdaptiveDecisionTestPolicy>();
    int64 Ticket = -99;
    const auto Observation = LifecyclePoint({0, 0});
    TestFalse(TEXT("Uninitialized requests rejected"), Policy->RequestDecision(Observation, Ticket));
    TestFalse(TEXT("Uninitialized handshake rejected"), Policy->Start(FGuid::NewGuid()));
    Policy->Init(LifecycleDefinition());
    TestFalse(TEXT("Local initialization is not remote readiness"), Policy->RequestDecision(Observation, Ticket));
    TestFalse(TEXT("Invalid session rejected"), Policy->Start(FGuid()));
    TestFalse(TEXT("Negative agent rejected"), Policy->Start(FGuid::NewGuid(), -1));
    TestFalse(TEXT("Negative episode rejected"), Policy->Start(FGuid::NewGuid(), 0, -1));
    TestTrue(TEXT("Confirmed session accepted"), Policy->Start(FGuid::NewGuid()));
    TestFalse(TEXT("Cannot replace an active session without reinitializing"), Policy->Start(FGuid::NewGuid()));
    const float NaN = std::numeric_limits<float>::quiet_NaN();
    const float Infinity = std::numeric_limits<float>::infinity();
    for (float Timeout : {0.0f, -1.0f, NaN, Infinity})
    {
        TestFalse(TEXT("Invalid timeout rejected"), Policy->RequestDecision(Observation, Ticket, Timeout));
    }
    auto BadShape = Observation;
    BadShape.GetMutable<FBoxPoint>().Shape = {1, 2};
    TInstancedStruct<FPoint> WrongType;
    ToUntypedInstancedStruct(WrongType).InitializeAs<FInteractionDefinition>();
    for (const auto& BadObservation : TArray<TInstancedStruct<FPoint>>{{}, WrongType, LifecyclePoint({0}),
        LifecyclePoint({2, 0}), LifecyclePoint({NaN, 0}), LifecyclePoint({Infinity, 0}), BadShape})
    {
        TestFalse(TEXT("Malformed observation rejected"), Policy->RequestDecision(BadObservation, Ticket));
        TestTrue(TEXT("Observation failure explained"), Policy->GetLastError().Contains(TEXT("Observation")));
    }
    TestEqual(TEXT("All pre-dispatch failures preserve ticket"), Ticket, int64(-99));
    TestEqual(TEXT("Invalid requests never dispatched"), Policy->DispatchCount, 0);
    auto Unbounded = LifecycleDefinition();
    Unbounded.ObsSpaceDefn.GetMutable<FBoxSpace>().Dimensions[0] = FBoxSpaceDimension::Unbounded();
    Policy->Init(Unbounded);
    Policy->Start(FGuid::NewGuid());
    TestFalse(TEXT("Unbounded observations must still be finite"), Policy->RequestDecision(LifecyclePoint({NaN, 0}), Ticket));

    for (const auto& BadAction : TArray<TInstancedStruct<FPoint>>{{}, WrongType, LifecyclePoint({0, 0}),
        LifecyclePoint({2}), LifecyclePoint({NaN}), LifecyclePoint({Infinity})})
    {
        Policy->RequestDecision(Observation, Ticket);
        auto Result = ResultFor(Policy->Sent);
        Result.Action = BadAction;
        TestFalse(TEXT("Malformed action rejected"), Policy->Deliver(Result));
        FAdaptiveDecision Output;
        Output.DecisionId = 99;
        TestTrue(TEXT("Matching invalid result fails its ticket"), Policy->TryGetDecision(Ticket, Output) == EAdaptiveRequestStatus::Failed);
        TestEqual(TEXT("Failed result does not release action"), Output.DecisionId, int64(99));
        TestTrue(TEXT("Action failure explained per ticket"), Policy->GetDecisionError(Ticket).Contains(TEXT("Action")));
    }
    Policy->RequestDecision(Observation, Ticket);
    TestFalse(TEXT("Unassigned policy version rejected"), Policy->Deliver(ResultFor(Policy->Sent, -1)));
    Policy->RequestDecision(Observation, Ticket);
    auto WrongShape = ResultFor(Policy->Sent);
    WrongShape.Action.GetMutable<FBoxPoint>().Shape = {1, 1};
    TestFalse(TEXT("Malformed action shape rejected"), Policy->Deliver(WrongShape));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FAdaptiveDecisionIdentityTest, "Schola.Inference.Adaptive.DecisionIdentity",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FAdaptiveDecisionIdentityTest::RunTest(const FString& Parameters)
{
    auto* Policy = ReadyPolicy();
    int64 Ticket;
    Policy->RequestDecision(LifecyclePoint({0, 0}), Ticket);
    const auto Expected = ResultFor(Policy->Sent);
    for (int32 Field = 0; Field < 4; ++Field)
    {
        auto Wrong = Expected;
        if (Field == 0) { Wrong.SessionId = FGuid::NewGuid(); }
        if (Field == 1) { ++Wrong.AgentId; }
        if (Field == 2) { ++Wrong.EpisodeId; }
        if (Field == 3) { ++Wrong.DecisionId; }
        TestFalse(TEXT("Wrong identity cannot release action"), Policy->Deliver(Wrong));
        auto WrongFailure = Policy->Sent;
        WrongFailure.SessionId = Wrong.SessionId;
        WrongFailure.AgentId = Wrong.AgentId;
        WrongFailure.EpisodeId = Wrong.EpisodeId;
        WrongFailure.DecisionId = Wrong.DecisionId;
        TestFalse(TEXT("Wrong failure cannot fail current request"), Policy->Reject(WrongFailure, TEXT("Unrelated failure")));
        FAdaptiveDecision Output;
        TestTrue(TEXT("Matching request stays pending"), Policy->TryGetDecision(Ticket, Output) == EAdaptiveRequestStatus::Pending);
    }
    TestTrue(TEXT("Matching response still accepted"), Policy->Deliver(Expected));
    TestFalse(TEXT("Late failure cannot overwrite ready result"), Policy->Reject(Policy->Sent, TEXT("Late failure")));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FAdaptiveDecisionExpiryTest, "Schola.Inference.Adaptive.DecisionExpiryAndFailures",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FAdaptiveDecisionExpiryTest::RunTest(const FString& Parameters)
{
    auto* Policy = ReadyPolicy();
    const auto Observation = LifecyclePoint({0, 0});
    int64 Ticket;
    FAdaptiveDecision Output;
    Output.DecisionId = 88;
    Policy->RequestDecision(Observation, Ticket, 2.0f);
    const auto Late = ResultFor(Policy->Sent);
    Policy->Now += 2.0;
    TestFalse(TEXT("Busy clears at deadline without polling"), Policy->IsInferenceBusy());
    TestFalse(TEXT("Late answer rejected even before a poll"), Policy->Deliver(Late));
    TestTrue(TEXT("Exact deadline expires pending answer"), Policy->TryGetDecision(Ticket, Output) == EAdaptiveRequestStatus::Expired);
    TestEqual(TEXT("Expired poll preserves output"), Output.DecisionId, int64(88));
    TestFalse(TEXT("Expired action cannot be cancelled or revived"), Policy->CancelDecision(Ticket));
    TestTrue(TEXT("Expiry explained"), Policy->GetDecisionError(Ticket).Contains(TEXT("expired")));

    Policy->RequestDecision(Observation, Ticket, 2.0f);
    Policy->Deliver(ResultFor(Policy->Sent));
    Policy->Now += 2.0;
    TestTrue(TEXT("Ready actions also expire if gameplay waits too long"), Policy->TryGetDecision(Ticket, Output) == EAdaptiveRequestStatus::Expired);

    Policy->RequestDecision(Observation, Ticket);
    TestTrue(TEXT("Service failure matches the request"), Policy->Reject(Policy->Sent, TEXT("Service unavailable")));
    TestTrue(TEXT("Failure observable"), Policy->TryGetDecision(Ticket, Output) == EAdaptiveRequestStatus::Failed);
    TestEqual(TEXT("Useful service error retained"), Policy->GetDecisionError(Ticket), FString(TEXT("Service unavailable")));
    TestTrue(TEXT("Unknown error is empty"), Policy->GetDecisionError(-1).IsEmpty());
    TestFalse(TEXT("Failure leaves no work busy"), Policy->IsInferenceBusy());
    Policy->bAcceptDispatch = false;
    int64 Unchanged = -77;
    TestFalse(TEXT("Rejected dispatch fails promptly"), Policy->RequestDecision(Observation, Unchanged));
    TestEqual(TEXT("Rejected dispatch preserves caller ticket"), Unchanged, int64(-77));
    TestFalse(TEXT("Rejected dispatch releases capacity"), Policy->IsInferenceBusy());
    Policy->bAcceptDispatch = true;
    TestTrue(TEXT("Can retry after failure"), Policy->RequestDecision(Observation, Ticket));
    TestTrue(TEXT("Successful retry clears last error"), Policy->GetLastError().IsEmpty());
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FAdaptiveDecisionLifecycleTest, "Schola.Inference.Adaptive.DecisionLifecycle",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FAdaptiveDecisionLifecycleTest::RunTest(const FString& Parameters)
{
    auto* Policy = ReadyPolicy();
    const auto Observation = LifecyclePoint({0, 0});
    FAdaptiveDecision Output;
    Output.DecisionId = 77;
    int64 Ticket;
    Policy->RequestDecision(Observation, Ticket);
    auto OldRequest = Policy->Sent;
    auto* Duplicate = DuplicateObject<UAdaptiveDecisionTestPolicy>(Policy, GetTransientPackage());
    TestTrue(TEXT("Duplicated configuration requires a new handshake"), Duplicate->GetSessionState() == EAdaptiveSessionState::Initialized);
    TestFalse(TEXT("Duplicate cannot inherit pending work"), Duplicate->IsInferenceBusy());
    TestTrue(TEXT("Duplicate has no inherited ticket"), Duplicate->TryGetDecision(Ticket, Output) == EAdaptiveRequestStatus::Unknown);
    TestFalse(TEXT("Wrong cancellation cannot stop work"), Policy->CancelDecision(Ticket + 1));
    TestTrue(TEXT("Pending request cancelled"), Policy->CancelDecision(Ticket));
    TestTrue(TEXT("Cancellation is observable"), Policy->TryGetDecision(Ticket, Output) == EAdaptiveRequestStatus::Cancelled);
    TestFalse(TEXT("Cancelled answer rejected"), Policy->Deliver(ResultFor(OldRequest)));
    TestEqual(TEXT("Cancelled poll preserves output"), Output.DecisionId, int64(77));

    Policy->RequestDecision(Observation, Ticket);
    OldRequest = Policy->Sent;
    TestTrue(TEXT("Reset advances encounter"), Policy->ResetEpisode());
    TestTrue(TEXT("Reset retains learning session readiness"), Policy->GetSessionState() == EAdaptiveSessionState::Ready);
    TestTrue(TEXT("Reset cancels pending answer"), Policy->TryGetDecision(Ticket, Output) == EAdaptiveRequestStatus::Cancelled);
    Policy->RequestDecision(Observation, Ticket);
    TestTrue(TEXT("Session retained across reset"), Policy->Sent.SessionId == OldRequest.SessionId);
    TestEqual(TEXT("Episode advances"), Policy->Sent.EpisodeId, OldRequest.EpisodeId + 1);
    TestTrue(TEXT("Tickets never reused"), Ticket > OldRequest.DecisionId);
    TestFalse(TEXT("Old encounter response cannot affect new request"), Policy->Deliver(ResultFor(OldRequest)));
    TestFalse(TEXT("Old encounter failure cannot affect new request"), Policy->Reject(OldRequest, TEXT("Old failure")));
    TestFalse(TEXT("Invalid reinitialization rejected"), Policy->Init(FInteractionDefinition()));
    TestTrue(TEXT("Invalid reinitialization preserves in-flight work"), Policy->TryGetDecision(Ticket, Output) == EAdaptiveRequestStatus::Pending);
    TestTrue(TEXT("Current response accepted"), Policy->Deliver(ResultFor(Policy->Sent)));
    Policy->ResetEpisode();
    TestTrue(TEXT("Reset also cancels a ready action"), Policy->TryGetDecision(Ticket, Output) == EAdaptiveRequestStatus::Cancelled);

    Policy->RequestDecision(Observation, Ticket);
    OldRequest = Policy->Sent;
    Policy->Init(LifecycleDefinition());
    TestTrue(TEXT("Successful reinit requires new handshake"), Policy->GetSessionState() == EAdaptiveSessionState::Initialized);
    TestFalse(TEXT("Pre-reinit callback rejected"), Policy->Deliver(ResultFor(OldRequest)));
    Policy->Start(FGuid::NewGuid());
    Policy->RequestDecision(Observation, Ticket);
    TestTrue(TEXT("Tickets also increase across sessions"), Ticket > OldRequest.DecisionId);
    OldRequest = Policy->Sent;
    Policy->Shutdown();
    Policy->Shutdown();
    TestTrue(TEXT("Shutdown is idempotent"), Policy->GetSessionState() == EAdaptiveSessionState::Stopped);
    TestFalse(TEXT("Shutdown rejects requests"), Policy->RequestDecision(Observation, Ticket));
    TestFalse(TEXT("Shutdown rejects callbacks"), Policy->Deliver(ResultFor(OldRequest)));
    TestFalse(TEXT("Restart requires Init"), Policy->Start(FGuid::NewGuid()));
    TestFalse(TEXT("Stopped session cannot reset"), Policy->ResetEpisode());

    Policy->Init(LifecycleDefinition());
    Policy->Start(FGuid::NewGuid(), 0, MAX_int64);
    TestFalse(TEXT("Episode ID overflow rejected"), Policy->ResetEpisode());
    Policy->RequestDecision(Observation, Ticket);
    OldRequest = Policy->Sent;
    Policy->ConditionalBeginDestroy();
    TestFalse(TEXT("Destruction invalidates outstanding action"), Policy->Deliver(ResultFor(OldRequest)));

    auto* DestroyedPolicy = ReadyPolicy();
    DestroyedPolicy->RequestDecision(Observation, Ticket);
    const auto DelayedResult = ResultFor(DestroyedPolicy->Sent);
    TWeakObjectPtr<UAdaptiveDecisionTestPolicy> WeakPolicy(DestroyedPolicy);
    bool bCallbackTouchedObject = false;
    // Same weak-reference pattern required of the future client's game-thread completion.
    auto DelayedCallback = [WeakPolicy, DelayedResult, &bCallbackTouchedObject]()
    {
        if (auto* LivePolicy = WeakPolicy.Get())
        {
            bCallbackTouchedObject = true;
            LivePolicy->Deliver(DelayedResult);
        }
    };
    DestroyedPolicy->MarkAsGarbage();
    CollectGarbage(RF_NoFlags);
    DelayedCallback();
    TestFalse(TEXT("Completion skips a collected policy via weak ownership"), bCallbackTouchedObject);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FAdaptiveDecisionBlueprintTest, "Schola.Inference.Adaptive.DecisionBlueprintAPI",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FAdaptiveDecisionBlueprintTest::RunTest(const FString& Parameters)
{
    auto* Policy = ReadyPolicy();
    for (const TCHAR* Name : {TEXT("RequestDecision"), TEXT("TryGetDecision"), TEXT("CancelDecision"),
        TEXT("ResetEpisode"), TEXT("Shutdown"), TEXT("GetDecisionError")})
    {
        UFunction* Function = Policy->FindFunction(FName(Name));
        TestTrue(Name, Function && Function->HasAnyFunctionFlags(FUNC_BlueprintCallable));
    }
    TestFalse(TEXT("Blueprint cannot manufacture readiness"), Policy->FindFunction(TEXT("StartSession")) != nullptr);
    TestTrue(TEXT("Request record available to Blueprint"), FAdaptiveDecisionRequest::StaticStruct()->HasMetaData(TEXT("BlueprintType")));
    for (TFieldIterator<FProperty> Property(FAdaptiveDecisionRequest::StaticStruct()); Property; ++Property)
    {
        TestTrue(Property->GetName(), Property->HasAnyPropertyFlags(CPF_BlueprintVisible));
    }
    struct FRequestParameters
    {
        TInstancedStruct<FPoint> Observation;
        int64 OutDecisionId = -1;
        float TimeoutSeconds = 1.0f;
        bool ReturnValue = false;
    } Args;
    Args.Observation = LifecyclePoint({0, 0});
    ToUntypedInstancedStruct(Args.Observation).InitializeAs<FInteractionDefinition>();
    Policy->ProcessEvent(Policy->FindFunctionChecked(TEXT("RequestDecision")), &Args);
    TestFalse(TEXT("Malformed Blueprint input rejected safely"), Args.ReturnValue);
    Args.Observation = LifecyclePoint({0, 0});
    Policy->ProcessEvent(Policy->FindFunctionChecked(TEXT("RequestDecision")), &Args);
    TestTrue(TEXT("Blueprint dispatch creates ticket"), Args.ReturnValue && Args.OutDecisionId >= 0);
    return true;
}
