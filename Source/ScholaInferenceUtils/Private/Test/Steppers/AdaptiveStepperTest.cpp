// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "Misc/AutomationTest.h"
#include "AdaptiveStepperTestTypes.h"
#include "GatedStepperTestTypes.h"
#include "Steppers/AdaptiveStepper.h"
#include "Test/Policies/AdaptiveDecisionTestPolicy.h"

namespace
{
TArray<TScriptInterface<IAgent>> MakeAgents(int32 Count, TArray<UAdaptiveStepperTestAgent*>& OutAgents)
{
    TArray<TScriptInterface<IAgent>> Agents;
    for (int32 Index = 0; Index < Count; ++Index)
    {
        auto* Agent = NewObject<UAdaptiveStepperTestAgent>();
        OutAgents.Add(Agent);
        Agents.Add(TScriptInterface<IAgent>(Agent));
    }
    return Agents;
}

UAdaptiveDecisionTestPolicy* TestPolicy(UAdaptiveStepper* Stepper, int32 Index)
{
    return Cast<UAdaptiveDecisionTestPolicy>(Stepper->GetPolicy(Index));
}

/** The duplicated bindings only copy reflected state; give each one its controlled behavior and a session. */
void StartBindings(UAdaptiveStepper* Stepper, bool bImmediate = true)
{
    for (int32 Index = 0; Index < Stepper->GetNumAgents(); ++Index)
    {
        auto* Policy = TestPolicy(Stepper, Index);
        Policy->bCompleteImmediately = bImmediate;
        Policy->bHandleExperience = true;
        Policy->ImmediateResult.PolicyVersion = 3;
        Policy->ImmediateResult.Action.InitializeAs<FBoxPoint>(TArray<float>{0.5f});
        Policy->Start(FGuid::NewGuid(), Index, 0);
    }
}

UAdaptiveStepper* MakeStepper(int32 Count, TArray<UAdaptiveStepperTestAgent*>& OutAgents, UAdaptiveDecisionTestPolicy*& OutTemplate)
{
    auto* Stepper = NewObject<UAdaptiveStepper>();
    OutTemplate = NewObject<UAdaptiveDecisionTestPolicy>();
    Stepper->Init(MakeAgents(Count, OutAgents), TScriptInterface<IPolicy>(OutTemplate));
    return Stepper;
}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FAdaptiveStepperInitRejectsTest, "Schola.Inference.Adaptive.Stepper.InitRejects",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FAdaptiveStepperInitRejectsTest::RunTest(const FString& Parameters)
{
    auto* Stepper = NewObject<UAdaptiveStepper>();
    TArray<UAdaptiveStepperTestAgent*> AdaptiveAgents;
    auto Agents = MakeAgents(1, AdaptiveAgents);
    auto* Template = NewObject<UAdaptiveDecisionTestPolicy>();
    auto* PlainPolicy = NewObject<UGatedTestPolicy>();
    TestFalse(TEXT("A non-adaptive policy is rejected"), Stepper->Init(Agents, TScriptInterface<IPolicy>(PlainPolicy)));
    TestFalse(TEXT("No agents is rejected"), Stepper->Init(TArray<TScriptInterface<IAgent>>(), TScriptInterface<IPolicy>(Template)));
    TArray<TScriptInterface<IAgent>> PlainAgents{TScriptInterface<IAgent>(NewObject<USequenceTestAgent>())};
    TestFalse(TEXT("An agent without IAdaptiveAgent is rejected"), Stepper->Init(PlainAgents, TScriptInterface<IPolicy>(Template)));
    TestFalse(TEXT("Failure is reported"), Stepper->GetLastError().IsEmpty());
    TestEqual(TEXT("A failed Init keeps no bindings"), Stepper->GetNumAgents(), 0);
    TestTrue(TEXT("Valid agents and policy succeed"), Stepper->Init(Agents, TScriptInterface<IPolicy>(Template)));
    TestTrue(TEXT("Success clears the error"), Stepper->GetLastError().IsEmpty());
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FAdaptiveStepperBindingPerAgentTest, "Schola.Inference.Adaptive.Stepper.BindingPerAgent",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FAdaptiveStepperBindingPerAgentTest::RunTest(const FString& Parameters)
{
    TArray<UAdaptiveStepperTestAgent*> Agents;
    UAdaptiveDecisionTestPolicy* Template = nullptr;
    auto* Stepper = MakeStepper(3, Agents, Template);
    TestEqual(TEXT("One binding per agent"), Stepper->GetNumAgents(), 3);
    TestFalse(TEXT("The template itself is never bound"), Template->IsInitialized());
    TSet<UAdaptiveNPCPolicy*> Distinct;
    for (int32 Index = 0; Index < 3; ++Index)
    {
        UAdaptiveNPCPolicy* Policy = Stepper->GetPolicy(Index);
        TestNotNull(TEXT("Binding exists"), Policy);
        TestTrue(TEXT("Binding is a duplicate of the template class"), Policy && Policy->GetClass() == Template->GetClass());
        TestTrue(TEXT("Binding is initialized with the agent definition"), Policy && Policy->IsInitialized());
        Distinct.Add(Policy);
    }
    TestEqual(TEXT("Bindings are distinct objects"), Distinct.Num(), 3);
    TestNull(TEXT("Out-of-range index returns null"), Stepper->GetPolicy(3));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FAdaptiveStepperWaitsForReadyTest, "Schola.Inference.Adaptive.Stepper.WaitsForReady",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FAdaptiveStepperWaitsForReadyTest::RunTest(const FString& Parameters)
{
    TArray<UAdaptiveStepperTestAgent*> Agents;
    UAdaptiveDecisionTestPolicy* Template = nullptr;
    auto* Stepper = MakeStepper(2, Agents, Template);
    Stepper->Step();
    TestFalse(TEXT("Not ready before sessions start"), Stepper->AreAllReady());
    TestEqual(TEXT("No action before the session is Ready"), Agents[0]->Applied.Num(), 0);
    TestEqual(TEXT("No observation before the session is Ready"), Agents[0]->ObserveCount, 0);
    StartBindings(Stepper);
    TestTrue(TEXT("Ready after sessions start"), Stepper->AreAllReady());
    Agents[1]->Status = EAgentStatus::Stopped;
    Stepper->Step();
    TestEqual(TEXT("A synchronous decision is applied in the same Step"), Agents[0]->Applied.Num(), 1);
    TestEqual(TEXT("Applied value is the policy action"), Agents[0]->Applied.Num() ? Agents[0]->Applied[0] : 0.f, 0.5f);
    TestEqual(TEXT("A stopped agent is skipped"), Agents[1]->Applied.Num(), 0);
    TestEqual(TEXT("Decisions counted"), Stepper->GetStats().Decisions, int64(1));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FAdaptiveStepperSubmitsOutcomeTest, "Schola.Inference.Adaptive.Stepper.SubmitsOutcome",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FAdaptiveStepperSubmitsOutcomeTest::RunTest(const FString& Parameters)
{
    TArray<UAdaptiveStepperTestAgent*> Agents;
    UAdaptiveDecisionTestPolicy* Template = nullptr;
    auto* Stepper = MakeStepper(1, Agents, Template);
    StartBindings(Stepper);
    auto* Policy = TestPolicy(Stepper, 0);
    auto* Agent = Agents[0];
    Agent->ObservationValue = 0.25f;
    Stepper->Step();
    const int64 FirstDecision = Policy->Sent.DecisionId;
    TestEqual(TEXT("Nothing is submitted when the first action is applied"), Policy->ExperienceDispatchCount, 0);

    Agent->Reward = 2.5f;
    Agent->ObservationValue = -0.75f;
    Stepper->Step();
    const FAdaptiveExperience& E = Policy->SubmittedExperience;
    TestEqual(TEXT("One outcome submitted"), Policy->ExperienceDispatchCount, 1);
    TestEqual(TEXT("Attributed to the applied decision"), E.DecisionId, FirstDecision);
    TestEqual(TEXT("Carries the decision's policy version"), E.PolicyVersion, int64(3));
    TestEqual(TEXT("Reward of the interval"), E.Reward, 2.5f);
    TestEqual(TEXT("Observation that produced the decision"), E.Observation.Get<FBoxPoint>().Values[0], 0.25f);
    TestEqual(TEXT("Next observation after acting"), E.NextObservation.Get<FBoxPoint>().Values[0], -0.75f);
    TestEqual(TEXT("Executed action is the applied one"), E.ExecutedAction.Get<FBoxPoint>().Values[0], 0.5f);
    TestFalse(TEXT("Not terminal"), E.bTerminated || E.bTruncated);
    TestEqual(TEXT("Reward was consumed"), Agent->Reward, 0.0f);
    TestEqual(TEXT("The next decision is applied in the same Step"), Agent->Applied.Num(), 2);
    TestTrue(TEXT("The next decision uses the next observation"), Policy->Sent.Observation.Get<FBoxPoint>().Values[0] == -0.75f);
    TestEqual(TEXT("Accepted counted"), Stepper->GetStats().Accepted, int64(1));

    Policy->ExperienceResult = EAdaptiveSubmitResult::Dropped;
    Stepper->Step();
    TestEqual(TEXT("A non-accepted submit is counted, not retried"), Stepper->GetStats().NotAccepted, int64(1));
    TestEqual(TEXT("Stepping continues after a dropped outcome"), Agent->Applied.Num(), 3);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FAdaptiveStepperEpisodeEndTest, "Schola.Inference.Adaptive.Stepper.EpisodeEnd",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FAdaptiveStepperEpisodeEndTest::RunTest(const FString& Parameters)
{
    TArray<UAdaptiveStepperTestAgent*> Agents;
    UAdaptiveDecisionTestPolicy* Template = nullptr;
    auto* Stepper = MakeStepper(1, Agents, Template);
    StartBindings(Stepper);
    auto* Policy = TestPolicy(Stepper, 0);
    auto* Agent = Agents[0];
    Stepper->Step();
    const int64 Episode = Policy->GetSessionInfo().EpisodeId;

    Agent->bTerminated = true;
    Agent->ObservationValue = 0.9f;
    Agent->Reward = 10.0f;
    Stepper->Step();
    const FAdaptiveExperience& E = Policy->SubmittedExperience;
    TestTrue(TEXT("Terminal outcome submitted"), E.bTerminated);
    TestEqual(TEXT("Submitted under the episode that ended (before ResetEpisode)"), E.EpisodeId, Episode);
    TestEqual(TEXT("The policy accepted it"), Stepper->GetStats().Accepted, int64(1));
    TestEqual(TEXT("Terminal next observation is the ending state"), E.NextObservation.Get<FBoxPoint>().Values[0], 0.9f);
    TestEqual(TEXT("Agent reset once"), Agent->ResetCount, 1);
    TestEqual(TEXT("Policy advanced the episode"), Policy->GetSessionInfo().EpisodeId, Episode + 1);
    TestEqual(TEXT("Episode counted"), Stepper->GetStats().Episodes, int64(1));
    TestEqual(TEXT("The new decision observes the reset encounter"), Policy->Sent.Observation.Get<FBoxPoint>().Values[0], 0.0f);
    TestEqual(TEXT("The new decision belongs to the new episode"), Policy->Sent.EpisodeId, Episode + 1);

    Agent->bTruncated = true;
    Stepper->Step();
    TestTrue(TEXT("Truncation is reported separately"), Policy->SubmittedExperience.bTruncated && !Policy->SubmittedExperience.bTerminated);
    TestEqual(TEXT("Truncation also resets"), Agent->ResetCount, 2);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FAdaptiveStepperPendingAndExpiryTest, "Schola.Inference.Adaptive.Stepper.PendingAndExpiry",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FAdaptiveStepperPendingAndExpiryTest::RunTest(const FString& Parameters)
{
    TArray<UAdaptiveStepperTestAgent*> Agents;
    UAdaptiveDecisionTestPolicy* Template = nullptr;
    auto* Stepper = MakeStepper(1, Agents, Template);
    Stepper->DecisionTimeoutSeconds = 1.0f;
    StartBindings(Stepper, false);
    auto* Policy = TestPolicy(Stepper, 0);
    auto* Agent = Agents[0];

    Stepper->Step();
    TestEqual(TEXT("Request dispatched"), Policy->DispatchCount, 1);
    TestEqual(TEXT("No action while pending"), Agent->Applied.Num(), 0);
    Stepper->Step();
    TestEqual(TEXT("No second request while one is pending"), Policy->DispatchCount, 1);

    FAdaptiveDecision Result = Policy->ImmediateResult;
    Result.SessionId = Policy->Sent.SessionId;
    Result.AgentId = Policy->Sent.AgentId;
    Result.EpisodeId = Policy->Sent.EpisodeId;
    Result.DecisionId = Policy->Sent.DecisionId;
    Agent->Reward = 4.0f; // earned while waiting: owned by no decision
    TestTrue(TEXT("Late completion accepted"), Policy->Deliver(Result));
    Stepper->Step();
    TestEqual(TEXT("Completed action applied on the next Step"), Agent->Applied.Num(), 1);
    TestEqual(TEXT("Reward earned while pending is discarded"), Agent->Reward, 0.0f);

    Stepper->Step(); // closes the interval and leaves a new request pending
    TestEqual(TEXT("Second request dispatched"), Policy->DispatchCount, 2);
    TestEqual(TEXT("Interval closed with zero reward"), Policy->SubmittedExperience.Reward, 0.0f);
    Policy->Now += 5.0;
    Stepper->Step();
    TestEqual(TEXT("Expired request is abandoned"), Stepper->GetStats().Abandoned, int64(1));
    TestEqual(TEXT("Nothing applied from an expired request"), Agent->Applied.Num(), 1);
    TestEqual(TEXT("A fresh request is made in the same Step"), Policy->DispatchCount, 3);
    TestEqual(TEXT("No outcome is submitted for an unapplied request"), Policy->ExperienceDispatchCount, 1);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FAdaptiveStepperShutdownTest, "Schola.Inference.Adaptive.Stepper.Shutdown",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FAdaptiveStepperShutdownTest::RunTest(const FString& Parameters)
{
    TArray<UAdaptiveStepperTestAgent*> Agents;
    UAdaptiveDecisionTestPolicy* Template = nullptr;
    auto* Stepper = MakeStepper(2, Agents, Template);
    StartBindings(Stepper);
    Stepper->Step();
    auto* Policy = TestPolicy(Stepper, 0);
    Stepper->Shutdown();
    TestEqual(TEXT("Bindings released"), Policy->ReleaseCount, 1);
    TestTrue(TEXT("Binding stopped"), Policy->GetSessionState() == EAdaptiveSessionState::Stopped);
    TestEqual(TEXT("No agents after shutdown"), Stepper->GetNumAgents(), 0);
    Stepper->Step();
    TestEqual(TEXT("Step after shutdown does nothing"), Agents[0]->Applied.Num(), 1);

    TestTrue(TEXT("Init after shutdown creates fresh bindings"),
        Stepper->Init(TArray<TScriptInterface<IAgent>>{TScriptInterface<IAgent>(Agents[0])}, TScriptInterface<IPolicy>(Template)));
    TestTrue(TEXT("Fresh binding is a new object"), Stepper->GetPolicy(0) != Policy);
    return true;
}
