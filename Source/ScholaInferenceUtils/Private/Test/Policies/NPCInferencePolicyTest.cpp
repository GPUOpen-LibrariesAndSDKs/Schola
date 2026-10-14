// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "Misc/AutomationTest.h"
#include "Policies/NPCInferencePolicy.h"
#include "Points/BoxPoint.h"
#include "Common/InstancedStructUtils.h"

namespace
{
TInstancedStruct<FPoint> MockPoint(float Value)
{
    TInstancedStruct<FPoint> Point;
    Point.InitializeAs<FBoxPoint>(TArray<float>{Value});
    return Point;
}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMockAdaptivePolicyTest, "Schola.Inference.Paradigms.Adaptive",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FMockAdaptivePolicyTest::RunTest(const FString& Parameters)
{
    auto* Policy = NewObject<UNPCInferencePolicy>();
    FNPCInferenceConfig Config;
    TestTrue(TEXT("Configure"), Policy->Configure(Config));
    auto Observation = MockPoint(1);
    auto Action = MockPoint(9);
    TestFalse(TEXT("Requires initialization"), Policy->Think(Observation, Action));
    TestTrue(TEXT("Failure preserves action"), Action == MockPoint(9));
    Policy->Init(FInteractionDefinition());
    TestTrue(TEXT("Initial inference"), Policy->Think(Observation, Action));
    TestTrue(TEXT("Initial passthrough"), Action == Observation);
    FPlayerBehaviorFeedback Feedback;
    Feedback.PlayerBehavior = Observation;
    Feedback.PreferredAction = MockPoint(2);
    Feedback.Reward = 1;
    TestTrue(TEXT("Player feedback"), Policy->SubmitFeedback(Feedback));
    Policy->Think(Observation, Action);
    TestTrue(TEXT("Adapts matching behavior"), Action == MockPoint(2));
    Feedback.PreferredAction = MockPoint(3);
    Feedback.Reward = -1;
    Policy->SubmitFeedback(Feedback);
    Policy->Think(Observation, Action);
    TestTrue(TEXT("Negative feedback does not replace learned action"), Action == MockPoint(2));
    Policy->Think(MockPoint(4), Action);
    TestTrue(TEXT("Unseen behavior stays unchanged"), Action == MockPoint(4));
    Policy->Configure(Config);
    Policy->Init(FInteractionDefinition());
    Policy->Think(Observation, Action);
    TestTrue(TEXT("Reconfigure clears learned state"), Action == Observation);
    Config.MaxBatchSize = 0;
    TestFalse(TEXT("Invalid configuration rejected"), Policy->Configure(Config));
    TestTrue(TEXT("Previous valid configuration remains ready"), Policy->Think(Observation, Action));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMockEventPolicyTest, "Schola.Inference.Paradigms.EventBased",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FMockEventPolicyTest::RunTest(const FString& Parameters)
{
    auto* Policy = NewObject<UNPCInferencePolicy>();
    FNPCInferenceConfig Config;
    Config.Paradigm = ENPCInferenceParadigm::EventBased;
    TestFalse(TEXT("Requires subscriptions"), Policy->Configure(Config));
    Config.Events = {FName(TEXT("TurnStarted")), FName(TEXT("PlayerMoved"))};
    TestTrue(TEXT("Configure subscriptions"), Policy->Configure(Config));
    Policy->Init(FInteractionDefinition());
    auto Action = MockPoint(0);
    TestFalse(TEXT("No inference without event"), Policy->Think(MockPoint(1), Action));
    TestFalse(TEXT("Unsubscribed event rejected"), Policy->NotifyEvent(TEXT("Other")));
    TestTrue(TEXT("Subscribed event accepted"), Policy->NotifyEvent(TEXT("PlayerMoved")));
    TArray<TInstancedStruct<FPoint>> Actions = {MockPoint(9)};
    TestFalse(TEXT("Invalid batch rejected"), Policy->BatchedThink({MockPoint(1), {}}, Actions));
    TestTrue(TEXT("Batch failure preserves output"), Actions.Num() == 1 && Actions[0] == MockPoint(9));
    TestTrue(TEXT("Event still permits full batch"), Policy->BatchedThink({MockPoint(1), MockPoint(2)}, Actions));
    TestEqual(TEXT("Whole batch returned"), Actions.Num(), 2);
    TestFalse(TEXT("Permit consumed"), Policy->Think(MockPoint(1), Action));
    TestFalse(TEXT("Feedback rejected for event policy"), Policy->SubmitFeedback(FPlayerBehaviorFeedback()));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMockBackgroundPolicyTest, "Schola.Inference.Paradigms.Background",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FMockBackgroundPolicyTest::RunTest(const FString& Parameters)
{
    auto* Policy = NewObject<UNPCInferencePolicy>();
    FNPCInferenceConfig Config;
    Config.Paradigm = ENPCInferenceParadigm::Background;
    Config.MaxBatchSize = 2;
    Config.WorkerCount = 2;
    TArray<TInstancedStruct<FPoint>> Observations = {MockPoint(1), MockPoint(2), MockPoint(3)};
    TArray<TInstancedStruct<FPoint>> Actions;
    Policy->Configure(Config);
    Policy->Init(FInteractionDefinition());
    TestTrue(TEXT("Background batch"), Policy->BatchedThink(Observations, Actions));
    TestTrue(TEXT("Output preserves order"), Actions == Observations);
    TestEqual(TEXT("Partial final batch dispatched"), Policy->GetStats().BatchDispatches, 2);
    TestTrue(TEXT("Contiguous lanes"), Policy->GetStats().WorkerLanes == TArray<int32>{0, 0, 1});
    Config.FanOut = ENPCInferenceFanOut::RoundRobin;
    Policy->Configure(Config);
    Policy->Init(FInteractionDefinition());
    Policy->BatchedThink(Observations, Actions);
    TestTrue(TEXT("Round robin lanes"), Policy->GetStats().WorkerLanes == TArray<int32>{0, 1, 0});
    Config.FanOut = ENPCInferenceFanOut::MassEntity;
    Policy->Configure(Config);
    Policy->Init(FInteractionDefinition());
    TestFalse(TEXT("Mismatched IDs rejected"), Policy->ThinkEntities({7}, Observations, Actions));
    TestFalse(TEXT("Duplicate IDs rejected"), Policy->ThinkEntities({7, 7, 8}, Observations, Actions));
    TestTrue(TEXT("Mass adapter"), Policy->ThinkEntities({7, 8, 9}, Observations, Actions));
    TestTrue(TEXT("Mass IDs retained"), Policy->GetStats().EntityIds == TArray<int64>{7, 8, 9});
    TestTrue(TEXT("Mass lanes"), Policy->GetStats().WorkerLanes == TArray<int32>{1, 0, 1});
    TestTrue(TEXT("Mass output order"), Actions == Observations);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMockSharedPolicyTest, "Schola.Inference.Paradigms.SharedPolicy",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FMockSharedPolicyTest::RunTest(const FString& Parameters)
{
    auto* Policy = NewObject<UNPCInferencePolicy>();
    FNPCInferenceConfig Config;
    Config.Paradigm = ENPCInferenceParadigm::SharedPolicy;
    Config.Head = TEXT("Combat");
    Policy->Configure(Config);
    Policy->Init(FInteractionDefinition());
    auto Action = MockPoint(0);
    TestFalse(TEXT("Unregistered head rejected"), Policy->Think(MockPoint(1), Action));
    Policy->RegisterHead(TEXT("Combat"), MockPoint(2));
    Policy->RegisterHead(TEXT("Navigation"), MockPoint(3));
    TArray<TInstancedStruct<FPoint>> Actions;
    TestFalse(TEXT("Unknown head rejected atomically"), Policy->ThinkShared(MockPoint(1), {TEXT("Combat"), TEXT("Unknown")}, Actions));
    TestEqual(TEXT("No trunk work on failure"), Policy->GetStats().TrunkEvaluations, 0);
    TestTrue(TEXT("Shared inference"), Policy->ThinkShared(MockPoint(1), {TEXT("Combat"), TEXT("Navigation")}, Actions));
    TestTrue(TEXT("Specialized actions in requested order"), Actions == TArray<TInstancedStruct<FPoint>>{MockPoint(2), MockPoint(3)});
    TestEqual(TEXT("One shared trunk"), Policy->GetStats().TrunkEvaluations, 1);
    TestEqual(TEXT("Two specialized heads"), Policy->GetStats().HeadEvaluations, 2);
    TestTrue(TEXT("IPolicy compatibility"), Policy->Think(MockPoint(1), Action));
    TestTrue(TEXT("Configured default head"), Action == MockPoint(2));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMockSearchPolicyTest, "Schola.Inference.Paradigms.AdvancedStrategy",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FMockSearchPolicyTest::RunTest(const FString& Parameters)
{
    auto* Policy = NewObject<UNPCInferencePolicy>();
    FNPCInferenceConfig Config;
    Config.Paradigm = ENPCInferenceParadigm::AdvancedStrategy;
    Config.SearchBudget = 4;
    FStrategySearchCandidate Risky;
    Risky.Action = MockPoint(1);
    Risky.OpponentReplyScores = {-2, 10};
    FStrategySearchCandidate Safe;
    Safe.Action = MockPoint(2);
    Safe.OpponentReplyScores = {3, 3};
    Policy->Configure(Config);
    Policy->Init(FInteractionDefinition());
    auto Action = MockPoint(0);
    TestFalse(TEXT("Requires search candidates"), Policy->Think(MockPoint(1), Action));
    TestTrue(TEXT("Candidates accepted"), Policy->SetSearchCandidates({Risky, Safe}));
    TestTrue(TEXT("Mock rollouts"), Policy->Think(MockPoint(1), Action));
    TestTrue(TEXT("Mock MCTS selects best rollout mean"), Action == Risky.Action);
    TestEqual(TEXT("Search respects budget"), Policy->GetStats().SearchEvaluations, 4);
    Config.SearchStrategy = ENPCSearchStrategy::Minimax;
    Policy->Configure(Config);
    Policy->Init(FInteractionDefinition());
    Policy->SetSearchCandidates({Risky, Safe});
    Policy->Think(MockPoint(1), Action);
    TestTrue(TEXT("Minimax models worst opponent reply"), Action == Safe.Action);
    Config.SearchBudget = 3;
    Policy->Configure(Config);
    TestFalse(TEXT("Insufficient search budget rejected"), Policy->SetSearchCandidates({Risky, Safe}));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNPCInferenceBlueprintTest, "Schola.Inference.Paradigms.BlueprintAPI",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FNPCInferenceBlueprintTest::RunTest(const FString& Parameters)
{
    auto* Policy = NewObject<UNPCInferencePolicy>();
    for (const TCHAR* Name : {TEXT("Configure"), TEXT("Init"), TEXT("Think"), TEXT("BatchedThink"),
        TEXT("SubmitFeedback"), TEXT("NotifyEvent"), TEXT("RegisterHead"), TEXT("ThinkShared"),
        TEXT("SetSearchCandidates"), TEXT("ThinkEntities"), TEXT("GetConfig"), TEXT("GetStats"), TEXT("IsInferenceBusy")})
    {
        UFunction* Function = Policy->FindFunction(FName(Name));
        TestTrue(Name, Function && Function->HasAnyFunctionFlags(FUNC_BlueprintCallable));
    }
    struct FConfigureParameters
    {
        FNPCInferenceConfig InConfig;
        bool ReturnValue = false;
    } ConfigureArgs;
    ConfigureArgs.InConfig.Paradigm = ENPCInferenceParadigm::EventBased;
    ConfigureArgs.InConfig.Events = {TEXT("TurnStarted")};
    Policy->ProcessEvent(Policy->FindFunctionChecked(TEXT("Configure")), &ConfigureArgs);
    TestTrue(TEXT("Blueprint configure dispatch"), ConfigureArgs.ReturnValue);
    Policy->Init(FInteractionDefinition());
    Policy->NotifyEvent(TEXT("TurnStarted"));
    struct FThinkParameters
    {
        FInstancedStruct Observation;
        FInstancedStruct Action;
        bool ReturnValue = false;
    } ThinkArgs;
    ThinkArgs.Observation.InitializeAs<FInteractionDefinition>();
    auto OriginalAction = MockPoint(9);
    ThinkArgs.Action = ToUntypedInstancedStruct(OriginalAction);
    Policy->ProcessEvent(Policy->FindFunctionChecked(TEXT("Think")), &ThinkArgs);
    TestFalse(TEXT("Wrong Blueprint payload type rejected safely"), ThinkArgs.ReturnValue);
    TestTrue(TEXT("Wrong payload preserves output"), ThinkArgs.Action == ToUntypedInstancedStruct(OriginalAction));
    auto Observation = MockPoint(1);
    ThinkArgs.Observation = ToUntypedInstancedStruct(Observation);
    Policy->ProcessEvent(Policy->FindFunctionChecked(TEXT("Think")), &ThinkArgs);
    TestTrue(TEXT("Typed Blueprint pin dispatch"), ThinkArgs.ReturnValue);
    TestTrue(TEXT("Correct Blueprint action"), ThinkArgs.Action == ThinkArgs.Observation);
    Policy->ProcessEvent(Policy->FindFunctionChecked(TEXT("Think")), &ThinkArgs);
    TestFalse(TEXT("Blueprint event permit consumed"), ThinkArgs.ReturnValue);
    return true;
}
