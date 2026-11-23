// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "Environment/StateTreeTrainingEnvironment.h"
#include "Engine/World.h"
#include "UObject/Package.h"

#if WITH_DEV_AUTOMATION_TESTS

// ========== Default Configuration Tests ==========

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FStateTreeTrainingEnvDefaultsTest, "Schola.StateTree.Environment.TrainingEnv.Defaults", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FStateTreeTrainingEnvDefaultsTest::RunTest(const FString& Parameters)
{
	AStateTreeTrainingEnvironment* Env = NewObject<AStateTreeTrainingEnvironment>();

	TestEqual(TEXT("Default MaxStepsPerEpisode should be 1000"), Env->MaxStepsPerEpisode, 1000);
	TestEqual(TEXT("Default CurrentStep should be 0"), Env->CurrentStep, 0);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FStateTreeTrainingEnvConfigurableTest, "Schola.StateTree.Environment.TrainingEnv.Configurable", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FStateTreeTrainingEnvConfigurableTest::RunTest(const FString& Parameters)
{
	AStateTreeTrainingEnvironment* Env = NewObject<AStateTreeTrainingEnvironment>();

	Env->MaxStepsPerEpisode = 500;
	TestEqual(TEXT("MaxStepsPerEpisode should be configurable"), Env->MaxStepsPerEpisode, 500);

	Env->CurrentStep = 100;
	TestEqual(TEXT("CurrentStep should be settable"), Env->CurrentStep, 100);

	return true;
}

// ========== Agent Activation Tests ==========

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FStateTreeTrainingEnvActivateAgentTest, "Schola.StateTree.Environment.TrainingEnv.ActivateAgent", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FStateTreeTrainingEnvActivateAgentTest::RunTest(const FString& Parameters)
{
	AStateTreeTrainingEnvironment* Env = NewObject<AStateTreeTrainingEnvironment>();

	// Initially no agents are active
	TestFalse(TEXT("Agent should not be active initially"), Env->IsAgentActive(TEXT("agent_1")));

	// Activate agent
	Env->ActivateAgent(TEXT("agent_1"));
	TestTrue(TEXT("Agent should be active after activation"), Env->IsAgentActive(TEXT("agent_1")));

	// Other agents should still be inactive
	TestFalse(TEXT("Other agents should remain inactive"), Env->IsAgentActive(TEXT("agent_2")));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FStateTreeTrainingEnvDeactivateAgentTest, "Schola.StateTree.Environment.TrainingEnv.DeactivateAgent", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FStateTreeTrainingEnvDeactivateAgentTest::RunTest(const FString& Parameters)
{
	AStateTreeTrainingEnvironment* Env = NewObject<AStateTreeTrainingEnvironment>();

	// Activate then deactivate
	Env->ActivateAgent(TEXT("agent_1"));
	TestTrue(TEXT("Agent should be active"), Env->IsAgentActive(TEXT("agent_1")));

	Env->DeactivateAgent(TEXT("agent_1"));
	TestFalse(TEXT("Agent should be inactive after deactivation"), Env->IsAgentActive(TEXT("agent_1")));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FStateTreeTrainingEnvMultipleAgentsTest, "Schola.StateTree.Environment.TrainingEnv.MultipleAgents", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FStateTreeTrainingEnvMultipleAgentsTest::RunTest(const FString& Parameters)
{
	AStateTreeTrainingEnvironment* Env = NewObject<AStateTreeTrainingEnvironment>();

	// Activate multiple agents
	Env->ActivateAgent(TEXT("branch_selector"));
	Env->ActivateAgent(TEXT("forward_agent"));
	Env->ActivateAgent(TEXT("backward_agent"));

	TestTrue(TEXT("branch_selector should be active"), Env->IsAgentActive(TEXT("branch_selector")));
	TestTrue(TEXT("forward_agent should be active"), Env->IsAgentActive(TEXT("forward_agent")));
	TestTrue(TEXT("backward_agent should be active"), Env->IsAgentActive(TEXT("backward_agent")));

	// Deactivate one
	Env->DeactivateAgent(TEXT("forward_agent"));
	TestTrue(TEXT("branch_selector should still be active"), Env->IsAgentActive(TEXT("branch_selector")));
	TestFalse(TEXT("forward_agent should be inactive"), Env->IsAgentActive(TEXT("forward_agent")));
	TestTrue(TEXT("backward_agent should still be active"), Env->IsAgentActive(TEXT("backward_agent")));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FStateTreeTrainingEnvDoubleActivationTest, "Schola.StateTree.Environment.TrainingEnv.DoubleActivation", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FStateTreeTrainingEnvDoubleActivationTest::RunTest(const FString& Parameters)
{
	AStateTreeTrainingEnvironment* Env = NewObject<AStateTreeTrainingEnvironment>();

	// Double activation should be safe (no crash)
	Env->ActivateAgent(TEXT("agent_1"));
	Env->ActivateAgent(TEXT("agent_1"));
	TestTrue(TEXT("Agent should remain active after double activation"), Env->IsAgentActive(TEXT("agent_1")));

	// Double deactivation should be safe
	Env->DeactivateAgent(TEXT("agent_1"));
	Env->DeactivateAgent(TEXT("agent_1"));
	TestFalse(TEXT("Agent should remain inactive after double deactivation"), Env->IsAgentActive(TEXT("agent_1")));

	return true;
}

// ========== GetBranchAction Tests ==========

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FStateTreeTrainingEnvGetBranchActionNoActionTest, "Schola.StateTree.Environment.TrainingEnv.GetBranchAction.NoAction", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FStateTreeTrainingEnvGetBranchActionNoActionTest::RunTest(const FString& Parameters)
{
	AStateTreeTrainingEnvironment* Env = NewObject<AStateTreeTrainingEnvironment>();

	// No actions have been set, should return -1
	int32 Branch = Env->GetBranchAction(TEXT("branch_selector"));
	TestEqual(TEXT("GetBranchAction with no action should return -1"), Branch, -1);

	return true;
}

// ========== Edge Cases ==========

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FStateTreeTrainingEnvEmptyAgentIdTest, "Schola.StateTree.Environment.TrainingEnv.EmptyAgentId", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FStateTreeTrainingEnvEmptyAgentIdTest::RunTest(const FString& Parameters)
{
	AStateTreeTrainingEnvironment* Env = NewObject<AStateTreeTrainingEnvironment>();

	// Empty agent ID should not crash
	Env->ActivateAgent(TEXT(""));
	TestTrue(TEXT("Empty agent ID activation should work"), Env->IsAgentActive(TEXT("")));

	Env->DeactivateAgent(TEXT(""));
	TestFalse(TEXT("Empty agent ID deactivation should work"), Env->IsAgentActive(TEXT("")));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FStateTreeTrainingEnvSpecialCharAgentIdTest, "Schola.StateTree.Environment.TrainingEnv.SpecialCharAgentId", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FStateTreeTrainingEnvSpecialCharAgentIdTest::RunTest(const FString& Parameters)
{
	AStateTreeTrainingEnvironment* Env = NewObject<AStateTreeTrainingEnvironment>();

	// Special characters should be handled
	Env->ActivateAgent(TEXT("agent_with_underscore"));
	TestTrue(TEXT("Agent with underscore should work"), Env->IsAgentActive(TEXT("agent_with_underscore")));

	Env->ActivateAgent(TEXT("agent-with-dash"));
	TestTrue(TEXT("Agent with dash should work"), Env->IsAgentActive(TEXT("agent-with-dash")));

	Env->ActivateAgent(TEXT("agent.with.dot"));
	TestTrue(TEXT("Agent with dot should work"), Env->IsAgentActive(TEXT("agent.with.dot")));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FStateTreeTrainingEnvDeactivatedAgentTerminatedTest, "Schola.StateTree.Environment.TrainingEnv.DeactivatedAgentTerminated", EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)
bool FStateTreeTrainingEnvDeactivatedAgentTerminatedTest::RunTest(const FString& Parameters)
{
	AStateTreeTrainingEnvironment* Env = NewObject<AStateTreeTrainingEnvironment>();

	Env->ActivateAgent(TEXT("forward"));
	TMap<FString, FAgentState> States;
	States.Add(TEXT("forward"));
	Env->DeactivateAgent(TEXT("forward"));
	TMap<FString, FInstancedStruct> Actions;
	Env->Step(Actions, States);
	TestTrue(TEXT("Deactivated agent is present and terminated"), States.Contains(TEXT("forward")) && States[TEXT("forward")].bTerminated);

	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
