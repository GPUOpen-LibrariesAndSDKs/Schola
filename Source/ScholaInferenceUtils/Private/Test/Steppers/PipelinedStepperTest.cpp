// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"
#include "Tests/AutomationCommon.h"
#include "CoreMinimal.h"
#include "TestStepper.h"
#include "Spaces/BoxSpace.h"
#include "Spaces/MultiDiscreteSpace.h"
#include "Steppers/PipelinedStepper.h"
#include "UObject/StrongObjectPtr.h"

/**
 * Latent test context for pipelined stepper.
 */
struct FPipelinedStepperTestContext
{
	// Strong references so objects survive GC across latent frames without manual AddToRoot.
	TStrongObjectPtr<UTestAgent> Agent;
	TStrongObjectPtr<UTestPolicy> Policy;
	TStrongObjectPtr<UPipelinedStepper> Stepper;

	int StepsRemaining = 0;
	int TotalSteps = 0;
	FAutomationTestBase* Test = nullptr;
	bool bInitialized = false;
};

// Create objects & initialize stepper
DEFINE_LATENT_AUTOMATION_COMMAND_ONE_PARAMETER(FCreatePipelinedStepperObjectsCommand, TSharedPtr<FPipelinedStepperTestContext>, Context);
bool FCreatePipelinedStepperObjectsCommand::Update()
{
	if (!Context.IsValid()) return true;
	if (Context->bInitialized) return true;

	FInteractionDefinition Def;
	Context->Agent = TStrongObjectPtr<UTestAgent>(NewObject<UTestAgent>());
	Context->Policy = TStrongObjectPtr<UTestPolicy>(NewObject<UTestPolicy>());
	Context->Stepper = TStrongObjectPtr<UPipelinedStepper>(NewObject<UPipelinedStepper>());

	IAgent::Execute_Define(Context->Agent.Get(), Def);

	TArray<TScriptInterface<IAgent>> Agents;

	Agents.Add(Context->Agent.Get());
	TScriptInterface<IPolicy> PolicyInterface = Context->Policy.Get();

	Context->bInitialized = Context->Stepper->Init(Agents, PolicyInterface);
	if (Context->Test)
	{
		Context->Test->TestTrue(TEXT("PipelinedStepper initialized"), Context->bInitialized);
	}
	return true; // done
}

// Perform one Step per frame until StepsRemaining exhausted
DEFINE_LATENT_AUTOMATION_COMMAND_ONE_PARAMETER(FStepPipelinedStepperLatentCommand, TSharedPtr<FPipelinedStepperTestContext>, Context);
bool FStepPipelinedStepperLatentCommand::Update()
{
	if (!Context.IsValid() || !Context->bInitialized)
	{
		return true; // abort
	}
	if (Context->StepsRemaining <= 0)
	{
		return true; // finished stepping
	}

	if (Context->Stepper)
	{
		Context->Stepper->Step();
		--Context->StepsRemaining;
		++Context->TotalSteps;
	}
	// Return false to continue next frame if more steps remain
	return Context->StepsRemaining <= 0;
}

// Verify that an action was eventually applied (LastActionReceived == 1)
DEFINE_LATENT_AUTOMATION_COMMAND_ONE_PARAMETER(FVerifyPipelinedStepperResultCommand, TSharedPtr<FPipelinedStepperTestContext>, Context);
bool FVerifyPipelinedStepperResultCommand::Update()
{
	if (!Context.IsValid() || !Context->Agent)
	{
		return true;
	}
	if (!Context->bInitialized)
	{
		if (Context->Test)
		{
			Context->Test->AddError(TEXT("Initialization failed; skipping action verification."));
		}
		return true;
	}
	int LastAction = Context->Agent->GetLastActionReceived();
	if (Context->Test)
	{
		// Context->Test->TestTrue(TEXT("Agent received a valid action"), LastAction >= 0);
		Context->Test->TestEqual(TEXT("Agent received last action 1"), LastAction, 1);
		// Thread usage assertions (Policy must have run off game thread at least once)
		const TSet<uint32> ThreadIds = UTestPolicy::GetThreadIdsCopy();
		const bool bSawNonGame = UTestPolicy::SawNonGameThread();
		Context->Test->TestTrue(TEXT("Policy executed on a non-game thread"), bSawNonGame);
		Context->Test->TestTrue(TEXT("Policy Think ran at least once"), ThreadIds.Num() > 0);
		Context->Test->AddInfo(FString::Printf(TEXT("Total Steps Executed: %d, LastAction=%d"), Context->TotalSteps, LastAction));
		Context->Test->AddInfo(FString::Printf(TEXT("Policy Thread Count=%d NonGame=%s"), ThreadIds.Num(), bSawNonGame ? TEXT("True") : TEXT("False")));
	}
	return true;
}

// (Add a cleanup latent command)
DEFINE_LATENT_AUTOMATION_COMMAND_ONE_PARAMETER(FCleanupPipelinedStepperCommand, TSharedPtr<FPipelinedStepperTestContext>, Context);
bool FCleanupPipelinedStepperCommand::Update()
{
    if (!Context.IsValid()) return true;

    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FPipelinedStepperTest, "Schola.Steppers.PipelinedStepper Latent", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FPipelinedStepperTest::RunTest(const FString& Parameters)
{
    TSharedPtr<FPipelinedStepperTestContext> Context = MakeShareable(new FPipelinedStepperTestContext());
    Context->StepsRemaining = 6; // Needs >=2 for pipeline (collect/dispatch then apply)
    Context->TotalSteps = 0;
    Context->Test = this;

	// Reset thread tracking before starting
	UTestPolicy::ResetThreadTracking();

    ADD_LATENT_AUTOMATION_COMMAND(FCreatePipelinedStepperObjectsCommand(Context));
    //ADD_LATENT_AUTOMATION_COMMAND(FWaitLatentCommand(0.01f));
    ADD_LATENT_AUTOMATION_COMMAND(FStepPipelinedStepperLatentCommand(Context));
    ADD_LATENT_AUTOMATION_COMMAND(FVerifyPipelinedStepperResultCommand(Context));
    ADD_LATENT_AUTOMATION_COMMAND(FCleanupPipelinedStepperCommand(Context));

    return true;
}

/**
 * Latent test context for pipelined stepper with a stateful policy.
 */
struct FPipelinedStepperStatefulTestContext
{
	TStrongObjectPtr<UTestStatefulPolicy> Policy;
	TStrongObjectPtr<UPipelinedStepper> Stepper;
	TArray<TStrongObjectPtr<UTestAgent>> Agents;

	int StepsRemaining = 0;
	FAutomationTestBase* Test = nullptr;
	bool bInitialized = false;
};

DEFINE_LATENT_AUTOMATION_COMMAND_ONE_PARAMETER(FCreateStatefulPipelinedStepperCommand, TSharedPtr<FPipelinedStepperStatefulTestContext>, Context);
bool FCreateStatefulPipelinedStepperCommand::Update()
{
	Context->Policy = TStrongObjectPtr<UTestStatefulPolicy>(NewObject<UTestStatefulPolicy>());
	Context->Stepper = TStrongObjectPtr<UPipelinedStepper>(NewObject<UPipelinedStepper>());

	TArray<TScriptInterface<IAgent>> Agents;
	for (int i = 0; i < 2; ++i)
	{
		Context->Agents.Emplace(NewObject<UTestAgent>());
		Agents.Add(Context->Agents.Last().Get());
	}

	Context->bInitialized = Context->Stepper->Init(Agents, TScriptInterface<IPolicy>(Context->Policy.Get()));
	Context->Test->TestTrue(TEXT("PipelinedStepper initialized"), Context->bInitialized);
	return true;
}

DEFINE_LATENT_AUTOMATION_COMMAND_ONE_PARAMETER(FStepStatefulPipelinedStepperCommand, TSharedPtr<FPipelinedStepperStatefulTestContext>, Context);
bool FStepStatefulPipelinedStepperCommand::Update()
{
	if (!Context->bInitialized || Context->StepsRemaining <= 0)
	{
		return true;
	}
	Context->Stepper->Step();
	--Context->StepsRemaining;
	return Context->StepsRemaining <= 0;
}

DEFINE_LATENT_AUTOMATION_COMMAND_ONE_PARAMETER(FVerifyStatefulPipelinedStepperCommand, TSharedPtr<FPipelinedStepperStatefulTestContext>, Context);
bool FVerifyStatefulPipelinedStepperCommand::Update()
{
	if (!Context->bInitialized)
	{
		return true;
	}
	const TArray<TScriptInterface<IPolicyState>>& States = Context->Stepper->GetCurrentStates();
	const int32 Count0 = UTestStatefulPolicy::GetCount(States[0]);
	const int32 Count1 = UTestStatefulPolicy::GetCount(States[1]);
	const int32 ThinkCount = Context->Policy->ThinkCount.load();

	Context->Test->TestTrue(TEXT("State advanced at least once"), Count0 > 0);
	Context->Test->TestEqual(TEXT("Both agents advanced the same number of times"), Count0, Count1);
	Context->Test->TestEqual(TEXT("Every Think call advanced exactly one state"), Count0 + Count1, ThinkCount);
	Context->Test->AddInfo(FString::Printf(TEXT("Count=%d ThinkCount=%d"), Count0, ThinkCount));
	return true;
}

DEFINE_LATENT_AUTOMATION_COMMAND_ONE_PARAMETER(FStepAndResetStatefulPipelinedStepperCommand, TSharedPtr<FPipelinedStepperStatefulTestContext>, Context);
bool FStepAndResetStatefulPipelinedStepperCommand::Update()
{
	if (!Context->bInitialized)
	{
		return true;
	}
	// The previous result has been handled, so this Step dispatches and the reset must be deferred until it completes
	Context->Stepper->Step();
	Context->Stepper->ResetStates();
	return true;
}

DEFINE_LATENT_AUTOMATION_COMMAND_ONE_PARAMETER(FVerifyStatefulPipelinedStepperResetCommand, TSharedPtr<FPipelinedStepperStatefulTestContext>, Context);
bool FVerifyStatefulPipelinedStepperResetCommand::Update()
{
	if (!Context->bInitialized)
	{
		return true;
	}
	const TArray<TScriptInterface<IPolicyState>>& States = Context->Stepper->GetCurrentStates();
	Context->Test->TestEqual(TEXT("Agent 0 reset applied after in-flight inference"), UTestStatefulPolicy::GetCount(States[0]), 0);
	Context->Test->TestEqual(TEXT("Agent 1 reset applied after in-flight inference"), UTestStatefulPolicy::GetCount(States[1]), 0);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FPipelinedStepperStatefulTest, "Schola.Steppers.PipelinedStepper Stateful Policy", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FPipelinedStepperStatefulTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FPipelinedStepperStatefulTestContext> Context = MakeShared<FPipelinedStepperStatefulTestContext>();
	Context->StepsRemaining = 6;
	Context->Test = this;

	ADD_LATENT_AUTOMATION_COMMAND(FCreateStatefulPipelinedStepperCommand(Context));
	ADD_LATENT_AUTOMATION_COMMAND(FStepStatefulPipelinedStepperCommand(Context));
	ADD_LATENT_AUTOMATION_COMMAND(FWaitLatentCommand(0.5f));
	ADD_LATENT_AUTOMATION_COMMAND(FVerifyStatefulPipelinedStepperCommand(Context));
	ADD_LATENT_AUTOMATION_COMMAND(FStepAndResetStatefulPipelinedStepperCommand(Context));
	ADD_LATENT_AUTOMATION_COMMAND(FWaitLatentCommand(0.5f));
	ADD_LATENT_AUTOMATION_COMMAND(FVerifyStatefulPipelinedStepperResetCommand(Context));

	return true;
}

DEFINE_LATENT_AUTOMATION_COMMAND_ONE_PARAMETER(FStepAndSwapAgentsPipelinedStepperCommand, TSharedPtr<FPipelinedStepperStatefulTestContext>, Context);
bool FStepAndSwapAgentsPipelinedStepperCommand::Update()
{
	if (!Context->bInitialized)
	{
		return true;
	}
	// The first Step dispatches, so agent 0 is removed and agent 2 added while that inference is in flight
	Context->Stepper->Step();
	Context->Agents.Emplace(NewObject<UTestAgent>());
	Context->Test->TestTrue(TEXT("RemoveAgent while in flight succeeds"), Context->Stepper->RemoveAgent(Context->Agents[0].Get()));
	Context->Test->TestTrue(TEXT("AddAgent while in flight succeeds"), Context->Stepper->AddAgent(Context->Agents[2].Get()));
	return true;
}

DEFINE_LATENT_AUTOMATION_COMMAND_ONE_PARAMETER(FVerifySwappedAgentsPipelinedStepperCommand, TSharedPtr<FPipelinedStepperStatefulTestContext>, Context);
bool FVerifySwappedAgentsPipelinedStepperCommand::Update()
{
	if (!Context->bInitialized)
	{
		return true;
	}
	const TArray<TScriptInterface<IPolicyState>>& States = Context->Stepper->GetCurrentStates();
	if (!Context->Test->TestEqual(TEXT("One state per remaining agent"), States.Num(), 2))
	{
		return true;
	}
	const int32 Count1 = UTestStatefulPolicy::GetCount(States[0]);
	const int32 Count2 = UTestStatefulPolicy::GetCount(States[1]);
	const int32 ThinkCount = Context->Policy->ThinkCount.load();

	Context->Test->TestEqual(TEXT("Removed agent never receives the in-flight action"), Context->Agents[0]->GetLastActionReceived(), -1);
	Context->Test->TestEqual(TEXT("Remaining agent keeps acting"), Context->Agents[1]->GetLastActionReceived(), 1);
	Context->Test->TestEqual(TEXT("Added agent starts acting"), Context->Agents[2]->GetLastActionReceived(), 1);
	Context->Test->TestEqual(TEXT("Added agent missed only the in-flight inference"), Count1, Count2 + 1);
	Context->Test->TestEqual(TEXT("The removed agent's state was advanced exactly once"), ThinkCount, Count1 + Count2 + 1);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FPipelinedStepperAddRemoveAgentTest, "Schola.Steppers.PipelinedStepper Add Remove Agent In Flight", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FPipelinedStepperAddRemoveAgentTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FPipelinedStepperStatefulTestContext> Context = MakeShared<FPipelinedStepperStatefulTestContext>();
	Context->StepsRemaining = 6;
	Context->Test = this;

	ADD_LATENT_AUTOMATION_COMMAND(FCreateStatefulPipelinedStepperCommand(Context));
	ADD_LATENT_AUTOMATION_COMMAND(FStepAndSwapAgentsPipelinedStepperCommand(Context));
	ADD_LATENT_AUTOMATION_COMMAND(FWaitLatentCommand(0.5f));
	ADD_LATENT_AUTOMATION_COMMAND(FStepStatefulPipelinedStepperCommand(Context));
	ADD_LATENT_AUTOMATION_COMMAND(FWaitLatentCommand(0.5f));
	ADD_LATENT_AUTOMATION_COMMAND(FVerifySwappedAgentsPipelinedStepperCommand(Context));

	return true;
}

DEFINE_LATENT_AUTOMATION_COMMAND_ONE_PARAMETER(FStepAndReaddSameAgentPipelinedStepperCommand, TSharedPtr<FPipelinedStepperStatefulTestContext>, Context);
bool FStepAndReaddSameAgentPipelinedStepperCommand::Update()
{
	if (!Context->bInitialized)
	{
		return true;
	}

	Context->Stepper->Step();
	Context->Test->TestTrue(TEXT("RemoveAgent while in flight succeeds"), Context->Stepper->RemoveAgent(Context->Agents[0].Get()));
	Context->Test->TestTrue(TEXT("Re-adding the same agent while in flight succeeds"), Context->Stepper->AddAgent(Context->Agents[0].Get()));
	return true;
}

DEFINE_LATENT_AUTOMATION_COMMAND_ONE_PARAMETER(FApplyOldMembershipActionPipelinedStepperCommand, TSharedPtr<FPipelinedStepperStatefulTestContext>, Context);
bool FApplyOldMembershipActionPipelinedStepperCommand::Update()
{
	if (!Context->bInitialized)
	{
		return true;
	}

	Context->Stepper->Step();
	Context->Test->TestEqual(TEXT("Re-added agent does not receive its old membership's action"), Context->Agents[0]->GetLastActionReceived(), -1);
	return true;
}

DEFINE_LATENT_AUTOMATION_COMMAND_ONE_PARAMETER(FApplyNewMembershipActionPipelinedStepperCommand, TSharedPtr<FPipelinedStepperStatefulTestContext>, Context);
bool FApplyNewMembershipActionPipelinedStepperCommand::Update()
{
	if (!Context->bInitialized)
	{
		return true;
	}

	Context->Stepper->Step();
	Context->Test->TestEqual(TEXT("Re-added agent receives an action from its new membership"), Context->Agents[0]->GetLastActionReceived(), 1);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FPipelinedStepperReaddSameAgentTest,
	"Schola.Steppers.PipelinedStepper Re-add Same Agent In Flight",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FPipelinedStepperReaddSameAgentTest::RunTest(const FString& Parameters)
{
	TSharedPtr<FPipelinedStepperStatefulTestContext> Context = MakeShared<FPipelinedStepperStatefulTestContext>();
	Context->Test = this;

	ADD_LATENT_AUTOMATION_COMMAND(FCreateStatefulPipelinedStepperCommand(Context));
	ADD_LATENT_AUTOMATION_COMMAND(FStepAndReaddSameAgentPipelinedStepperCommand(Context));
	ADD_LATENT_AUTOMATION_COMMAND(FWaitLatentCommand(0.5f));
	ADD_LATENT_AUTOMATION_COMMAND(FApplyOldMembershipActionPipelinedStepperCommand(Context));
	ADD_LATENT_AUTOMATION_COMMAND(FWaitLatentCommand(0.5f));
	ADD_LATENT_AUTOMATION_COMMAND(FApplyNewMembershipActionPipelinedStepperCommand(Context));

	return true;
}
