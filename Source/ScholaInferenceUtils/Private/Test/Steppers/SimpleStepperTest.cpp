// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "TestStepper.h"
#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "UObject/Object.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/GarbageCollection.h"

#include "Spaces/BoxSpace.h"
#include "Spaces/MultiDiscreteSpace.h"
#include "Agent/AgentInterface.h"
#include "Steppers/SimpleStepper.h"
#include "Policies/PolicyInterface.h"

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSimpleStepperTest, "Schola.Steppers.SimpleStepper Test", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FSimpleStepperTest::RunTest(const FString& Parameters)
{
	
	FInteractionDefinition AgentDefn;
	UTestAgent* TestAgent = NewObject<UTestAgent>();
	IAgent::Execute_Define(TestAgent, AgentDefn);

	UTestPolicy* TestPolicy = NewObject<UTestPolicy>();

	USimpleStepper* SimpleStepper = NewObject<USimpleStepper>();

	TArray<TScriptInterface<IAgent>> Agents;
	Agents.Add(TestAgent);

	TScriptInterface<IPolicy> PolicyInterface = TestPolicy;

	bool InitStepper = SimpleStepper->Init(Agents, PolicyInterface);

	TestTrue("SimpleStepper initialized successfully", InitStepper);

	SimpleStepper->Step();

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSimpleStepperStatefulTest, "Schola.Steppers.SimpleStepper Stateful Policy", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FSimpleStepperStatefulTest::RunTest(const FString& Parameters)
{
	UTestStatefulPolicy* TestPolicy = NewObject<UTestStatefulPolicy>();
	USimpleStepper* SimpleStepper = NewObject<USimpleStepper>();

	TArray<TScriptInterface<IAgent>> Agents;
	Agents.Add(NewObject<UTestAgent>());
	Agents.Add(NewObject<UTestAgent>());

	TestTrue(TEXT("SimpleStepper initialized"), SimpleStepper->Init(Agents, TScriptInterface<IPolicy>(TestPolicy)));
	TestEqual(TEXT("Both agents added"), SimpleStepper->NumAgents(), 2);
	TestEqual(TEXT("Agent 0 starts at 0"), UTestStatefulPolicy::GetCount(SimpleStepper->GetState(0)), 0);

	for (int32 i = 0; i < 3; ++i)
	{
		SimpleStepper->Step();
	}
	TestEqual(TEXT("Agent 0 state advanced every step"), UTestStatefulPolicy::GetCount(SimpleStepper->GetState(0)), 3);
	TestEqual(TEXT("Agent 1 state advanced every step"), UTestStatefulPolicy::GetCount(SimpleStepper->GetState(1)), 3);

	SimpleStepper->ResetState(0);
	TestEqual(TEXT("Reset agent 0 is back to 0"), UTestStatefulPolicy::GetCount(SimpleStepper->GetState(0)), 0);
	TestEqual(TEXT("Agent 1 is unaffected by agent 0's reset"), UTestStatefulPolicy::GetCount(SimpleStepper->GetState(1)), 3);

	SimpleStepper->Step();
	TestEqual(TEXT("Agent 0 resumes from its reset state"), UTestStatefulPolicy::GetCount(SimpleStepper->GetState(0)), 1);
	TestEqual(TEXT("Agent 1 keeps its own history"), UTestStatefulPolicy::GetCount(SimpleStepper->GetState(1)), 4);

	SimpleStepper->ResetStates();
	TestEqual(TEXT("ResetStates resets agent 0"), UTestStatefulPolicy::GetCount(SimpleStepper->GetState(0)), 0);
	TestEqual(TEXT("ResetStates resets agent 1"), UTestStatefulPolicy::GetCount(SimpleStepper->GetState(1)), 0);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSimpleStepperAddRemoveAgentTest, "Schola.Steppers.SimpleStepper Add Remove Agent", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FSimpleStepperAddRemoveAgentTest::RunTest(const FString& Parameters)
{
	UTestStatefulPolicy* TestPolicy = NewObject<UTestStatefulPolicy>();
	USimpleStepper* SimpleStepper = NewObject<USimpleStepper>();
	UTestAgent* Agent0 = NewObject<UTestAgent>();
	UTestAgent* Agent1 = NewObject<UTestAgent>();
	UTestAgent* Agent2 = NewObject<UTestAgent>();

	AddExpectedMessage(TEXT("call Init first"), EAutomationExpectedMessageFlags::Contains, 1);
	TestEqual(TEXT("AddAgent before Init fails"), SimpleStepper->AddAgent(Agent0), INDEX_NONE);

	TArray<TScriptInterface<IAgent>> Agents;
	Agents.Add(Agent0);
	Agents.Add(Agent1);
	TestTrue(TEXT("SimpleStepper initialized"), SimpleStepper->Init(Agents, TScriptInterface<IPolicy>(TestPolicy)));
	SimpleStepper->Step();
	SimpleStepper->Step();

	TestEqual(TEXT("AddAgent returns the new agent's index"), SimpleStepper->AddAgent(Agent2), 2);
	TestEqual(TEXT("Added agent is stepped"), SimpleStepper->NumAgents(), 3);
	TestTrue(TEXT("GetAgent returns the added agent"), SimpleStepper->GetAgent(2).GetObject() == Agent2);
	TestEqual(TEXT("Added agent starts at 0"), UTestStatefulPolicy::GetCount(SimpleStepper->GetState(2)), 0);

	SimpleStepper->Step();
	TestEqual(TEXT("Existing agent keeps its history"), UTestStatefulPolicy::GetCount(SimpleStepper->GetState(0)), 3);
	TestEqual(TEXT("Added agent advanced once"), UTestStatefulPolicy::GetCount(SimpleStepper->GetState(2)), 1);

	TestEqual(TEXT("FindAgent returns the agent's index"), SimpleStepper->FindAgent(Agent1), 1);
	TestTrue(TEXT("RemoveAgentAt succeeds"), SimpleStepper->RemoveAgentAt(SimpleStepper->FindAgent(Agent0)));
	TestEqual(TEXT("Removed agent is no longer stepped"), SimpleStepper->NumAgents(), 2);
	TestEqual(TEXT("FindAgent no longer finds the removed agent"), SimpleStepper->FindAgent(Agent0), INDEX_NONE);
	TestTrue(TEXT("Later agents shift down and keep their order"), SimpleStepper->GetAgent(0).GetObject() == Agent1 && SimpleStepper->GetAgent(1).GetObject() == Agent2);
	TestEqual(TEXT("Remaining states stay with their agents"), UTestStatefulPolicy::GetCount(SimpleStepper->GetState(0)), 3);
	TestEqual(TEXT("Remaining states stay with their agents"), UTestStatefulPolicy::GetCount(SimpleStepper->GetState(1)), 1);

	AddExpectedMessage(TEXT("Invalid agent index"), EAutomationExpectedMessageFlags::Contains, 4);
	TestFalse(TEXT("Removing INDEX_NONE fails"), SimpleStepper->RemoveAgentAt(INDEX_NONE));
	TestFalse(TEXT("Removing past the end fails"), SimpleStepper->RemoveAgentAt(2));
	TestEqual(TEXT("Removing an agent that is not managed removes nothing"), SimpleStepper->RemoveAgent(Agent0), 0);
	TestNull(TEXT("GetAgent with an invalid index returns null"), SimpleStepper->GetAgent(2).GetObject());
	TestNull(TEXT("GetState with an invalid index returns null"), SimpleStepper->GetState(2).GetObject());

	TestEqual(TEXT("A removed agent can be added again"), SimpleStepper->AddAgent(Agent0), 2);
	TestEqual(TEXT("A re-added agent starts from a fresh state"), UTestStatefulPolicy::GetCount(SimpleStepper->GetState(2)), 0);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSimpleStepperRejectsNoneStateTest, "Schola.Steppers.SimpleStepper Rejects None State", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FSimpleStepperRejectsNoneStateTest::RunTest(const FString& Parameters)
{
	UTestStatefulPolicy* TestPolicy = NewObject<UTestStatefulPolicy>();
	USimpleStepper* SimpleStepper = NewObject<USimpleStepper>();

	TArray<TScriptInterface<IAgent>> Agents;
	Agents.Add(NewObject<UTestAgent>());
	TestTrue(TEXT("SimpleStepper initialized"), SimpleStepper->Init(Agents, TScriptInterface<IPolicy>(TestPolicy)));

	TestPolicy->bReturnNoneState = true;
	AddExpectedMessage(TEXT("failed to create the initial state"), EAutomationExpectedMessageFlags::Contains, 1);
	TestEqual(TEXT("AddAgent fails when the policy returns a None state"), SimpleStepper->AddAgent(NewObject<UTestAgent>()), INDEX_NONE);
	TestEqual(TEXT("The rejected agent is not added"), SimpleStepper->NumAgents(), 1);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSimpleStepperDuplicateAgentTest, "Schola.Steppers.SimpleStepper Duplicate Agent", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FSimpleStepperDuplicateAgentTest::RunTest(const FString& Parameters)
{
	USimpleStepper* SimpleStepper = NewObject<USimpleStepper>();
	UTestAgent* Agent = NewObject<UTestAgent>();

	TArray<TScriptInterface<IAgent>> Agents;
	Agents.Add(Agent);
	TestTrue(TEXT("SimpleStepper initialized"), SimpleStepper->Init(Agents, TScriptInterface<IPolicy>(NewObject<UTestStatefulPolicy>())));
	SimpleStepper->Step();
	SimpleStepper->Step();

	TestEqual(TEXT("The same agent can be added again"), SimpleStepper->AddAgent(Agent), 1);
	TestTrue(TEXT("Each entry gets its own state"), SimpleStepper->GetState(0).GetObject() != SimpleStepper->GetState(1).GetObject());
	TestEqual(TEXT("FindAgent returns the first entry"), SimpleStepper->FindAgent(Agent), 0);

	SimpleStepper->Step();
	TestEqual(TEXT("First entry keeps its history"), UTestStatefulPolicy::GetCount(SimpleStepper->GetState(0)), 3);
	TestEqual(TEXT("Second entry advanced once"), UTestStatefulPolicy::GetCount(SimpleStepper->GetState(1)), 1);

	TestTrue(TEXT("Removing the first entry succeeds"), SimpleStepper->RemoveAgentAt(0));
	TestEqual(TEXT("The other entry is still managed"), SimpleStepper->FindAgent(Agent), 0);
	TestEqual(TEXT("The other entry keeps its state"), UTestStatefulPolicy::GetCount(SimpleStepper->GetState(0)), 1);

	UTestAgent* OtherAgent = NewObject<UTestAgent>();
	SimpleStepper->AddAgent(OtherAgent);
	SimpleStepper->AddAgent(Agent);
	TestEqual(TEXT("RemoveAgent removes every entry for the agent"), SimpleStepper->RemoveAgent(Agent), 2);
	TestEqual(TEXT("Other agents are kept"), SimpleStepper->NumAgents(), 1);
	TestTrue(TEXT("The kept agent is the other agent"), SimpleStepper->GetAgent(0).GetObject() == OtherAgent);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSimpleStepperStateGarbageCollectionTest, "Schola.Steppers.SimpleStepper State Garbage Collection", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FSimpleStepperStateGarbageCollectionTest::RunTest(const FString& Parameters)
{
	TStrongObjectPtr<USimpleStepper> SimpleStepper(NewObject<USimpleStepper>());

	TArray<TScriptInterface<IAgent>> Agents;
	Agents.Add(NewObject<UTestAgent>());
	TestTrue(TEXT("SimpleStepper initialized"), SimpleStepper->Init(Agents, TScriptInterface<IPolicy>(NewObject<UTestStatefulPolicy>())));

	SimpleStepper->Step();
	SimpleStepper->Step();

	TWeakObjectPtr<UObject> WeakCurrent = SimpleStepper->GetState(0).GetObject();

	CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS, true);

	TestTrue(TEXT("Current state survives GC while the stepper holds it"), WeakCurrent.IsValid());
	TestNotNull(TEXT("Interface pointer still valid after GC"), SimpleStepper->GetState(0).GetInterface());
	TestEqual(TEXT("State contents preserved across GC"), UTestStatefulPolicy::GetCount(SimpleStepper->GetState(0)), 2);

	SimpleStepper->Step();
	TestEqual(TEXT("Stepping continues after GC"), UTestStatefulPolicy::GetCount(SimpleStepper->GetState(0)), 3);

	SimpleStepper.Reset();
	CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS, true);

	TestFalse(TEXT("Current state is collected once the stepper is gone"), WeakCurrent.IsValid());

	return true;
}