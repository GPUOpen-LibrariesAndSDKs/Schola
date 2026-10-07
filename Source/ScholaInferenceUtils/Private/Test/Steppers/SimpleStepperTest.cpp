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
	TestEqual(TEXT("One state per agent"), SimpleStepper->GetCurrentStates().Num(), 2);
	TestEqual(TEXT("Agent 0 starts at 0"), UTestStatefulPolicy::GetCount(SimpleStepper->GetCurrentStates()[0]), 0);

	for (int32 i = 0; i < 3; ++i)
	{
		SimpleStepper->Step();
	}
	TestEqual(TEXT("Agent 0 state advanced every step"), UTestStatefulPolicy::GetCount(SimpleStepper->GetCurrentStates()[0]), 3);
	TestEqual(TEXT("Agent 1 state advanced every step"), UTestStatefulPolicy::GetCount(SimpleStepper->GetCurrentStates()[1]), 3);

	SimpleStepper->ResetAgentState(0);
	TestEqual(TEXT("Reset agent 0 is back to 0"), UTestStatefulPolicy::GetCount(SimpleStepper->GetCurrentStates()[0]), 0);
	TestEqual(TEXT("Agent 1 is unaffected by agent 0's reset"), UTestStatefulPolicy::GetCount(SimpleStepper->GetCurrentStates()[1]), 3);

	SimpleStepper->Step();
	TestEqual(TEXT("Agent 0 resumes from its reset state"), UTestStatefulPolicy::GetCount(SimpleStepper->GetCurrentStates()[0]), 1);
	TestEqual(TEXT("Agent 1 keeps its own history"), UTestStatefulPolicy::GetCount(SimpleStepper->GetCurrentStates()[1]), 4);

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
	TestFalse(TEXT("AddAgent before Init fails"), SimpleStepper->AddAgent(Agent0));

	TArray<TScriptInterface<IAgent>> Agents;
	Agents.Add(Agent0);
	Agents.Add(Agent1);
	TestTrue(TEXT("SimpleStepper initialized"), SimpleStepper->Init(Agents, TScriptInterface<IPolicy>(TestPolicy)));
	SimpleStepper->Step();
	SimpleStepper->Step();

	AddExpectedMessage(TEXT("already managed"), EAutomationExpectedMessageFlags::Contains, 1);
	TestFalse(TEXT("Adding an agent twice fails"), SimpleStepper->AddAgent(Agent0));

	TestTrue(TEXT("AddAgent succeeds"), SimpleStepper->AddAgent(Agent2));
	TestEqual(TEXT("Added agent is stepped"), SimpleStepper->Agents.Num(), 3);
	TestEqual(TEXT("Added agent gets its own state"), SimpleStepper->GetCurrentStates().Num(), 3);
	TestEqual(TEXT("Added agent starts at 0"), UTestStatefulPolicy::GetCount(SimpleStepper->GetCurrentStates()[2]), 0);

	SimpleStepper->Step();
	TestEqual(TEXT("Existing agent keeps its history"), UTestStatefulPolicy::GetCount(SimpleStepper->GetCurrentStates()[0]), 3);
	TestEqual(TEXT("Added agent advanced once"), UTestStatefulPolicy::GetCount(SimpleStepper->GetCurrentStates()[2]), 1);

	TestTrue(TEXT("RemoveAgent succeeds"), SimpleStepper->RemoveAgent(Agent0));
	TestEqual(TEXT("Removed agent is no longer stepped"), SimpleStepper->Agents.Num(), 2);
	TestEqual(TEXT("Removed agent's state is released"), SimpleStepper->GetCurrentStates().Num(), 2);
	TestTrue(TEXT("Remaining agents keep their order"), SimpleStepper->Agents[0].GetObject() == Agent1 && SimpleStepper->Agents[1].GetObject() == Agent2);
	TestEqual(TEXT("Remaining states stay with their agents"), UTestStatefulPolicy::GetCount(SimpleStepper->GetCurrentStates()[0]), 3);
	TestEqual(TEXT("Remaining states stay with their agents"), UTestStatefulPolicy::GetCount(SimpleStepper->GetCurrentStates()[1]), 1);

	AddExpectedMessage(TEXT("not managed"), EAutomationExpectedMessageFlags::Contains, 1);
	TestFalse(TEXT("Removing an unknown agent fails"), SimpleStepper->RemoveAgent(Agent0));

	TestTrue(TEXT("A removed agent can be added again"), SimpleStepper->AddAgent(Agent0));
	TestEqual(TEXT("A re-added agent starts from a fresh state"), UTestStatefulPolicy::GetCount(SimpleStepper->GetCurrentStates()[2]), 0);

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

	TWeakObjectPtr<UObject> WeakCurrent = SimpleStepper->GetCurrentStates()[0].GetObject();

	CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS, true);

	TestTrue(TEXT("Current state survives GC while the stepper holds it"), WeakCurrent.IsValid());
	TestNotNull(TEXT("Interface pointer still valid after GC"), SimpleStepper->GetCurrentStates()[0].GetInterface());
	TestEqual(TEXT("State contents preserved across GC"), UTestStatefulPolicy::GetCount(SimpleStepper->GetCurrentStates()[0]), 2);

	SimpleStepper->Step();
	TestEqual(TEXT("Stepping continues after GC"), UTestStatefulPolicy::GetCount(SimpleStepper->GetCurrentStates()[0]), 3);

	SimpleStepper.Reset();
	CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS, true);

	TestFalse(TEXT("Current state is collected once the stepper is gone"), WeakCurrent.IsValid());

	return true;
}