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
	TestEqual(TEXT("One state per agent"), SimpleStepper->CurrentStates.Num(), 2);
	TestEqual(TEXT("Agent 0 starts at 0"), UTestStatefulPolicy::GetCount(SimpleStepper->CurrentStates[0]), 0);

	for (int32 i = 0; i < 3; ++i)
	{
		SimpleStepper->Step();
	}
	TestEqual(TEXT("Agent 0 state advanced every step"), UTestStatefulPolicy::GetCount(SimpleStepper->CurrentStates[0]), 3);
	TestEqual(TEXT("Agent 1 state advanced every step"), UTestStatefulPolicy::GetCount(SimpleStepper->CurrentStates[1]), 3);

	SimpleStepper->ResetAgentState(0);
	TestEqual(TEXT("Reset agent 0 is back to 0"), UTestStatefulPolicy::GetCount(SimpleStepper->CurrentStates[0]), 0);
	TestEqual(TEXT("Agent 1 is unaffected by agent 0's reset"), UTestStatefulPolicy::GetCount(SimpleStepper->CurrentStates[1]), 3);

	SimpleStepper->Step();
	TestEqual(TEXT("Agent 0 resumes from its reset state"), UTestStatefulPolicy::GetCount(SimpleStepper->CurrentStates[0]), 1);
	TestEqual(TEXT("Agent 1 keeps its own history"), UTestStatefulPolicy::GetCount(SimpleStepper->CurrentStates[1]), 4);

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

	TWeakObjectPtr<UObject> WeakCurrent = SimpleStepper->CurrentStates[0].GetObject();
	TWeakObjectPtr<UObject> WeakNext = SimpleStepper->NextStates[0].GetObject();

	CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS, true);

	TestTrue(TEXT("Current state survives GC while the stepper holds it"), WeakCurrent.IsValid());
	TestTrue(TEXT("Next state survives GC while the stepper holds it"), WeakNext.IsValid());
	TestNotNull(TEXT("Interface pointer still valid after GC"), SimpleStepper->CurrentStates[0].GetInterface());
	TestEqual(TEXT("State contents preserved across GC"), UTestStatefulPolicy::GetCount(SimpleStepper->CurrentStates[0]), 2);

	SimpleStepper->Step();
	TestEqual(TEXT("Stepping continues after GC"), UTestStatefulPolicy::GetCount(SimpleStepper->CurrentStates[0]), 3);

	SimpleStepper.Reset();
	CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS, true);

	TestFalse(TEXT("Current state is collected once the stepper is gone"), WeakCurrent.IsValid());
	TestFalse(TEXT("Next state is collected once the stepper is gone"), WeakNext.IsValid());

	return true;
}