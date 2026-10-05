// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "Test/Policies/ChatHistoryTestPolicy.h"
#include "Test/Policies/PassthroughBoxPolicy.h"

#include "Points/BoxPoint.h"
#include "Policies/PolicyInterface.h"
#include "Policies/PolicyStateInterface.h"

#if WITH_AUTOMATION_TESTS

namespace ScholaPolicyStateTestPrivate
{
	static TInstancedStruct<FPoint> MakeObservation(float Value)
	{
		return TInstancedStruct<FPoint>::Make<FBoxPoint>(TArray<float> { Value });
	}

	static const UTestChatHistoryState* AsHistory(const TScriptInterface<IPolicyState>& State)
	{
		return Cast<UTestChatHistoryState>(State.GetObject());
	}

	static float ActionValue(const TInstancedStruct<FPoint>& Action)
	{
		const FBoxPoint* Box = Action.GetPtr<FBoxPoint>();
		return Box && Box->Values.Num() > 0 ? Box->Values[0] : -1.0f;
	}
} // namespace ScholaPolicyStateTestPrivate

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FPolicyStateStatelessDefaultTest,
	"Schola.Policies.IPolicyState.Stateless Policy Creates Null State",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPolicyStateStatelessDefaultTest::RunTest(const FString& Parameters)
{
	UPassthroughBoxPolicy* PolicyObj = NewObject<UPassthroughBoxPolicy>();
	IPolicy* Policy = Cast<IPolicy>(PolicyObj);

	TScriptInterface<IPolicyState> State = NewObject<UTestChatHistoryState>();
	TestTrue(TEXT("CreateInitialState succeeds"), Policy->CreateInitialState(PolicyObj, State));
	TestNull(TEXT("Stateless policy returns a null state"), State.GetObject());

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FPolicyStateHistoryCarriedAcrossStepsTest,
	"Schola.Policies.IPolicyState.State Carried Across Steps",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPolicyStateHistoryCarriedAcrossStepsTest::RunTest(const FString& Parameters)
{
	using namespace ScholaPolicyStateTestPrivate;

	UChatHistoryTestPolicy* PolicyObj = NewObject<UChatHistoryTestPolicy>();
	IPolicy* Policy = Cast<IPolicy>(PolicyObj);

	TScriptInterface<IPolicyState> CurrentState;
	TScriptInterface<IPolicyState> NextState;
	TestTrue(TEXT("Create current state"), Policy->CreateInitialState(PolicyObj, CurrentState));
	TestTrue(TEXT("Create next state"), Policy->CreateInitialState(PolicyObj, NextState));
	if (!AsHistory(CurrentState) || !AsHistory(NextState))
	{
		AddError(TEXT("CreateInitialState did not return chat history states"));
		return false;
	}
	TestEqual(TEXT("Initial history is empty"), AsHistory(CurrentState)->Messages.Num(), 0);

	for (int32 Step = 1; Step <= 3; ++Step)
	{
		TInstancedStruct<FPoint> Action;
		TestTrue(*FString::Printf(TEXT("Think succeeds on step %d"), Step),
			Policy->Think(MakeObservation(static_cast<float>(Step)), CurrentState, Action, NextState));
		TestEqual(*FString::Printf(TEXT("Input state is not modified on step %d"), Step),
			AsHistory(CurrentState)->Messages.Num(), Step - 1);
		TestEqual(*FString::Printf(TEXT("Action reflects history length on step %d"), Step),
			ActionValue(Action), static_cast<float>(Step));
		Swap(CurrentState, NextState);
	}

	TestEqual(TEXT("History holds every observation"), AsHistory(CurrentState)->Messages.Num(), 3);

	CurrentState->Reset();
	TestEqual(TEXT("Reset clears the history"), AsHistory(CurrentState)->Messages.Num(), 0);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FPolicyStateAliasedStatesRejectedTest,
	"Schola.Policies.IPolicyState.Aliased States Rejected",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPolicyStateAliasedStatesRejectedTest::RunTest(const FString& Parameters)
{
	UChatHistoryTestPolicy* PolicyObj = NewObject<UChatHistoryTestPolicy>();
	IPolicy* Policy = Cast<IPolicy>(PolicyObj);

	TScriptInterface<IPolicyState> State;
	Policy->CreateInitialState(PolicyObj, State);

	TInstancedStruct<FPoint> Action;
	TestFalse(TEXT("Think fails when InState and OutState are the same object"),
		Policy->Think(ScholaPolicyStateTestPrivate::MakeObservation(1.0f), State, Action, State));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FPolicyStateBatchedStatesIndependentTest,
	"Schola.Policies.IPolicyState.BatchedThink Keeps States Per Agent",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPolicyStateBatchedStatesIndependentTest::RunTest(const FString& Parameters)
{
	using namespace ScholaPolicyStateTestPrivate;

	UChatHistoryTestPolicy* PolicyObj = NewObject<UChatHistoryTestPolicy>();
	IPolicy* Policy = Cast<IPolicy>(PolicyObj);

	TArray<TScriptInterface<IPolicyState>> CurrentStates;
	TArray<TScriptInterface<IPolicyState>> NextStates;
	CurrentStates.SetNum(2);
	NextStates.SetNum(2);
	for (int32 i = 0; i < 2; ++i)
	{
		Policy->CreateInitialState(PolicyObj, CurrentStates[i]);
		Policy->CreateInitialState(PolicyObj, NextStates[i]);
	}
	// Give agent 1 a head start so the histories differ
	CastChecked<UTestChatHistoryState>(CurrentStates[1].GetObject())->Messages.Add(TEXT("earlier message"));

	TArray<TInstancedStruct<FPoint>> Observations = { MakeObservation(1.0f), MakeObservation(2.0f) };
	TArray<TInstancedStruct<FPoint>> Actions;
	TestTrue(TEXT("BatchedThink succeeds"), Policy->BatchedThink(Observations, CurrentStates, Actions, NextStates));

	TestEqual(TEXT("Agent 0 history advanced by one"), AsHistory(NextStates[0])->Messages.Num(), 1);
	TestEqual(TEXT("Agent 1 history advanced by one"), AsHistory(NextStates[1])->Messages.Num(), 2);
	TestEqual(TEXT("Agent 0 action"), ActionValue(Actions[0]), 1.0f);
	TestEqual(TEXT("Agent 1 action"), ActionValue(Actions[1]), 2.0f);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FPolicyStateCopyFromTest,
	"Schola.Policies.IPolicyState.CopyFrom",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPolicyStateCopyFromTest::RunTest(const FString& Parameters)
{
	UTestChatHistoryState* Source = NewObject<UTestChatHistoryState>();
	UTestChatHistoryState* Target = NewObject<UTestChatHistoryState>();
	Source->Messages = { TEXT("hello"), TEXT("world") };

	TestTrue(TEXT("CopyFrom succeeds for the same state type"), Target->CopyFrom(*Source));
	TestTrue(TEXT("Messages copied"), Target->Messages == Source->Messages);

	Source->Messages.Add(TEXT("again"));
	TestEqual(TEXT("Copy is independent of the source"), Target->Messages.Num(), 2);

	return true;
}

#endif // WITH_AUTOMATION_TESTS
