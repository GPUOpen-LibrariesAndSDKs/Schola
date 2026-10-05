// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "Policies/NNEPolicy.h"
#include "Policies/NNEPolicyState.h"

#if WITH_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNNEPolicyStateResetTest, "Schola.Policies.NNE.NNEPolicyState.Reset", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FNNEPolicyStateResetTest::RunTest(const FString& Parameters)
{
	UNNEPolicyState* State = NewObject<UNNEPolicyState>();
	State->Buffers.Emplace(TArray<int32> { -1, -1, 2 }, 3);
	State->Buffers.Emplace(TArray<int32> { -1, 4 });
	State->Buffers[0].StateBuffer[5] = 1.0f;
	State->Buffers[1].StateBuffer[0] = 2.0f;

	State->Reset();

	for (int32 i = 0; i < State->Buffers.Num(); ++i)
	{
		for (float Value : State->Buffers[i].StateBuffer)
		{
			TestEqual(*FString::Printf(TEXT("Buffer %d is zeroed"), i), Value, 0.0f);
		}
	}
	TestEqual(TEXT("Reset keeps buffer sizes"), State->Buffers[0].StateBuffer.Num(), 6);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNNEPolicyStateCopyFromTest, "Schola.Policies.NNE.NNEPolicyState.CopyFrom", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FNNEPolicyStateCopyFromTest::RunTest(const FString& Parameters)
{
	UNNEPolicyState* Source = NewObject<UNNEPolicyState>();
	UNNEPolicyState* Target = NewObject<UNNEPolicyState>();
	Source->Buffers.Emplace(TArray<int32> { -1, 2 });
	Source->Buffers[0].StateBuffer = { 3.0f, 4.0f };

	TestTrue(TEXT("CopyFrom succeeds for another UNNEPolicyState"), Target->CopyFrom(*Source));
	TestEqual(TEXT("Buffer count copied"), Target->Buffers.Num(), 1);
	TestEqual(TEXT("Values copied"), Target->Buffers[0].StateBuffer[1], 4.0f);

	Source->Buffers[0].StateBuffer[1] = 5.0f;
	TestEqual(TEXT("Copy is independent of the source"), Target->Buffers[0].StateBuffer[1], 4.0f);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNNEPolicyCreateInitialStateRequiresInitTest, "Schola.Policies.NNE.NNEPolicy.CreateInitialState Requires Init", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FNNEPolicyCreateInitialStateRequiresInitTest::RunTest(const FString& Parameters)
{
	UNNEPolicy* Policy = NewObject<UNNEPolicy>();
	TScriptInterface<IPolicyState> State;

	AddExpectedMessage(TEXT("Network not loaded"), EAutomationExpectedMessageFlags::Contains, 1);
	TestFalse(TEXT("CreateInitialState fails before Init"), Policy->CreateInitialState(Policy, State));
	TestNull(TEXT("No state is created before Init"), State.GetObject());
	return true;
}

#endif // WITH_AUTOMATION_TESTS
