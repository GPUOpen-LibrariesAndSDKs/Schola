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
	TestEqual(TEXT("Reset keeps buffer sizes"), State->Buffers[0].StateBuffer.Num(), 8);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNNEPolicyStateCopyFromTest, "Schola.Policies.NNE.NNEPolicyState.CopyFrom", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FNNEPolicyStateCopyFromTest::RunTest(const FString& Parameters)
{
	UNNEPolicyState* Source = NewObject<UNNEPolicyState>();
	UNNEPolicyState* Target = NewObject<UNNEPolicyState>();
	Source->Buffers.Emplace(TArray<int32> { -1, 2 });
	Source->Buffers[0].StateBuffer[0] = 3.0f;
	Source->Buffers[0].StateBuffer[1] = 4.0f;
	Target->Buffers.Emplace(TArray<int32> { -1, 2 });
	const float* TargetData = Target->Buffers[0].StateBuffer.GetData();

	TestTrue(TEXT("CopyFrom succeeds for another UNNEPolicyState"), Target->CopyFrom(*Source));
	TestEqual(TEXT("Values copied"), Target->Buffers[0].StateBuffer[1], 4.0f);
	TestTrue(TEXT("Buffer is copied in place"), Target->Buffers[0].StateBuffer.GetData() == TargetData);

	Source->Buffers[0].StateBuffer[1] = 5.0f;
	TestEqual(TEXT("Copy is independent of the source"), Target->Buffers[0].StateBuffer[1], 4.0f);

	UNNEPolicyState* Empty = NewObject<UNNEPolicyState>();
	TestFalse(TEXT("CopyFrom rejects a state with different buffers"), Empty->CopyFrom(*Source));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNNEPolicyStateNotifyStateUpdateTest, "Schola.Policies.NNE.NNEPolicyState.NotifyStateUpdate", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FNNEPolicyStateNotifyStateUpdateTest::RunTest(const FString& Parameters)
{
	UNNEPolicyState* State = NewObject<UNNEPolicyState>();
	State->Buffers.Emplace(TArray<int32> { -1, -1, 2 }, 3);
	State->Buffers.Emplace(TArray<int32> { -1, 2 });
	// The last two floats of each buffer are the pending state the model wrote
	State->Buffers[0].StateBuffer = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f };
	State->Buffers[1].StateBuffer = { 6.0f, 7.0f, 8.0f, 9.0f };

	TestTrue(TEXT("NotifyStateUpdate succeeds"), State->NotifyStateUpdate());
	TestEqual(TEXT("Oldest sequence entry dropped"), State->Buffers[0].StateBuffer[0], 2.0f);
	TestEqual(TEXT("Sequence shifted one entry earlier"), State->Buffers[0].StateBuffer[3], 5.0f);
	TestEqual(TEXT("Pending state committed as newest"), State->Buffers[0].StateBuffer[5], 7.0f);
	TestEqual(TEXT("Buffer without a sequence dimension takes the pending state"), State->Buffers[1].StateBuffer[1], 9.0f);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNNEPolicyCreateInitialStateRequiresInitTest, "Schola.Policies.NNE.NNEPolicy.CreateInitialState Requires Init", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FNNEPolicyCreateInitialStateRequiresInitTest::RunTest(const FString& Parameters)
{
	UNNEPolicy* Policy = NewObject<UNNEPolicy>();
	TScriptInterface<IPolicyState> State;

	AddExpectedMessage(TEXT("Network not loaded"), EAutomationExpectedMessageFlags::Contains, 1);
	TestFalse(TEXT("CreateInitialState fails before Init"), Policy->CreateInitialState(State));
	TestNull(TEXT("No state is created before Init"), State.GetObject());
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FNNEPolicyBlueprintCreateInitialStateRequiresInitTest, "Schola.Policies.NNE.NNEPolicy.Blueprint CreateInitialState Requires Init", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

bool FNNEPolicyBlueprintCreateInitialStateRequiresInitTest::RunTest(const FString& Parameters)
{
	UNNEPolicy* Policy = NewObject<UNNEPolicy>();

	AddExpectedMessage(TEXT("Network not loaded"), EAutomationExpectedMessageFlags::Contains, 1);
	TestNull(TEXT("Blueprint wrapper returns null before Init"), Policy->K2_CreateInitialState().GetObject());
	return true;
}

#endif // WITH_AUTOMATION_TESTS
