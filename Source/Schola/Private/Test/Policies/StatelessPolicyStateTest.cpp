// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "Test/Policies/StatelessPolicyStateTestTypes.h"

#include "Points/BoxPoint.h"
#include "Policies/PolicyInterface.h"
#include "Policies/StatelessPolicyState.h"

#if WITH_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FStatelessPolicyStateSharedInstanceTest,
	"Schola.Policies.UStatelessPolicyState.Shared Instance",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FStatelessPolicyStateSharedInstanceTest::RunTest(const FString& Parameters)
{
	TScriptInterface<IPolicyState> State = UStatelessPolicyState::Get();
	TestNotNull(TEXT("Shared instance has a native interface"), State.GetInterface());
	TestTrue(TEXT("Get returns the same instance every time"), State.GetObject() == UStatelessPolicyState::Get().GetObject());
	TestTrue(TEXT("Shared instance is rooted"), State.GetObject() && State.GetObject()->IsRooted());

	State->Reset();
	TestTrue(TEXT("NotifyStateUpdate succeeds"), State->NotifyStateUpdate());
	TestTrue(TEXT("CopyFrom another stateless state succeeds"), State->CopyFrom(*NewObject<UStatelessPolicyState>()));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FStatelessPolicyStateBlueprintDefaultTest,
	"Schola.Policies.UStatelessPolicyState.BlueprintPolicy Default Round Trip",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FStatelessPolicyStateBlueprintDefaultTest::RunTest(const FString& Parameters)
{
	UTestDefaultBlueprintPolicy* PolicyObj = NewObject<UTestDefaultBlueprintPolicy>();
	IPolicy* Policy = Cast<IPolicy>(PolicyObj);

	// The native override calls the BlueprintNativeEvent, which goes through ProcessEvent and FInterfaceProperty
	TScriptInterface<IPolicyState> State;
	TestTrue(TEXT("CreateInitialState succeeds"), Policy->CreateInitialState(State));
	TestNotNull(TEXT("Interface pointer survives the reflection round trip"), State.GetInterface());
	TestTrue(TEXT("Default is the shared stateless state"), State.GetObject() == UStatelessPolicyState::Get().GetObject());

	TInstancedStruct<FPoint> Observation = TInstancedStruct<FPoint>::Make<FBoxPoint>(TArray<float> { 1.0f });
	TInstancedStruct<FPoint> Action;
	TestTrue(TEXT("Think succeeds with the stateless state"), Policy->Think(Observation, State, Action));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FStatelessPolicyStateBlueprintNoneTest,
	"Schola.Policies.UStatelessPolicyState.BlueprintPolicy Returning None",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FStatelessPolicyStateBlueprintNoneTest::RunTest(const FString& Parameters)
{
	UTestNoneStateBlueprintPolicy* PolicyObj = NewObject<UTestNoneStateBlueprintPolicy>();
	IPolicy* Policy = Cast<IPolicy>(PolicyObj);

	TScriptInterface<IPolicyState> State;
	AddExpectedMessage(TEXT("not a native IPolicyState"), EAutomationExpectedMessageFlags::Contains, 1);
	TestFalse(TEXT("CreateInitialState rejects a None return"), Policy->CreateInitialState(State));
	TestNull(TEXT("OutState is null after rejection"), State.GetObject());

	TInstancedStruct<FPoint> Observation = TInstancedStruct<FPoint>::Make<FBoxPoint>(TArray<float> { 1.0f });
	TInstancedStruct<FPoint> Action;
	AddExpectedMessage(TEXT("not a valid native IPolicyState"), EAutomationExpectedMessageFlags::Contains, 1);
	TestFalse(TEXT("UBlueprintPolicy::Think rejects a null state"), Policy->Think(Observation, State, Action));

	return true;
}

#endif // WITH_AUTOMATION_TESTS
