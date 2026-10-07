// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "Test/Policies/PolicyStateInheritanceTestTypes.h"

#include "Points/BoxPoint.h"
#include "Policies/PolicyInterface.h"
#include "Policies/PolicyStateInterface.h"

#if WITH_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FPolicyStateNonZeroOffsetTest,
	"Schola.Policies.IPolicyState.Inheritance.Second Interface Base",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPolicyStateNonZeroOffsetTest::RunTest(const FString& Parameters)
{
	UTestMultiInterfaceState* StateObj = NewObject<UTestMultiInterfaceState>();
	StateObj->Value = 7;

	IPolicyState* ViaCast = Cast<IPolicyState>(StateObj);
	TScriptInterface<IPolicyState> ViaObject = StateObj;
	TScriptInterface<IPolicyState> ViaUntypedObject = static_cast<UObject*>(StateObj);

	TestNotNull(TEXT("Cast<IPolicyState> finds the interface"), ViaCast);
	TestTrue(TEXT("IPolicyState subobject is at a non-zero offset"), static_cast<void*>(ViaCast) != static_cast<void*>(StateObj));
	TestTrue(TEXT("TScriptInterface from typed pointer points at the IPolicyState subobject"), ViaObject.GetInterface() == ViaCast);
	TestTrue(TEXT("TScriptInterface from UObject* points at the IPolicyState subobject"), ViaUntypedObject.GetInterface() == ViaCast);
	TestTrue(TEXT("_getUObject returns the owning object"), ViaCast->_getUObject() == StateObj);
	TestEqual(TEXT("Unrelated interface still works"), Cast<ITestUnrelatedInterface>(StateObj)->GetMarker(), 42);

	ViaObject->Reset();
	TestEqual(TEXT("Reset through the interface dispatches to the object"), StateObj->Value, 0);

	UTestMultiInterfaceState* Other = NewObject<UTestMultiInterfaceState>();
	Other->Value = 9;
	TestTrue(TEXT("CopyFrom through the interface"), ViaObject->CopyFrom(*Cast<IPolicyState>(Other)));
	TestEqual(TEXT("CopyFrom copied the value"), StateObj->Value, 9);
	TestEqual(TEXT("ToString uses the implementing class"), ViaObject->ToString(), FString(TEXT("TestMultiInterfaceState")));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FPolicyStateSubInterfaceTest,
	"Schola.Policies.IPolicyState.Inheritance.Sub Interface",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPolicyStateSubInterfaceTest::RunTest(const FString& Parameters)
{
	UTestRecurrentState* StateObj = NewObject<UTestRecurrentState>();
	StateObj->History = { 1.0f, 2.0f };

	TestTrue(TEXT("Implements the base interface"), StateObj->Implements<UPolicyState>());
	TestTrue(TEXT("Implements the sub interface"), StateObj->Implements<UTestRecurrentPolicyState>());

	TScriptInterface<IPolicyState> AsBase = StateObj;
	TScriptInterface<ITestRecurrentPolicyState> AsSub = StateObj;
	TestNotNull(TEXT("TScriptInterface<IPolicyState> resolves"), AsBase.GetInterface());
	TestNotNull(TEXT("TScriptInterface<ITestRecurrentPolicyState> resolves"), AsSub.GetInterface());

	TScriptInterface<IPolicyState> SubToBase = AsSub;
	TestTrue(TEXT("Converting sub to base interface keeps the same subobject"), SubToBase.GetInterface() == AsBase.GetInterface());

	ITestRecurrentPolicyState* Downcast = Cast<ITestRecurrentPolicyState>(AsBase.GetObject());
	TestNotNull(TEXT("Base state can be cast back to the sub interface"), Downcast);
	TestEqual(TEXT("Sub interface method dispatches"), Downcast ? Downcast->GetSequenceLength() : -1, 2);

	AsBase->Reset();
	TestEqual(TEXT("Reset through the base interface"), StateObj->History.Num(), 0);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FPolicyStateDerivedClassTest,
	"Schola.Policies.IPolicyState.Inheritance.Derived State Class",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPolicyStateDerivedClassTest::RunTest(const FString& Parameters)
{
	UTestDerivedChatHistoryState* Derived = NewObject<UTestDerivedChatHistoryState>();
	Derived->Messages = { TEXT("a"), TEXT("b") };

	TScriptInterface<IPolicyState> State = Derived;
	TestNotNull(TEXT("Derived class inherits the interface"), State.GetInterface());

	State->Reset();
	TestEqual(TEXT("Reset dispatches to the most derived override"), Derived->ResetCount, 1);
	TestEqual(TEXT("Derived override still runs the base Reset"), Derived->Messages.Num(), 0);

	UTestChatHistoryState* Base = NewObject<UTestChatHistoryState>();
	Base->Messages = { TEXT("from base") };
	TestTrue(TEXT("Derived state copies from a base-class state"), State->CopyFrom(*Cast<IPolicyState>(Base)));
	TestEqual(TEXT("Messages copied from base"), Derived->Messages.Num(), 1);

	TestTrue(TEXT("Base state copies from a derived-class state"), Cast<IPolicyState>(Base)->CopyFrom(*State));

	UTestMultiInterfaceState* Unrelated = NewObject<UTestMultiInterfaceState>();
	TestFalse(TEXT("CopyFrom rejects an unrelated state type"), State->CopyFrom(*Cast<IPolicyState>(Unrelated)));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FPolicyStateReflectionRoundTripTest,
	"Schola.Policies.IPolicyState.Reflection.BlueprintNativeEvent Round Trip",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPolicyStateReflectionRoundTripTest::RunTest(const FString& Parameters)
{
	UTestReflectedStatePolicy* PolicyObj = NewObject<UTestReflectedStatePolicy>();
	IPolicy* Policy = Cast<IPolicy>(PolicyObj);

	// The native override calls the BlueprintNativeEvent, which goes through ProcessEvent and FInterfaceProperty
	TScriptInterface<IPolicyState> State;
	TestTrue(TEXT("CreateInitialState succeeds"), Policy->CreateInitialState(State));
	TestTrue(TEXT("Returned object is the derived state class"), State.GetObject() && State.GetObject()->IsA<UTestDerivedChatHistoryState>());
	TestNotNull(TEXT("Interface pointer survives the reflection round trip"), State.GetInterface());
	TestTrue(TEXT("Interface pointer matches a direct cast"), State.GetInterface() == Cast<IPolicyState>(State.GetObject()));
	TestTrue(TEXT("State is created in the transient package"), State.GetObject() && State.GetObject()->GetOuter() == GetTransientPackage());

	if (State)
	{
		State->Reset();
		TestEqual(TEXT("State is usable after the round trip"), CastChecked<UTestDerivedChatHistoryState>(State.GetObject())->ResetCount, 1);
	}

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FPolicyStateNotifyStateUpdateOverrideTest,
	"Schola.Policies.IPolicyState.Inheritance.NotifyStateUpdate Override",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPolicyStateNotifyStateUpdateOverrideTest::RunTest(const FString& Parameters)
{
	UChatHistoryTestPolicy* PolicyObj = NewObject<UChatHistoryTestPolicy>();
	IPolicy* Policy = Cast<IPolicy>(PolicyObj);

	UTestWindowedChatHistoryState* StateObj = NewObject<UTestWindowedChatHistoryState>();
	StateObj->Messages = { TEXT("a"), TEXT("b"), TEXT("c") };
	TScriptInterface<IPolicyState> State = StateObj;

	TInstancedStruct<FPoint> Observation = TInstancedStruct<FPoint>::Make<FBoxPoint>(TArray<float> { 1.0f });
	TInstancedStruct<FPoint> Action;
	TestTrue(TEXT("Think succeeds"), Policy->Think(Observation, State, Action));
	const FBoxPoint* ActionPoint = Action.GetPtr<FBoxPoint>();
	TestTrue(TEXT("Action was computed"), ActionPoint && ActionPoint->Values.Num() == 1);
	TestEqual(TEXT("Action reads history before NotifyStateUpdate"), ActionPoint && ActionPoint->Values.Num() == 1 ? ActionPoint->Values[0] : -1.0f, 4.0f);
	TestEqual(TEXT("NotifyStateUpdate called once per Think"), StateObj->NotifyStateUpdateCount, 1);
	TestEqual(TEXT("The state's override decides what carries over, then the new message is added"), StateObj->Messages.Num(), 2);
	TestEqual(TEXT("Most recent previous message kept"), StateObj->Messages.Num() > 0 ? StateObj->Messages[0] : FString(), FString(TEXT("c")));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FPolicyStateRejectedUpdateLeavesStateUnchangedTest,
	"Schola.Policies.IPolicyState.Inheritance.Rejected Update Leaves State Unchanged",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPolicyStateRejectedUpdateLeavesStateUnchangedTest::RunTest(const FString& Parameters)
{
	UChatHistoryTestPolicy* PolicyObj = NewObject<UChatHistoryTestPolicy>();
	IPolicy* Policy = Cast<IPolicy>(PolicyObj);

	UTestWindowedChatHistoryState* StateObj = NewObject<UTestWindowedChatHistoryState>();
	StateObj->Messages = { TEXT("a"), TEXT("b"), TEXT("c") };
	StateObj->bRejectUpdates = true;
	TScriptInterface<IPolicyState> State = StateObj;

	TInstancedStruct<FPoint> Observation = TInstancedStruct<FPoint>::Make<FBoxPoint>(TArray<float> { 1.0f });
	TInstancedStruct<FPoint> Action;
	TestFalse(TEXT("Think fails when the state rejects the update"), Policy->Think(Observation, State, Action));
	TestTrue(TEXT("Action inference completed before the rejected state update"), Action.IsValid());
	TestEqual(TEXT("Rejected update does not call the mutating path"), StateObj->NotifyStateUpdateCount, 0);
	TestEqual(TEXT("Rejected update preserves every history message"), StateObj->Messages.Num(), 3);
	TestEqual(TEXT("First history message is unchanged"), StateObj->Messages[0], FString(TEXT("a")));
	TestEqual(TEXT("Last history message is unchanged"), StateObj->Messages[2], FString(TEXT("c")));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FPolicyStateBlueprintPolicyUpdateOrderTest,
	"Schola.Policies.IPolicyState.Reflection.BlueprintPolicy Calls NotifyStateUpdate Between Events",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPolicyStateBlueprintPolicyUpdateOrderTest::RunTest(const FString& Parameters)
{
	UTestReflectedStatePolicy* PolicyObj = NewObject<UTestReflectedStatePolicy>();
	IPolicy* Policy = Cast<IPolicy>(PolicyObj);

	UTestWindowedChatHistoryState* StateObj = NewObject<UTestWindowedChatHistoryState>();
	StateObj->Messages = { TEXT("a"), TEXT("b"), TEXT("c") };
	TScriptInterface<IPolicyState> State = StateObj;

	TInstancedStruct<FPoint> Observation = TInstancedStruct<FPoint>::Make<FBoxPoint>(TArray<float> { 1.0f });
	TInstancedStruct<FPoint> Action;
	TestTrue(TEXT("Think succeeds"), Policy->Think(Observation, State, Action));
	TestEqual(TEXT("ComputeAction reads the state before NotifyStateUpdate"), PolicyObj->ComputeActionSawMessages, 3);
	TestEqual(TEXT("Think calls NotifyStateUpdate exactly once"), StateObj->NotifyStateUpdateCount, 1);
	TestEqual(TEXT("WriteState runs after NotifyStateUpdate"), StateObj->Messages.Num(), 2);
	TestEqual(TEXT("Carried-over message comes first"), StateObj->Messages.Num() > 0 ? StateObj->Messages[0] : FString(), FString(TEXT("c")));
	TestEqual(TEXT("WriteState's message comes last"), StateObj->Messages.Num() > 1 ? StateObj->Messages[1] : FString(), FString(TEXT("new")));

	TestTrue(TEXT("Think succeeds for a stateless call"), Policy->Think(Observation, TScriptInterface<IPolicyState>(), Action));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
	FPolicyStateBlueprintImplementationBlockedTest,
	"Schola.Policies.IPolicyState.Reflection.Blueprint Implementation Blocked",
	EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FPolicyStateBlueprintImplementationBlockedTest::RunTest(const FString& Parameters)
{
#if WITH_METADATA
	TestTrue(TEXT("IPolicyState cannot be implemented in Blueprint"), UPolicyState::StaticClass()->HasMetaData(TEXT("CannotImplementInterfaceInBlueprint")));
#endif

	// A Blueprint-only implementation would look like this: an object with no native interface pointer.
	// Callers must treat it as invalid rather than dereference it.
	TScriptInterface<IPolicyState> BlueprintOnlyLike;
	BlueprintOnlyLike.SetObject(NewObject<UChatHistoryTestPolicy>());
	TestNotNull(TEXT("Object is set"), BlueprintOnlyLike.GetObject());
	TestNull(TEXT("No native interface pointer"), BlueprintOnlyLike.GetInterface());
	TestFalse(TEXT("operator bool reports it as unusable"), static_cast<bool>(BlueprintOnlyLike));

	return true;
}

#endif // WITH_AUTOMATION_TESTS
