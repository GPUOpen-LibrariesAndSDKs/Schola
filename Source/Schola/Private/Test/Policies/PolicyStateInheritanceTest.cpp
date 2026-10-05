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
	TestTrue(TEXT("CreateInitialState succeeds"), Policy->CreateInitialState(PolicyObj, State));
	TestTrue(TEXT("Returned object is the derived state class"), State.GetObject() && State.GetObject()->IsA<UTestDerivedChatHistoryState>());
	TestNotNull(TEXT("Interface pointer survives the reflection round trip"), State.GetInterface());
	TestTrue(TEXT("Interface pointer matches a direct cast"), State.GetInterface() == Cast<IPolicyState>(State.GetObject()));
	TestTrue(TEXT("State uses the requested outer"), State.GetObject() && State.GetObject()->GetOuter() == PolicyObj);

	if (State)
	{
		State->Reset();
		TestEqual(TEXT("State is usable after the round trip"), CastChecked<UTestDerivedChatHistoryState>(State.GetObject())->ResetCount, 1);
	}

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
