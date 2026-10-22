// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#include "Misc/AutomationTest.h"
#include "Points/BoxPoint.h"
#include "Policies/SharedPolicyTypes.h"
#include "SharedPolicyStepperTestTypes.h"
#include "Steppers/SharedPolicyStepper.h"
#include "TestStepper.h"

// Named, not anonymous: unity builds would collide on an anonymous helper.
namespace SharedPolicyStepperTest
{
	enum class EHeadAction : int32
	{
		Combat = 0,
		Support = 1,
	};

	struct FFixture
	{
		const FName Combat = TEXT("Combat");
		const FName Support = TEXT("Support");
		UTestAgent* Guard = NewObject<UTestAgent>();
		UTestAgent* Medic = NewObject<UTestAgent>();
		UMockSharedPolicy* Policy = NewObject<UMockSharedPolicy>();
		USharedPolicyStepper* Stepper = NewObject<USharedPolicyStepper>();

		FFixture()
		{
			Policy->RegisterHead(Combat, EHeadAction::Combat);
			Policy->RegisterHead(Support, EHeadAction::Support);
		}

		TArray<FSharedNPCBinding> MixedBindings() const
		{
			return {FSharedNPCBinding(Guard, Combat), FSharedNPCBinding(Medic, Support)};
		}
	};
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyStepperTwoNpcTest, "Schola.Inference.SharedPolicy.Stepper.TwoNpcRouting",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyStepperTwoNpcTest::RunTest(const FString& Parameters)
{
	SharedPolicyStepperTest::FFixture Fixture;
	TestTrue(TEXT("Init with Guard->Combat and Medic->Support"),
		Fixture.Stepper->Init(Fixture.MixedBindings(), Fixture.Policy));

	Fixture.Stepper->Step();

	TestEqual(TEXT("Guard received the Combat action"), Fixture.Guard->GetLastActionReceived(),
		SharedPolicyStepperTest::EHeadAction::Combat);
	TestEqual(TEXT("Medic received the Support action"), Fixture.Medic->GetLastActionReceived(),
		SharedPolicyStepperTest::EHeadAction::Support);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyStepperCollectsObservationsTest,
	"Schola.Inference.SharedPolicy.Stepper.CollectsObservations",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyStepperCollectsObservationsTest::RunTest(const FString& Parameters)
{
	SharedPolicyStepperTest::FFixture Fixture;
	TestTrue(TEXT("Init succeeds"), Fixture.Stepper->Init(Fixture.MixedBindings(), Fixture.Policy));

	Fixture.Stepper->Step();

	TestEqual(TEXT("Policy received one observation per NPC"), Fixture.Policy->LastObservations.Num(), 2);
	TestEqual(TEXT("Policy received one head per NPC"), Fixture.Policy->LastHeads.Num(), 2);
	TestEqual(TEXT("First head is Combat"), Fixture.Policy->LastHeads[0], Fixture.Combat);
	TestEqual(TEXT("Second head is Support"), Fixture.Policy->LastHeads[1], Fixture.Support);
	TestTrue(TEXT("First observation is a box point"), Fixture.Policy->LastObservations[0].GetPtr<FBoxPoint>() != nullptr);
	TestTrue(TEXT("Second observation is a box point"), Fixture.Policy->LastObservations[1].GetPtr<FBoxPoint>() != nullptr);
	TestEqual(TEXT("One BatchedThinkWithHeads call"), Fixture.Policy->ThinkCalls, 1);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyStepperSharedHeadTest, "Schola.Inference.SharedPolicy.Stepper.SharedHead",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyStepperSharedHeadTest::RunTest(const FString& Parameters)
{
	SharedPolicyStepperTest::FFixture Fixture;
	const TArray<FSharedNPCBinding> Bindings = {
		FSharedNPCBinding(Fixture.Guard, Fixture.Combat), FSharedNPCBinding(Fixture.Medic, Fixture.Combat)};
	TestTrue(TEXT("Two NPCs may share Combat"), Fixture.Stepper->Init(Bindings, Fixture.Policy));

	Fixture.Stepper->Step();

	TestEqual(TEXT("Guard received Combat"), Fixture.Guard->GetLastActionReceived(),
		SharedPolicyStepperTest::EHeadAction::Combat);
	TestEqual(TEXT("Medic received Combat"), Fixture.Medic->GetLastActionReceived(),
		SharedPolicyStepperTest::EHeadAction::Combat);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyStepperInvalidHeadTest, "Schola.Inference.SharedPolicy.Stepper.InvalidHead",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyStepperInvalidHeadTest::RunTest(const FString& Parameters)
{
	SharedPolicyStepperTest::FFixture Fixture;
	const TArray<FSharedNPCBinding> Bindings = {FSharedNPCBinding(Fixture.Guard, TEXT("Unknown"))};
	TestFalse(TEXT("Unknown head is rejected at Init"), Fixture.Stepper->Init(Bindings, Fixture.Policy));
	TestEqual(TEXT("Failed Init stores no bindings"), Fixture.Stepper->NPCBindings.Num(), 0);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyStepperDuplicateAgentTest,
	"Schola.Inference.SharedPolicy.Stepper.DuplicateAgent",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyStepperDuplicateAgentTest::RunTest(const FString& Parameters)
{
	SharedPolicyStepperTest::FFixture Fixture;
	const TArray<FSharedNPCBinding> Bindings = {
		FSharedNPCBinding(Fixture.Guard, Fixture.Combat), FSharedNPCBinding(Fixture.Guard, Fixture.Support)};
	TestFalse(TEXT("Duplicate NPC is rejected at Init"), Fixture.Stepper->Init(Bindings, Fixture.Policy));
	TestEqual(TEXT("Failed Init stores no bindings"), Fixture.Stepper->NPCBindings.Num(), 0);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyStepperActionCountTest,
	"Schola.Inference.SharedPolicy.Stepper.ActionCountMismatch",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyStepperActionCountTest::RunTest(const FString& Parameters)
{
	SharedPolicyStepperTest::FFixture Fixture;
	TestTrue(TEXT("Init succeeds"), Fixture.Stepper->Init(Fixture.MixedBindings(), Fixture.Policy));
	Fixture.Policy->ForcedActionCount = 1;

	Fixture.Stepper->Step();

	TestEqual(TEXT("Think was called"), Fixture.Policy->ThinkCalls, 1);
	TestEqual(TEXT("Guard was not given a short action list"), Fixture.Guard->GetLastActionReceived(), -1);
	TestEqual(TEXT("Medic was not given a short action list"), Fixture.Medic->GetLastActionReceived(), -1);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyStepperThinkFailureTest,
	"Schola.Inference.SharedPolicy.Stepper.ThinkFailure",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyStepperThinkFailureTest::RunTest(const FString& Parameters)
{
	SharedPolicyStepperTest::FFixture Fixture;
	TestTrue(TEXT("Init succeeds"), Fixture.Stepper->Init(Fixture.MixedBindings(), Fixture.Policy));
	Fixture.Policy->bThinkSucceeds = false;

	Fixture.Stepper->Step();

	TestEqual(TEXT("Think was called"), Fixture.Policy->ThinkCalls, 1);
	TestEqual(TEXT("Guard keeps the default action"), Fixture.Guard->GetLastActionReceived(), -1);
	TestEqual(TEXT("Medic keeps the default action"), Fixture.Medic->GetLastActionReceived(), -1);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyStepperEmptyBindingsTest,
	"Schola.Inference.SharedPolicy.Stepper.EmptyBindings",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyStepperEmptyBindingsTest::RunTest(const FString& Parameters)
{
	SharedPolicyStepperTest::FFixture Fixture;
	TestFalse(TEXT("Empty bindings are rejected"),
		Fixture.Stepper->Init(TArray<FSharedNPCBinding>(), Fixture.Policy));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSharedPolicyStepperNullPolicyTest,
	"Schola.Inference.SharedPolicy.Stepper.NullPolicy",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
bool FSharedPolicyStepperNullPolicyTest::RunTest(const FString& Parameters)
{
	SharedPolicyStepperTest::FFixture Fixture;
	TestFalse(TEXT("Null policy is rejected"), Fixture.Stepper->Init(Fixture.MixedBindings(), nullptr));
	return true;
}
