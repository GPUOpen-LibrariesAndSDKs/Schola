// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Misc/AutomationTest.h"

#include "Test/Actuators/ActuatorTestWorld.h"

#include "Actuators/RotationActuator.h"
#include "Engine/World.h"
#include "GameFramework/Actor.h"
#include "Points/BoxPoint.h"
#include "Spaces/BoxSpace.h"
#include "Spaces/BoxSpaceDimension.h"

#if WITH_DEV_AUTOMATION_TESTS

static URotationActuator* CreateRotationActuator(AActor* Outer)
{
	return NewObject<URotationActuator>(Outer, NAME_None, RF_Transient);
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRotationActuator_GetActionSpace_Normalized_Test,
	"Schola.Actuators.Rotation.GetActionSpace.Normalized",
	GScholaActuatorEditorTestFlags)

bool FRotationActuator_GetActionSpace_Normalized_Test::RunTest(const FString& Parameters)
{
	AActor* Owner = NewObject<AActor>(GetTransientPackage());
	URotationActuator* Actuator = CreateRotationActuator(Owner);
	Actuator->bNormalizeAndRescale = true;

	FInstancedStruct SpaceInst;
	Actuator->GetActionSpace_Implementation(SpaceInst);
	const FBoxSpace& Space = SpaceInst.Get<FBoxSpace>();
	TestEqual(TEXT("Pitch+Yaw+Roll -> three dims"), Space.Dimensions.Num(), 3);
	TestEqual(TEXT("Normalized dim low"), Space.Dimensions[0].Low, 0.f);
	TestEqual(TEXT("Normalized dim high"), Space.Dimensions[0].High, 1.f);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRotationActuator_GetActionSpace_PitchOnly_Test,
	"Schola.Actuators.Rotation.GetActionSpace.PitchOnly",
	GScholaActuatorEditorTestFlags)

bool FRotationActuator_GetActionSpace_PitchOnly_Test::RunTest(const FString& Parameters)
{
	AActor* Owner = NewObject<AActor>(GetTransientPackage());
	URotationActuator* Actuator = CreateRotationActuator(Owner);
	Actuator->bHasYaw = false;
	Actuator->bHasRoll = false;
	Actuator->PitchBounds = FBoxSpaceDimension(-45.f, 45.f);

	FInstancedStruct SpaceInst;
	Actuator->GetActionSpace_Implementation(SpaceInst);
	const FBoxSpace& Space = SpaceInst.Get<FBoxSpace>();
	TestEqual(TEXT("Pitch only -> one dimension"), Space.Dimensions.Num(), 1);
	TestEqual(TEXT("Pitch low"), Space.Dimensions[0].Low, -45.f);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRotationActuator_ConvertActionToFRotator_Raw_Test,
	"Schola.Actuators.Rotation.ConvertActionToFRotator.Raw",
	GScholaActuatorEditorTestFlags)

bool FRotationActuator_ConvertActionToFRotator_Raw_Test::RunTest(const FString& Parameters)
{
	AActor* Owner = NewObject<AActor>(GetTransientPackage());
	URotationActuator* Actuator = CreateRotationActuator(Owner);
	Actuator->bNormalizeAndRescale = false;

	FBoxPoint Action({12.f, 34.f, 56.f});
	const FRotator R = Actuator->ConvertActionToFRotator(Action);
	TestEqual(TEXT("Pitch"), static_cast<float>(R.Pitch), 12.f, 0.0001f);
	TestEqual(TEXT("Yaw"), static_cast<float>(R.Yaw), 34.f, 0.0001f);
	TestEqual(TEXT("Roll"), static_cast<float>(R.Roll), 56.f, 0.0001f);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRotationActuator_ConvertActionToFRotator_Rescaled_Test,
	"Schola.Actuators.Rotation.ConvertActionToFRotator.Rescaled",
	GScholaActuatorEditorTestFlags)

bool FRotationActuator_ConvertActionToFRotator_Rescaled_Test::RunTest(const FString& Parameters)
{
	AActor* Owner = NewObject<AActor>(GetTransientPackage());
	URotationActuator* Actuator = CreateRotationActuator(Owner);
	Actuator->bHasYaw = false;
	Actuator->bHasRoll = false;
	Actuator->bNormalizeAndRescale = true;
	Actuator->PitchBounds = FBoxSpaceDimension(0.f, 100.f);

	const FBoxPoint Action({0.5f});
	const FRotator R = Actuator->ConvertActionToFRotator(Action);
	TestEqual(TEXT("Midpoint of [0,100] from normalized 0.5"), static_cast<float>(R.Pitch), 50.f, 0.0001f);
	TestEqual(TEXT("Yaw unused"), static_cast<float>(R.Yaw), 0.f, 0.0001f);
	TestEqual(TEXT("Roll unused"), static_cast<float>(R.Roll), 0.f, 0.0001f);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRotationActuator_ConvertActionToFRotator_SpaceOrder_Test,
	"Schola.Actuators.Rotation.ConvertActionToFRotator.SpaceOrder",
	GScholaActuatorEditorTestFlags)

bool FRotationActuator_ConvertActionToFRotator_SpaceOrder_Test::RunTest(const FString& Parameters)
{
	AActor* Owner = NewObject<AActor>(GetTransientPackage());
	URotationActuator* Actuator = CreateRotationActuator(Owner);
	Actuator->bNormalizeAndRescale = false;

	const FBoxPoint Action({12.f, 34.f, 56.f});
	const FRotator R = Actuator->ConvertActionToFRotator(Action);
	TestEqual(TEXT("Pitch"), static_cast<float>(R.Pitch), 12.f, 0.0001f);
	TestEqual(TEXT("Yaw"), static_cast<float>(R.Yaw), 34.f, 0.0001f);
	TestEqual(TEXT("Roll"), static_cast<float>(R.Roll), 56.f, 0.0001f);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRotationActuator_ConvertActionToFRotator_SpaceOrderRescaled_Test,
	"Schola.Actuators.Rotation.ConvertActionToFRotator.SpaceOrderRescaled",
	GScholaActuatorEditorTestFlags)

bool FRotationActuator_ConvertActionToFRotator_SpaceOrderRescaled_Test::RunTest(const FString& Parameters)
{
	AActor* Owner = NewObject<AActor>(GetTransientPackage());
	URotationActuator* Actuator = CreateRotationActuator(Owner);
	Actuator->bNormalizeAndRescale = true;
	Actuator->PitchBounds = FBoxSpaceDimension(-10.f, 10.f);
	Actuator->YawBounds = FBoxSpaceDimension(0.f, 100.f);
	Actuator->RollBounds = FBoxSpaceDimension(-50.f, 50.f);

	const FBoxPoint Action({0.f, 1.f, 0.f});
	const FRotator R = Actuator->ConvertActionToFRotator(Action);
	TestEqual(TEXT("Pitch rescaled from 0 on [-10,10]"), static_cast<float>(R.Pitch), -10.f, 0.0001f);
	TestEqual(TEXT("Yaw rescaled from 1 on [0,100]"), static_cast<float>(R.Yaw), 100.f, 0.0001f);
	TestEqual(TEXT("Roll rescaled from 0 on [-50,50]"), static_cast<float>(R.Roll), -50.f, 0.0001f);

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRotationActuator_GenerateId_Test,
	"Schola.Actuators.Rotation.GenerateId",
	GScholaActuatorEditorTestFlags)

bool FRotationActuator_GenerateId_Test::RunTest(const FString& Parameters)
{
	AActor* Owner = NewObject<AActor>(GetTransientPackage());
	URotationActuator* Actuator = CreateRotationActuator(Owner);
	const FString Id = Actuator->GenerateId();
	TestTrue(TEXT("Id starts with Rotation"), Id.StartsWith(TEXT("Rotation")));

	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRotationActuator_TakeAction_NoOwner_Test,
	"Schola.Actuators.Rotation.TakeAction.NoOwner",
	GScholaActuatorEditorTestFlags)

bool FRotationActuator_TakeAction_NoOwner_Test::RunTest(const FString& Parameters)
{
	URotationActuator* Actuator = NewObject<URotationActuator>(GetTransientPackage(), NAME_None, RF_Transient);
	Actuator->bHasYaw = false;
	Actuator->bHasRoll = false;

	AddExpectedMessage(
		TEXT("URotationActuator::TakeAction_Implementation(): No Owner found to apply rotation to - RotationActuator_0"),
		ELogVerbosity::Warning,
		EAutomationExpectedMessageFlags::Contains,
		1,
		false);

	FInstancedStruct InAction;
	InAction.InitializeAs<FBoxPoint>(FBoxPoint({5.f}));
	Actuator->TakeAction_Implementation(InAction);

	return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
