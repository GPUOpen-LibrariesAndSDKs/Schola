// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Actuators/RotationActuator.h"
#include "LogScholaInteractors.h"
#include "GameFramework/Actor.h"
#include "Spaces/BoxSpace.h"

void URotationActuator::GetActionSpace_Implementation(FInstancedStruct& OutActionSpace) const
{
    OutActionSpace.InitializeAs<FBoxSpace>();
	FBoxSpace& SpaceDefinition = OutActionSpace.GetMutable<FBoxSpace>();

	if (bHasPitch)
	{
		if (bNormalizeAndRescale)
		{
			SpaceDefinition.Dimensions.Add(FBoxSpaceDimension::ZeroOneUnitDimension());
		}
		else
		{
			SpaceDefinition.Dimensions.Add(PitchBounds);
		}
	}

	if (bHasYaw)
	{
		if (bNormalizeAndRescale)
		{
			SpaceDefinition.Dimensions.Add(FBoxSpaceDimension::ZeroOneUnitDimension());
		}
		else
		{
			SpaceDefinition.Dimensions.Add(YawBounds);
		}
		
	}

	if (bHasRoll)
	{
		if (bNormalizeAndRescale)
		{
			SpaceDefinition.Dimensions.Add(FBoxSpaceDimension::ZeroOneUnitDimension());
		}
		else
		{
			SpaceDefinition.Dimensions.Add(RollBounds);
		}
	}
}

FRotator URotationActuator::ConvertActionToFRotator(const FBoxPoint& Action)
{
	float Pitch = 0;
	float Yaw = 0;
	float Roll = 0;

	int Offset = 0;
	if (bHasPitch)
	{
		Pitch = Action[Offset++];
		if (bNormalizeAndRescale)
		{
			Pitch = PitchBounds.RescaleValue(Pitch);
		}
	}

	if (bHasYaw)
	{
		Yaw = Action[Offset++];
		if (bNormalizeAndRescale)
		{
			Yaw = YawBounds.RescaleValue(Yaw);
		}
	}

	if (bHasRoll)
	{
		Roll = Action[Offset++];
		if (bNormalizeAndRescale)
		{
			Roll = RollBounds.RescaleValue(Roll);
		}
	}

	return FRotator(Pitch, Yaw, Roll);
}

void URotationActuator::TakeAction_Implementation(const FInstancedStruct& InAction)
{
	const FBoxPoint& Action = InAction.Get<FBoxPoint>();
	AActor* LocalTarget = this->GetOwner();
	
	if (LocalTarget != nullptr)
	{
		const FRotator& Rotation = ConvertActionToFRotator(Action);
		this->OnRotationDelegate.Broadcast(Rotation);
		LocalTarget->AddActorLocalRotation(Rotation, bSweep, nullptr, TeleportType);
	}
	else
	{
		UE_LOGFMT(LogScholaInteractors, Warning, "URotationActuator::TakeAction_Implementation(): No Owner found to apply rotation to - {0}", GetName());
	}
}

FString URotationActuator::GenerateId() const
{
	FString Output = FString("Rotation");

	if (bHasPitch)
	{
		Output.Appendf(TEXT("_Pitch_%f_%f"), PitchBounds.Low, PitchBounds.High);
	}
	if (bHasYaw)
	{
		Output.Appendf(TEXT("_Yaw_%f_%f"), YawBounds.Low, YawBounds.High);
	}
	if (bHasRoll)
	{
		Output.Appendf(TEXT("_Roll_%f_%f"), RollBounds.Low, RollBounds.High);
	}
	// Add if we are rescaling and Normalizing
	if (bNormalizeAndRescale)
	{
		Output.Append("_Rescaled");
	}

	return Output;
}
