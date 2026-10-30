// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Spaces/BoxSpaceDimension.h"
#include "PositionalEnums.h"
#include "ActuatorInterface.h"
#include "Points/BoxPoint.h"

#include "RotationActuator.generated.h"


/**
 * @brief Delegate signature for rotation input events.
 * 
 * Broadcasts the rotation delta when an action is received.
 */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnRotationInputSignature, const FRotator&, RotationInput);

/**
 * @brief Rotation actuator that applies rotation changes to an Actor.
 * 
 * Accepts a BoxPoint action where each dimension corresponds to Pitch, Yaw, or Roll rotation.
 * Rotation can be applied as raw delta values or normalized [0,1] values that are rescaled.
 */
UCLASS(BlueprintType, Blueprintable, meta=(BlueprintSpawnableComponent))
class SCHOLAINTERACTORS_API URotationActuator : public UActorComponent, public IScholaActuator
{
    GENERATED_BODY()

public:

	/** The Min/Max value for the Pitch of the tracked rotation (rotation around Y-axis) */
	UPROPERTY(EditAnywhere, meta = (EditCondition = "bHasPitch"), Category = "Actuator Settings")
	FBoxSpaceDimension PitchBounds = FBoxSpaceDimension(-180, 180);

	/** The Min/Max value for the Yaw of the tracked rotation (rotation around Z-axis) */
	UPROPERTY(EditAnywhere, meta = (EditCondition = "bHasYaw"), Category = "Actuator Settings")
	FBoxSpaceDimension YawBounds = FBoxSpaceDimension(-180, 180);

	/** The Min/Max value for the Roll of the tracked rotation (rotation around X-axis) */
	UPROPERTY(EditAnywhere, meta = (EditCondition = "bHasRoll"), Category = "Actuator Settings")
	FBoxSpaceDimension RollBounds = FBoxSpaceDimension(-180, 180);

	/** Toggle for whether this actuator rotates the Agent along the Pitch dimension */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Meta = (InlineEditConditionToggle), Category = "Actuator Settings")
	bool bHasPitch = true;

	/** Toggle for whether this actuator rotates the Agent along the Yaw dimension */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Meta = (InlineEditConditionToggle), Category = "Actuator Settings")
	bool bHasYaw = true;

	/** Toggle for whether this actuator rotates the Agent along the Roll dimension */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Meta = (InlineEditConditionToggle), Category = "Actuator Settings")
	bool bHasRoll = true;

	/** Type of teleportation to use. See SetActorLocation documentation for more details. */
	UPROPERTY(EditAnywhere, Category = "Actuator Settings")
	ETeleportType TeleportType = ETeleportType::None;

	/** Toggle for whether to sweep while teleporting the actor. See SetActorLocation documentation for more details. */
	UPROPERTY(EditAnywhere, Category = "Actuator Settings")
	bool bSweep;

	/** Toggle for whether to use a [0,1] scale that is then rescaled onto the whole range for each rotator. Otherwise, uses the raw output as the delta rotation. */
	UPROPERTY(EditAnywhere, Category = "Actuator Settings")
	bool bNormalizeAndRescale = false;

	/** A delegate invoked when this actuator receives input from a brain. Useful for debugging and logging. */
	UPROPERTY(BlueprintAssignable)
	FOnRotationInputSignature OnRotationDelegate;

	/**
	 * @brief Get the action space for this actuator.
	 * 
	 * Returns a BoxSpace with dimensions for each enabled rotation axis (Pitch, Yaw, Roll).
	 * If bNormalizeAndRescale is true, dimensions are [0,1], otherwise uses the configured bounds.
	 * 
	 * @param[out] OutActionSpace The action space definition to be populated
	 */
	virtual void GetActionSpace_Implementation(FInstancedStruct& OutActionSpace) const;

	/**
	 * @brief Convert a Box Point with values to an FRotator.
	 * 
	 * Interprets the BoxPoint values as rotation angles for enabled dimensions.
	 * Applies rescaling if bNormalizeAndRescale is enabled.
	 * 
	 * @param[in] Action BoxPoint that will be converted
	 * @return FRotator equivalent to the converted BoxPoint
	 */
	FRotator ConvertActionToFRotator(const FBoxPoint& Action);

	/**
	 * @brief Apply a rotation action to the owner actor.
	 * 
	 * Converts the action BoxPoint to a rotation delta and applies it to the owning actor's local rotation.
	 * 
	 * @param[in] InAction The action to apply (BoxPoint)
	 */
	virtual void TakeAction_Implementation(const FInstancedStruct& InAction);

	/**
	 * @brief Generate a unique ID string for this actuator.
	 * 
	 * @return FString describing the actuator configuration (e.g., "Rotation_Pitch_-180.0_180.0_Yaw_-180.0_180.0")
	 */
	FString GenerateId() const;
};
