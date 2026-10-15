// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Common/InteractionDefinition.h"
#include "Common/InstancedStructUtils.h"
#include "Spaces/Space.h"
#include "Policies/PolicyInterface.h"
#include "Points/Point.h"
#include "BlueprintPolicy.generated.h"

/**
 * @class UBlueprintPolicy
 * @brief Abstract base class for Blueprint-implementable policies.
 * 
 * This class allows policies to be implemented entirely in Blueprint, providing
 * a convenient way to create custom decision-making logic without C++ code.
 * Derive from this class in Blueprint and implement the Think and Init events
 * to create a custom policy.
 */
UCLASS(Blueprintable, BlueprintType, Abstract, EditInlineNew)
class SCHOLA_API UBlueprintPolicy : public UObject, public IPolicy
{
	GENERATED_BODY()

public:

	/**
	 * @brief Native implementation of Think that forwards to the Blueprint event.
	 * @param[in] InObservations The observations from the environment.
	 * @param[out] OutAction Output parameter that receives the generated action.
	 * @return True if inference succeeded.
	 */
	bool Think(const TInstancedStruct<FPoint>& InObservations, TInstancedStruct<FPoint>& OutAction) override
	{
		this->Think(ToUntypedInstancedStruct(InObservations), ToUntypedInstancedStruct(OutAction));
		return true;
	}

	/**
	 * @brief Blueprint event for generating actions from observations.
	 * 
	 * Implement this event in Blueprint to define how observations are
	 * converted into actions.
	 * 
	 * @param[in] InObservations The observations from the environment.
	 * @param[out] OutAction Output parameter that receives the generated action.
	 */
	UFUNCTION(BlueprintCallable,BlueprintImplementableEvent, Category = "Schola|Policy")
	void Think(const FInstancedStruct& InObservations, FInstancedStruct& OutAction);

	/** Native IPolicy bridge. Failed batches preserve the caller's actions. */
	bool BatchedThink(const TArray<TInstancedStruct<FPoint>>& InObservations,
		TArray<TInstancedStruct<FPoint>>& OutActions) override
	{
		TArray<FInstancedStruct> Observations;
		for (const auto& Observation : InObservations)
		{
			if (!Observation.IsValid() || !Observation.GetScriptStruct()->IsChildOf(FPoint::StaticStruct()))
			{
				return false;
			}
			Observations.Add(ToUntypedInstancedStruct(Observation));
		}
		TArray<FInstancedStruct> Actions;
		if (!this->BatchedThink(Observations, Actions) || Actions.Num() != InObservations.Num())
		{
			return false;
		}
		TArray<TInstancedStruct<FPoint>> Results;
		for (const auto& Action : Actions)
		{
			if (!Action.IsValid() || !Action.GetScriptStruct()->IsChildOf(FPoint::StaticStruct()))
			{
				return false;
			}
			Results.Add(ToTypedInstancedStruct<FPoint>(Action));
		}
		OutActions = MoveTemp(Results);
		return true;
	}

	/** Override to process a batch together. Defaults to the existing per-observation Think event. */
	UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "Schola|Policy")
	bool BatchedThink(const TArray<FInstancedStruct>& InObservations, TArray<FInstancedStruct>& OutActions);

	virtual bool BatchedThink_Implementation(const TArray<FInstancedStruct>& InObservations,
		TArray<FInstancedStruct>& OutActions)
	{
		for (const auto& Observation : InObservations)
		{
			if (!Observation.IsValid() || !Observation.GetScriptStruct()->IsChildOf(FPoint::StaticStruct()))
			{
				return false;
			}
		}
		TArray<FInstancedStruct> Results;
		for (const auto& Observation : InObservations)
		{
			FInstancedStruct Action;
			this->Think(Observation, Action);
			if (!Action.IsValid() || !Action.GetScriptStruct()->IsChildOf(FPoint::StaticStruct()))
			{
				return false;
			}
			Results.Add(MoveTemp(Action));
		}
		OutActions = MoveTemp(Results);
		return true;
	}

	/** Override when a Blueprint policy runs asynchronous work; synchronous default is false. */
	UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "Schola|Policy")
	bool IsInferenceBusy() const override;

	virtual bool IsInferenceBusy_Implementation() const { return false; }


	/**
	 * @brief Blueprint event for initializing the policy.
	 * 
	 * Implement this event in Blueprint to set up the policy when it is
	 * first created or when the interaction definition changes.
	 * 
	 * @param[in] InPolicyDefinition The interaction definition specifying observation and action spaces.
	 * @return True if initialization succeeded, false otherwise.
	 */
	UFUNCTION(BlueprintCallable, BlueprintImplementableEvent, Category = "Schola|Policy")
	bool Init(const FInteractionDefinition& InPolicyDefinition) override;

};

