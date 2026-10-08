// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Common/InteractionDefinition.h"
#include "Common/InstancedStructUtils.h"
#include "Spaces/Space.h"
#include "Policies/PolicyInterface.h"
#include "Policies/PolicyStateInterface.h"
#include "Points/Point.h"
#include "BlueprintPolicy.generated.h"

/**
 * @class UBlueprintPolicy
 * @brief Abstract base class for Blueprint-implementable policies.
 * 
 * This class allows policies to be implemented entirely in Blueprint, providing
 * a convenient way to create custom decision-making logic without C++ code.
 * Derive from this class in Blueprint and implement the ComputeAction and Init events
 * to create a custom policy. Override CreateInitialState and WriteState to give the
 * policy state.
 */
UCLASS(Blueprintable, BlueprintType, Abstract, EditInlineNew)
class SCHOLA_API UBlueprintPolicy : public UObject, public IPolicy
{
	GENERATED_BODY()

public:

	/**
	 * @brief Native implementation of Think that drives the Blueprint events.
	 *
	 * Calls ComputeAction, then IPolicyState::NotifyStateUpdate on the state, then WriteState,
	 * so Blueprint implementations always carry history over the way the state class defines.
	 * @param[in] InObservations The observations from the environment.
	 * @param[in,out] InOutState The policy state, advanced in place to the next state.
	 * @param[out] OutAction Output parameter that receives the generated action.
	 * @return True if inference succeeded, false if NotifyStateUpdate failed.
	 */
	bool Think(
		const TInstancedStruct<FPoint>&		  InObservations,
		const TScriptInterface<IPolicyState>& InOutState,
		TInstancedStruct<FPoint>&			  OutAction) override
	{
		this->ComputeAction(ToUntypedInstancedStruct(InObservations), InOutState, ToUntypedInstancedStruct(OutAction));
		if (InOutState)
		{
			if (!InOutState->NotifyStateUpdate())
			{
				return false;
			}
			this->WriteState(ToUntypedInstancedStruct(InObservations), InOutState);
		}
		return true;
	}

	/**
	 * @brief Blueprint event for generating actions from observations.
	 * 
	 * Implement this event in Blueprint to define how observations and the
	 * previous step's state are converted into actions. Only read InState here;
	 * write the current step's data in WriteState.
	 * 
	 * @param[in] InObservations The observations from the environment.
	 * @param[in] InState The policy state from the previous step, or null if stateless.
	 * @param[out] OutAction Output parameter that receives the generated action.
	 */
	UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "Schola|Policy")
	void ComputeAction(const FInstancedStruct& InObservations, const TScriptInterface<IPolicyState>& InState, FInstancedStruct& OutAction);

	virtual void ComputeAction_Implementation(const FInstancedStruct& InObservations, const TScriptInterface<IPolicyState>& InState, FInstancedStruct& OutAction) {}

	/**
	 * @brief Blueprint event for writing the current step's data into the state.
	 *
	 * Called after ComputeAction and IPolicyState::NotifyStateUpdate, and only for
	 * non-null states. Update InOutState in place; do not replace it with a new
	 * object. Stateless policies can leave this unimplemented.
	 *
	 * @param[in] InObservations The observations from the environment.
	 * @param[in] InOutState The policy state to write into.
	 */
	UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "Schola|Policy")
	void WriteState(const FInstancedStruct& InObservations, const TScriptInterface<IPolicyState>& InOutState);

	virtual void WriteState_Implementation(const FInstancedStruct& InObservations, const TScriptInterface<IPolicyState>& InOutState) {}

	/**
	 * @brief Native implementation of CreateInitialState that forwards to the Blueprint event.
	 * @param[out] OutState Receives the new state, or null if the policy is stateless.
	 * @return True if the state was created (or the policy is stateless).
	 */
	bool CreateInitialState(TScriptInterface<IPolicyState>& OutState) const override
	{
		OutState = this->CreateInitialState();
		return true;
	}

	/**
	 * @brief Blueprint event for creating the state at the start of an episode.
	 * 
	 * Override this event in Blueprint to construct a state object (of a C++ class
	 * implementing IPolicyState) in the transient package. The default
	 * implementation returns null, i.e. a stateless policy.
	 * 
	 * @return The new state object, or null if the policy is stateless.
	 */
	UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "Schola|Policy")
	TScriptInterface<IPolicyState> CreateInitialState() const;

	virtual TScriptInterface<IPolicyState> CreateInitialState_Implementation() const
	{
		return nullptr;
	}

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
