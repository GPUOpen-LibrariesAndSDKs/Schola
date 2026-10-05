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
 * Derive from this class in Blueprint and implement the Think and Init events
 * to create a custom policy. Override CreateInitialState to give the policy state.
 */
UCLASS(Blueprintable, BlueprintType, Abstract, EditInlineNew)
class SCHOLA_API UBlueprintPolicy : public UObject, public IPolicy
{
	GENERATED_BODY()

public:

	/**
	 * @brief Native implementation of Think that forwards to the Blueprint event.
	 * @param[in] InObservations The observations from the environment.
	 * @param[in] InState The policy state from the previous step.
	 * @param[out] OutAction Output parameter that receives the generated action.
	 * @param[out] OutState Pre-created state object that receives the next state.
	 * @return True if inference succeeded.
	 */
	bool Think(
		const TInstancedStruct<FPoint>&		  InObservations,
		const TScriptInterface<IPolicyState>& InState,
		TInstancedStruct<FPoint>&			  OutAction,
		TScriptInterface<IPolicyState>&		  OutState) override
	{
		this->Think(ToUntypedInstancedStruct(InObservations), InState, ToUntypedInstancedStruct(OutAction), OutState);
		return true;
	}

	/**
	 * @brief Blueprint event for generating actions from observations.
	 * 
	 * Implement this event in Blueprint to define how observations are
	 * converted into actions. Read the current state from InState and write
	 * the next state into OutState; do not replace OutState with a new object.
	 * 
	 * @param[in] InObservations The observations from the environment.
	 * @param[in] InState The policy state from the previous step, or null if stateless.
	 * @param[out] OutAction Output parameter that receives the generated action.
	 * @param[in] OutState Pre-created state object to write the next state into, or null if stateless.
	 */
	UFUNCTION(BlueprintCallable, BlueprintImplementableEvent, Category = "Schola|Policy")
	void Think(const FInstancedStruct& InObservations, const TScriptInterface<IPolicyState>& InState, FInstancedStruct& OutAction, const TScriptInterface<IPolicyState>& OutState);

	/**
	 * @brief Native implementation of CreateInitialState that forwards to the Blueprint event.
	 * @param[in] InOuter The outer for the created state object.
	 * @param[out] OutState Receives the new state, or null if the policy is stateless.
	 * @return True if the state was created (or the policy is stateless).
	 */
	bool CreateInitialState(UObject* InOuter, TScriptInterface<IPolicyState>& OutState) const override
	{
		OutState = this->CreateInitialState(InOuter);
		return true;
	}

	/**
	 * @brief Blueprint event for creating the state at the start of an episode.
	 * 
	 * Override this event in Blueprint to construct a state object (of a C++ class
	 * implementing IPolicyState) with InOuter as its outer. The default
	 * implementation returns null, i.e. a stateless policy.
	 * 
	 * @param[in] InOuter The outer to construct the state object with.
	 * @return The new state object, or null if the policy is stateless.
	 */
	UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "Schola|Policy")
	TScriptInterface<IPolicyState> CreateInitialState(UObject* InOuter) const;

	virtual TScriptInterface<IPolicyState> CreateInitialState_Implementation(UObject* InOuter) const
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
