// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Interface.h"
#include "Common/InteractionDefinition.h"
#include "Common/LogSchola.h"
#include "Policies/PolicyStateInterface.h"
#include "StructUtils/InstancedStruct.h"
#include "PolicyInterface.generated.h"


/**
 * @brief UInterface wrapper for policy implementations.
 */
UINTERFACE(BlueprintType)
class SCHOLA_API UPolicy : public UInterface
{
	GENERATED_BODY()
};

/**
 * @class IPolicy
 * @brief Interface for policy implementations in the Schola framework.
 * 
 * A policy maps observations and an internal state to actions and an updated state,
 * typically through a trained neural network model or other decision-making algorithm.
 * The state is owned by the caller rather than the policy, so one policy instance can
 * serve many agents, each with its own state. Stateless policies use null states.
 */
class SCHOLA_API IPolicy
{
    GENERATED_BODY()

public:

    /**
     * @brief Generates an action and the next state from the given observations and current state.
     * 
     * This is the core inference method that takes observations from the
     * environment and produces an action for the agent to execute.
     * 
     * InState and OutState must be different objects, both created by CreateInitialState,
     * or both null for a stateless policy. Implementations write into OutState and must not
     * replace it, since Think may run off the game thread where creating UObjects is unsafe.
     * 
     * @param[in] InObservations The observations from the environment.
     * @param[in] InState The policy state from the previous step (or the initial state).
     * @param[out] OutAction Output parameter that receives the generated action.
     * @param[out] OutState Pre-created state object that receives the next state.
     * @return True if inference succeeded, false otherwise.
     */
    virtual bool Think(
        const TInstancedStruct<FPoint>&         InObservations,
        const TScriptInterface<IPolicyState>&   InState,
        TInstancedStruct<FPoint>&               OutAction,
        TScriptInterface<IPolicyState>&         OutState) PURE_VIRTUAL(IPolicy::Think, return false;);
    
    /**
     * @brief Generates actions and next states from a batch of observations and states.
     * 
     * This method processes multiple observations at once, which can be more
     * efficient than processing them individually. The default implementation
     * calls Think for each observation, but derived classes can override this
     * for optimized batch processing.
     * 
     * @param[in] InObservations Array of observations to process.
     * @param[in] InStates Array of current recurrent states, one per observation.
     * @param[out] OutActions Output array that receives the generated actions.
     * @param[out] OutStates Array of pre-created recurrent states, one per observation, that receive the next states.
     * @return True if all inferences succeeded, false otherwise.
     */
    virtual bool BatchedThink(
        const TArray<TInstancedStruct<FPoint>>&         InObservations,
        const TArray<TScriptInterface<IPolicyState>>&   InStates,
        TArray<TInstancedStruct<FPoint>>&               OutActions,
        TArray<TScriptInterface<IPolicyState>>&         OutStates)
    {
        if (InStates.Num() != InObservations.Num() || OutStates.Num() != InObservations.Num())
        {
            UE_LOGFMT(LogSchola, Error, "IPolicy::BatchedThink(): Got {0} observations, {1} input states and {2} output states, expected one state of each per observation",
                InObservations.Num(), InStates.Num(), OutStates.Num());
            return false;
        }

        // Default implementation for batch processing of observations.
        // Pre-size the output so Think can write each action in-place without TArray growth or Add copies.
        // Implement in derived classes to add specialized batched handling.
        OutActions.SetNum(InObservations.Num());
        for (int32 Index = 0; Index < InObservations.Num(); ++Index)
        {
            if (!this->Think(InObservations[Index], InStates[Index], OutActions[Index], OutStates[Index]))
            {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Creates a state object holding the state at the start of an episode.
     * 
     * Must be called on the game thread after Init. Since Think requires InState and
     * OutState to be different objects, callers create two states per agent, swap them
     * after each Think, and keep them alive through a UPROPERTY reference. The default
     * implementation is for stateless policies and returns a null state.
     * 
     * @param[in] InOuter The outer for the created state object, typically the caller.
     * @param[out] OutState Receives the new state, or null if the policy is stateless.
     * @return True if the state was created (or the policy is stateless), false otherwise.
     */
    virtual bool CreateInitialState(UObject* InOuter, TScriptInterface<IPolicyState>& OutState) const
    {
        OutState = nullptr;
        return true;
    }

    /**
     * @brief Initializes the policy from an interaction definition.
     * 
     * This method sets up the policy with the observation and action space
     * definitions, allowing it to validate inputs and outputs and configure
     * any internal structures needed for inference.
     * 
     * @param[in] InPolicyDefinition An object defining the policy's input/output shapes and parameters.
     * @return True if initialization succeeded, false otherwise.
     */
	virtual bool Init(const FInteractionDefinition& InPolicyDefinition) PURE_VIRTUAL(IPolicy::Init, return true; );
    
    /**
     * @brief Checks if the policy is currently performing inference.
     * 
     * This method can be used to determine if the policy is busy processing
     * a request, which is useful for asynchronous inference implementations.
     * 
     * @return True if inference is in progress, false otherwise.
     */
    virtual bool IsInferenceBusy() const PURE_VIRTUAL(IPolicy::IsInferenceBusy, return false; );

};
