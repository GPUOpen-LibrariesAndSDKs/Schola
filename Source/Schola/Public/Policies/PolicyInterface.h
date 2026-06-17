// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Interface.h"
#include "Common/InteractionDefinition.h"
#include "Spaces/Space.h"
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
 * A policy maps observations to actions, typically through a trained
 * neural network model or other decision-making algorithm. This interface
 * defines the core methods that all policies must implement for inference
 * and initialization.
 */
class SCHOLA_API IPolicy
{
    GENERATED_BODY()

public:

    /**
     * @brief Generates an action from the given observations.
     * 
     * This is the core inference method that takes observations from the
     * environment and produces an action for the agent to execute.
     * 
     * @param[in] InObservations The observations from the environment.
     * @param[out] OutAction Output parameter that receives the generated action.
     * @return True if inference succeeded, false otherwise.
     */
    virtual bool Think(const TInstancedStruct<FPoint>& InObservations, TInstancedStruct<FPoint>& OutAction) PURE_VIRTUAL(IPolicy::Think, return false;);
    
    /**
     * @brief Generates actions from a batch of observations.
     * 
     * This method processes multiple observations at once, which can be more
     * efficient than processing them individually. The default implementation
     * calls Think for each observation, but derived classes can override this
     * for optimized batch processing.
     * 
     * @param[in] InObservations Array of observations to process.
     * @param[out] OutActions Output array that receives the generated actions.
     * @return True if all inferences succeeded, false otherwise.
     */
    virtual bool BatchedThink(const TArray<TInstancedStruct<FPoint>>& InObservations, TArray<TInstancedStruct<FPoint>>& OutActions)
    {
        // Default implementation for batch processing of observations.
        // Pre-size the output so Think can write each action in-place without TArray growth or Add copies.
        // Implement in derived classes to add specialized batched handling.
        OutActions.SetNum(InObservations.Num());
        for (int32 Index = 0; Index < InObservations.Num(); ++Index)
        {
            if (!this->Think(InObservations[Index], OutActions[Index]))
            {
                return false;
            }
        }
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
