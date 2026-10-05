// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Policies/PolicyInterface.h"
#include "Agent/AgentInterface.h"
#include "StepperInterface.generated.h"

/**
 * @brief UInterface for stepper implementations.
 * 
 * This interface allows Blueprint and C++ implementations of steppers that manage
 * the interaction loop between agents and policies.
 */
UINTERFACE(Blueprintable)
class SCHOLAINFERENCEUTILS_API UStepper : public UInterface
{
    GENERATED_BODY()
};

/**
 * @brief Interface for stepper implementations.
 * 
 * A stepper manages the observation-action loop by coordinating agents and policies.
 * It collects observations from agents, passes them to a policy for inference, and
 * applies the resulting actions back to the agents.
 */
class SCHOLAINFERENCEUTILS_API IStepper
{
    GENERATED_BODY()

public:

    /**
     * @brief Initialize the stepper with a single agent and policy.
     * 
     * Convenience method that wraps the agent in an array and calls the multi-agent Init.
     * 
     * @param[in] InAgent The agent to use for stepping
     * @param[in] InPolicy The policy to use for inference
     * @return true if initialization succeeded, false otherwise
     */
    virtual bool Init(const TScriptInterface<IAgent>& InAgent, const TScriptInterface<IPolicy>& InPolicy)
    {
        // Initialize the stepper with a single agent and policy
        TArray<TScriptInterface<IAgent>> Agents = {InAgent};
        return this->Init(Agents, InPolicy);
    }

    /**
     * @brief Initialize the stepper with multiple agents and a policy.
     * 
     * Sets up the stepper to manage multiple agents using a single shared policy.
     * The policy must support batched inference for multiple agents, and must already
     * be initialized, since the stepper creates each agent's policy state from it.
     * 
     * @param[in] InAgents Array of agents to manage
     * @param[in] InPolicy The policy to use for inference
     * @return true if initialization succeeded, false otherwise
     */
    virtual bool Init(const TArray<TScriptInterface<IAgent>>& InAgents, const TScriptInterface<IPolicy>& InPolicy) = 0;

protected:

    /**
     * @brief Create one initial policy state per agent.
     * 
     * @param[in] InPolicy The initialized policy to create states from
     * @param[in] InOuter The outer for the created state objects
     * @param[in] NumAgents Number of states to create
     * @param[out] OutStates Receives one state per agent (null entries for stateless policies)
     * @return true if every state was created, false otherwise
     */
    static bool CreateAgentStates(IPolicy& InPolicy, UObject* InOuter, int32 NumAgents, TArray<TScriptInterface<IPolicyState>>& OutStates)
    {
        OutStates.SetNum(NumAgents);
        for (int32 i = 0; i < NumAgents; ++i)
        {
            if (!InPolicy.CreateInitialState(InOuter, OutStates[i]))
            {
                OutStates.Reset();
                return false;
            }
        }
        return true;
    }

};


