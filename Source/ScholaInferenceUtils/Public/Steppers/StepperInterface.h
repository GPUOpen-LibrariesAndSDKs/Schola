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
     * Replaces any agents added before, as if each agent in InAgents were passed to AddAgent.
     * Any policy works, since IPolicy::BatchedThink defaults to calling Think once per agent.
     * The policy must already be initialized, since the stepper creates each agent's policy state from it.
     * 
     * @param[in] InAgents Array of agents to manage
     * @param[in] InPolicy The policy to use for inference
     * @return true if initialization succeeded, false otherwise
     */
    virtual bool Init(const TArray<TScriptInterface<IAgent>>& InAgents, const TScriptInterface<IPolicy>& InPolicy) = 0;

    /*
     * The stepper manages its agents like a TArray: each agent is identified by its index, agents
     * keep the order they were added in, and removing an agent shifts every later agent down by one.
     * The same agent object can be added more than once, and each entry gets its own policy state.
     * All of these functions must be called on the Game Thread.
     */

    /**
     * @brief Start stepping an agent with the stepper's policy, like TArray::Add.
     *
     * The stepper creates and owns the agent's policy state, starting from the state at the
     * start of an episode. Must be called after Init.
     *
     * @param[in] InAgent The agent to add. Can already be managed by this stepper
     * @return The index of the added agent, or INDEX_NONE if it could not be added
     */
    virtual int32 AddAgent(const TScriptInterface<IAgent>& InAgent) = 0;

    /**
     * @brief Stop stepping an agent and release its policy state, like TArray::RemoveAt.
     *
     * @param[in] AgentIndex Index of the agent to remove
     * @return true if the index was valid and the agent has been removed, false otherwise
     */
    virtual bool RemoveAgentAt(int32 AgentIndex) = 0;

    /**
     * @brief Stop stepping every entry for an agent and release their policy states, like TArray::Remove.
     *
     * @param[in] InAgent The agent to remove
     * @return The number of entries removed
     */
    virtual int32 RemoveAgent(const TScriptInterface<IAgent>& InAgent) = 0;

    /**
     * @brief Find an agent, like TArray::Find.
     *
     * @param[in] InAgent The agent to look for
     * @return The index of the first entry for InAgent, or INDEX_NONE if it is not managed by this stepper
     */
    virtual int32 FindAgent(const TScriptInterface<IAgent>& InAgent) const = 0;

    /** @return The number of agents managed by this stepper, like TArray::Num */
    virtual int32 NumAgents() const = 0;

    /**
     * @param[in] AgentIndex Index of the agent
     * @return The agent at AgentIndex, or null if the index is invalid
     */
    virtual TScriptInterface<IAgent> GetAgent(int32 AgentIndex) const = 0;

    /**
     * @param[in] AgentIndex Index of the agent
     * @return The policy state of the agent at AgentIndex, or null if the index is invalid
     */
    virtual TScriptInterface<IPolicyState> GetState(int32 AgentIndex) const = 0;

    /**
     * @brief Reset one agent's policy state to the start of an episode.
     *
     * @param[in] AgentIndex Index of the agent
     */
    virtual void ResetState(int32 AgentIndex) = 0;

    /** @brief Reset every agent's policy state to the start of an episode. */
    virtual void ResetStates() = 0;

};


