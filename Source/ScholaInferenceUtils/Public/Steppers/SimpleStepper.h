// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

# pragma once
#include "CoreMinimal.h"
#include "Steppers/StepperInterface.h"
#include "Agent/AgentInterface.h"
#include "LogScholaInferenceUtils.h"
#include "SimpleStepper.generated.h"

/**
 * @brief Simple synchronous stepper implementation.
 * 
 * A straightforward stepper that performs the full observation-inference-action loop
 * synchronously on the calling thread. On each Step() call:
 * 1. Collects observations from all agents
 * 2. Performs batched inference using the policy
 * 3. Applies the resulting actions to the agents
 * 
 * This stepper blocks during policy inference and is suitable for simple scenarios
 * or policies with fast inference times.
 */
UCLASS(BlueprintType, EditInlineNew, DefaultToInstanced)
class SCHOLAINFERENCEUTILS_API USimpleStepper : public UObject, public IStepper
{
	GENERATED_BODY()

public:
	/** Agents stepped each tick in array order. */
	UPROPERTY(BlueprintReadOnly, Category = "Stepper")
	TArray<TScriptInterface<IAgent>> Agents;

	/** Policy used for inference in Step(). */
	UPROPERTY(BlueprintReadOnly, Category = "Stepper")
	TScriptInterface<IPolicy>		 Policy;

	/**
	 * @brief Initialize the stepper with the given agents and policy.
	 * 
	 * @param[in] InAgents The array of agents to manage
	 * @param[in] InPolicy The initialized policy to use for inference
	 * @return true if initialization succeeded, false otherwise
	 */
	UFUNCTION(BlueprintCallable, Category = "Schola|Stepper")
	bool Init(const TArray<TScriptInterface<IAgent>>& InAgents, const TScriptInterface<IPolicy>& InPolicy) override
	{	
		this->Agents.Reset();
		this->CurrentStates.Reset();
		this->Policy = InPolicy;

		if(InAgents.Num() == 0)
		{
			UE_LOGFMT(LogScholaInferenceUtils, Error, "USimpleStepper::Init(): Initialized with no agents!");
			return false;
		}

		if (!this->Policy)
		{
			UE_LOGFMT(LogScholaInferenceUtils, Error, "USimpleStepper::Init(): Initialized with no policy!");
			return false;
		}

		for (const TScriptInterface<IAgent>& Agent : InAgents)
		{
			if (!this->AddAgent(Agent))
			{
				this->Agents.Reset();
				this->CurrentStates.Reset();
				return false;
			}
		}

		return true;
	}

	UFUNCTION(BlueprintCallable, Category = "Schola|Stepper")
	bool AddAgent(const TScriptInterface<IAgent>& InAgent) override
	{
		if (!this->Policy)
		{
			UE_LOGFMT(LogScholaInferenceUtils, Error, "USimpleStepper::AddAgent(): No policy, call Init first!");
			return false;
		}
		if (!InAgent.GetObject())
		{
			UE_LOGFMT(LogScholaInferenceUtils, Error, "USimpleStepper::AddAgent(): Agent is null!");
			return false;
		}
		if (this->FindAgentIndex(InAgent) != INDEX_NONE)
		{
			UE_LOGFMT(LogScholaInferenceUtils, Error, "USimpleStepper::AddAgent(): Agent {0} is already managed by this stepper!", InAgent.GetObject()->GetName());
			return false;
		}

		TScriptInterface<IPolicyState> State;
		if (!CreateAgentState(*this->Policy, State))
		{
			UE_LOGFMT(LogScholaInferenceUtils, Error, "USimpleStepper::AddAgent(): Policy failed to create the initial state for agent {0}!", InAgent.GetObject()->GetName());
			return false;
		}

		this->Agents.Add(InAgent);
		this->CurrentStates.Add(State);
		return true;
	}

	UFUNCTION(BlueprintCallable, Category = "Schola|Stepper")
	bool RemoveAgent(const TScriptInterface<IAgent>& InAgent) override
	{
		const int32 AgentIndex = this->FindAgentIndex(InAgent);
		if (AgentIndex == INDEX_NONE)
		{
			UE_LOGFMT(LogScholaInferenceUtils, Error, "USimpleStepper::RemoveAgent(): Agent is not managed by this stepper!");
			return false;
		}

		this->Agents.RemoveAt(AgentIndex);
		this->CurrentStates.RemoveAt(AgentIndex);
		return true;
	}

	/**
	 * @brief Get each agent's current policy state, for inspection and tests.
	 *
	 * @return One state per agent, in the same order as Agents (null entries for stateless policies)
	 */
	const TArray<TScriptInterface<IPolicyState>>& GetCurrentStates() const { return CurrentStates; }

	/**
	 * @brief Reset every agent's policy state to the start of an episode.
	 */
	UFUNCTION(BlueprintCallable, Category = "Schola|Stepper")
	void ResetStates()
	{
		for (int32 i = 0; i < this->CurrentStates.Num(); i++)
		{
			this->ResetAgentState(i);
		}
	}

	/**
	 * @brief Reset one agent's policy state to the start of an episode.
	 * 
	 * @param[in] AgentIndex Index of the agent in Agents
	 */
	UFUNCTION(BlueprintCallable, Category = "Schola|Stepper")
	void ResetAgentState(int32 AgentIndex)
	{
		if (!this->CurrentStates.IsValidIndex(AgentIndex))
		{
			UE_LOGFMT(LogScholaInferenceUtils, Error, "USimpleStepper::ResetAgentState(): Invalid agent index {0}", AgentIndex);
			return;
		}
		if (this->CurrentStates[AgentIndex])
		{
			this->CurrentStates[AgentIndex]->Reset();
		}
	}

	/**
	 * @brief Execute one step of the agent-policy loop.
	 * 
	 * Performs the full observation-inference-action cycle synchronously:
	 * - Collects observations from all agents
	 * - Calls the policy's BatchedThink method, which advances each agent's policy state in place
	 * - Applies the resulting actions
	 * 
	 * This method blocks during policy inference.
	 */
	UFUNCTION(BlueprintCallable, Category = "Schola|Stepper")
	void Step()
	{
		if (this->Agents.Num() == 0)
		{
			UE_LOGFMT(LogScholaInferenceUtils, Error, "USimpleStepper::Step(): No agents to step!");
			return;
		}
		if (!this->Policy)
		{
			UE_LOGFMT(LogScholaInferenceUtils, Error, "USimpleStepper::Step(): No policy to use for stepping!");
			return;
		}

		TArray<TInstancedStruct<FPoint>> Observations;
		TArray<TInstancedStruct<FPoint>> Actions;
		for (int i = 0; i < Agents.Num(); i++)
		{
			TInstancedStruct<FPoint> Observation;
			IAgent::Execute_Observe(Agents[i].GetObject(), Observation);
			Observations.Add(Observation);
		}
		if (this->Policy->BatchedThink(Observations, this->CurrentStates, Actions))
		{
			if (Actions.Num() == this->Agents.Num())
			{
				for (int i = 0; i < Agents.Num(); i++)
				{
					IAgent::Execute_Act(Agents[i].GetObject(), Actions[i]);
				}
			}
			else
			{
				UE_LOGFMT(LogScholaInferenceUtils, Error, "USimpleStepper::Step(): Number of actions ({0}) does not match number of agents ({1})!", Actions.Num(), Agents.Num());
			}
		}
		else
		{
			UE_LOGFMT(LogScholaInferenceUtils, Error, "USimpleStepper::Step(): Policy failed to think!");
		}
			
		
	}

private:
	/** Each agent's policy state, parallel to Agents and advanced in place by each Step(). Null entries for stateless policies. */
	UPROPERTY()
	TArray<TScriptInterface<IPolicyState>> CurrentStates;

	/** @return The index of InAgent in Agents, or INDEX_NONE if it is not managed by this stepper */
	int32 FindAgentIndex(const TScriptInterface<IAgent>& InAgent) const
	{
		return this->Agents.IndexOfByPredicate([&InAgent](const TScriptInterface<IAgent>& Agent) { return Agent.GetObject() == InAgent.GetObject(); });
	}
};