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

	/** Each agent's policy state, read by the next Step(). Null entries for stateless policies. */
	UPROPERTY(BlueprintReadOnly, Category = "Stepper")
	TArray<TScriptInterface<IPolicyState>> CurrentStates;

	/** Each agent's policy state written by the next Step(), swapped with CurrentStates on success. */
	UPROPERTY()
	TArray<TScriptInterface<IPolicyState>> NextStates;

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
		this->Agents = InAgents;
		this->Policy = InPolicy;

		if(this->Agents.Num() == 0)
		{
			UE_LOGFMT(LogScholaInferenceUtils, Error, "USimpleStepper::Init(): Initialized with no agents!");
			return false;
		}

		if (!this->Policy)
		{
			UE_LOGFMT(LogScholaInferenceUtils, Error, "USimpleStepper::Init(): Initialized with no policy!");
			return false;
		}

		if (!CreateAgentStates(*this->Policy, this, this->Agents.Num(), this->CurrentStates)
			|| !CreateAgentStates(*this->Policy, this, this->Agents.Num(), this->NextStates))
		{
			UE_LOGFMT(LogScholaInferenceUtils, Error, "USimpleStepper::Init(): Policy failed to create initial states!");
			return false;
		}

		return true;
	}

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
	 * - Calls the policy's BatchedThink method with each agent's policy state
	 * - Advances each agent's policy state and applies the resulting actions
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
		if (this->Policy->BatchedThink(Observations, this->CurrentStates, Actions, this->NextStates))
		{
			Swap(this->CurrentStates, this->NextStates);
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
};