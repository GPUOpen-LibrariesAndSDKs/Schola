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
			if (this->AddAgent(Agent) == INDEX_NONE)
			{
				this->Agents.Reset();
				this->CurrentStates.Reset();
				return false;
			}
		}

		return true;
	}

	UFUNCTION(BlueprintCallable, Category = "Schola|Stepper")
	int32 AddAgent(const TScriptInterface<IAgent>& InAgent) override
	{
		if (!this->Policy)
		{
			UE_LOGFMT(LogScholaInferenceUtils, Error, "USimpleStepper::AddAgent(): Stepper is not initialized, call Init first!");
			return INDEX_NONE;
		}
		if (!InAgent.GetObject())
		{
			UE_LOGFMT(LogScholaInferenceUtils, Error, "USimpleStepper::AddAgent(): Agent is null!");
			return INDEX_NONE;
		}

		TScriptInterface<IPolicyState> State;
		if (!this->Policy->CreateInitialState(State))
		{
			UE_LOGFMT(LogScholaInferenceUtils, Error, "USimpleStepper::AddAgent(): Policy failed to create the initial state for agent {0}!", InAgent.GetObject()->GetName());
			return INDEX_NONE;
		}

		this->CurrentStates.Add(State);
		return this->Agents.Add(InAgent);
	}

	UFUNCTION(BlueprintCallable, Category = "Schola|Stepper")
	bool RemoveAgentAt(int32 AgentIndex) override
	{
		if (!this->Agents.IsValidIndex(AgentIndex))
		{
			UE_LOGFMT(LogScholaInferenceUtils, Error, "USimpleStepper::RemoveAgentAt(): Invalid agent index {0}", AgentIndex);
			return false;
		}

		this->Agents.RemoveAt(AgentIndex);
		this->CurrentStates.RemoveAt(AgentIndex);
		return true;
	}

	UFUNCTION(BlueprintCallable, Category = "Schola|Stepper")
	int32 RemoveAgent(const TScriptInterface<IAgent>& InAgent) override
	{
		int32 NumRemoved = 0;
		for (int32 i = this->Agents.Num() - 1; i >= 0; i--)
		{
			if (this->Agents[i].GetObject() == InAgent.GetObject())
			{
				this->RemoveAgentAt(i);
				NumRemoved++;
			}
		}
		return NumRemoved;
	}

	UFUNCTION(BlueprintPure, Category = "Schola|Stepper")
	int32 FindAgent(const TScriptInterface<IAgent>& InAgent) const override
	{
		return this->Agents.IndexOfByPredicate([&InAgent](const TScriptInterface<IAgent>& Agent) { return Agent.GetObject() == InAgent.GetObject(); });
	}

	UFUNCTION(BlueprintPure, Category = "Schola|Stepper")
	int32 NumAgents() const override { return this->Agents.Num(); }

	UFUNCTION(BlueprintPure, Category = "Schola|Stepper")
	TScriptInterface<IAgent> GetAgent(int32 AgentIndex) const override
	{
		if (!this->Agents.IsValidIndex(AgentIndex))
		{
			UE_LOGFMT(LogScholaInferenceUtils, Error, "USimpleStepper::GetAgent(): Invalid agent index {0}", AgentIndex);
			return TScriptInterface<IAgent>();
		}
		return this->Agents[AgentIndex];
	}

	UFUNCTION(BlueprintPure, Category = "Schola|Stepper")
	TScriptInterface<IPolicyState> GetState(int32 AgentIndex) const override
	{
		if (!this->CurrentStates.IsValidIndex(AgentIndex))
		{
			UE_LOGFMT(LogScholaInferenceUtils, Error, "USimpleStepper::GetState(): Invalid agent index {0}", AgentIndex);
			return TScriptInterface<IPolicyState>();
		}
		return this->CurrentStates[AgentIndex];
	}

	UFUNCTION(BlueprintCallable, Category = "Schola|Stepper")
	void ResetState(int32 AgentIndex) override
	{
		if (!this->CurrentStates.IsValidIndex(AgentIndex))
		{
			UE_LOGFMT(LogScholaInferenceUtils, Error, "USimpleStepper::ResetState(): Invalid agent index {0}", AgentIndex);
			return;
		}
		this->CurrentStates[AgentIndex]->Reset();
	}

	UFUNCTION(BlueprintCallable, Category = "Schola|Stepper")
	void ResetStates() override
	{
		for (int32 i = 0; i < this->CurrentStates.Num(); i++)
		{
			this->ResetState(i);
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
	/** Each agent's policy state, parallel to Agents and advanced in place by each Step(). Always valid, since AddAgent checks it. */
	UPROPERTY()
	TArray<TScriptInterface<IPolicyState>> CurrentStates;
};