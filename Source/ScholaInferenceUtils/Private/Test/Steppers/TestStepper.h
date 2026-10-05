// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "UObject/Object.h"
#include "Spaces/BoxSpace.h"
#include "Spaces/DiscreteSpace.h"
#include "Agent/AgentInterface.h"
#include "Steppers/SimpleStepper.h"
#include "Policies/PolicyInterface.h"
#include "TestStepper.generated.h"

/**
 * @brief Mock agent implementation for testing steppers.
 * 
 * A simple test agent with a 3D box observation space and a discrete action space.
 * Used for unit testing stepper implementations without requiring a full game environment.
 */
UCLASS()
class UTestAgent : public UObject, public IAgent
{
	GENERATED_BODY()

private:
	/** @brief Last action index received by the agent */
	int						 LastActionReceived = -1;
	
	/** @brief Interaction definition specifying observation and action spaces */
	FInteractionDefinition Defn;
	
	/** @brief Current status of the agent */
	EAgentStatus			 Status = EAgentStatus::Running;	

public:

	/**
	 * @brief Constructor.
	 * 
	 * Initializes the agent with a 3D box observation space [-1, 1] and
	 * a discrete action space with 3 possible actions.
	 */
	UTestAgent();

	/**
	 * @brief Define the agent's interaction spaces.
	 * 
	 * @param OutDefinition [out] The interaction definition containing observation and action spaces
	 */
	void Define_Implementation(FInteractionDefinition& OutDefinition) override;

	/**
	 * @brief Apply an action to the agent.
	 * 
	 * @param[in] InAction The action to apply (expected to be FMultiDiscretePoint)
	 */
	void Act_Implementation(const FInstancedStruct& InAction) override;

	/**
	 * @brief Get the current observation from the agent.
	 * 
	 * Returns a fixed mock observation for testing purposes.
	 * 
	 * @param[out] OutObservations The observation data (FBoxPoint with 3 values)
	 */
	void Observe_Implementation(FInstancedStruct& OutObservations) override;

	/**
	 * @brief Test helper accessor (not exposing in production interface).
	 * 
	 * @return The index of the last action received by the agent
	 */
	int GetLastActionReceived() const { return LastActionReceived; }

	/**
	 * @brief Set the agent's status.
	 * 
	 * @param[in] NewStatus The new status for the agent
	 */
	void SetStatus_Implementation(EAgentStatus NewStatus) override;

	/**
	 * @brief Get the agent's current status.
	 * 
	 * @return The current agent status
	 */
	EAgentStatus GetStatus_Implementation() override;
};

/**
 * @brief Mock policy implementation for testing steppers.
 * 
 * A simple test policy that always returns action index 1. Includes thread
 * tracking capabilities to verify that steppers correctly dispatch inference
 * to background threads. Simulates an inference busy state to test stepper
 * synchronization.
 */
UCLASS()
class UTestPolicy : public UObject, public IPolicy
{

	GENERATED_BODY()

private:
	/** Policy's interaction definition */
	FInteractionDefinition PolicyDefinition;

	/** Flag indicating whether inference is currently in flight */
	std::atomic<bool> bInferenceInFlight {false};

public:


	UTestPolicy();

	/**
	 * @brief Perform inference to get an action from observations.
	 * 
	 * Always returns action index 1 for testing. Tracks thread IDs to verify
	 * async execution.
	 * 
	 * @param[in] InObservations The observations to process
	 * @param[in] InState Unused, the test policy is stateless
	 * @param[out] OutAction The computed action (always action index 1)
	 * @param[out] OutState Unused, the test policy is stateless
	 * @return true if inference succeeded, false if inference was already in flight
	 */
	bool Think(
		const TInstancedStruct<FPoint>&		  InObservations,
		const TScriptInterface<IPolicyState>& InState,
		TInstancedStruct<FPoint>&			  OutAction,
		TScriptInterface<IPolicyState>&		  OutState) override;

	/**
	 * @brief Initialize the policy with a definition of the expected observation and action spaces. 
	 * 
	 * @param[in] InPolicyDefinition The definition of the observation and action spaces
	 * @return true if initialization succeeded
	 */
	bool Init(const FInteractionDefinition& InPolicyDefinition) override;

	/**
	 * @brief Check if inference is currently busy.
	 * 
	 * @return true if inference is in flight, false otherwise
	 */
	bool IsInferenceBusy() const override;

	/**
	 * @brief Reset thread tracking state.
	 * 
	 * Thread tracking helpers (testing / diagnostics only).
	 * Captures the current thread as the game thread for comparison.
	 */
	static void ResetThreadTracking();
	
	/**
	 * @brief Get a copy of all thread IDs that have executed Think.
	 * 
	 * @return Set of thread IDs that have called Think
	 */
	static TSet<uint32> GetThreadIdsCopy();
	
	/**
	 * @brief Check if Think was ever called from a non-game thread.
	 * 
	 * @return true if Think was called from a background thread, false if only from game thread
	 */
	static bool SawNonGameThread();
	
};

/**
 * @brief Test policy state counting how many steps an agent has taken.
 */
UCLASS()
class UTestCounterState : public UObject, public IPolicyState
{
	GENERATED_BODY()

public:
	/** Number of Think calls this state has been advanced through */
	UPROPERTY()
	int32 Count = 0;

	void Reset() override { Count = 0; }

	bool CopyFrom(const IPolicyState& Other) override
	{
		const UTestCounterState* OtherState = Cast<UTestCounterState>(Other._getUObject());
		if (!OtherState)
		{
			return false;
		}
		Count = OtherState->Count;
		return true;
	}
};

/**
 * @brief Stateful test policy for steppers.
 * 
 * Writes InState's count plus one into OutState and always returns action index 1.
 * Counts its Think calls so tests can check no state update was lost or applied twice.
 */
UCLASS()
class UTestStatefulPolicy : public UObject, public IPolicy
{
	GENERATED_BODY()

public:
	/** Total number of successful Think calls across all agents */
	std::atomic<int32> ThinkCount { 0 };

	bool Think(
		const TInstancedStruct<FPoint>&		  InObservations,
		const TScriptInterface<IPolicyState>& InState,
		TInstancedStruct<FPoint>&			  OutAction,
		TScriptInterface<IPolicyState>&		  OutState) override;

	bool CreateInitialState(UObject* InOuter, TScriptInterface<IPolicyState>& OutState) const override;

	bool Init(const FInteractionDefinition& InPolicyDefinition) override { return true; }

	bool IsInferenceBusy() const override { return false; }

	/**
	 * @brief Read the count from a state created by this policy.
	 * 
	 * @param[in] State The state to read
	 * @return The state's count, or -1 if it is not a UTestCounterState
	 */
	static int32 GetCount(const TScriptInterface<IPolicyState>& State);
};
