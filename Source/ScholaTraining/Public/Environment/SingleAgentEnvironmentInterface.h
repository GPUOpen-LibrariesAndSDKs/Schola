// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Containers/Map.h"
#include "Environment/EnvironmentInterface.h"
#include "SingleAgentEnvironmentInterface.generated.h"

/**
 * @brief Blueprint interface for single-agent Schola environments.
 * @details Use this interface when implementing an environment with a single agent.
 * Provides a simplified API compared to multi-agent environments.
 */
UINTERFACE(Blueprintable, BlueprintType)
class SCHOLATRAINING_API USingleAgentScholaEnvironment : public UBaseScholaEnvironment
{
	GENERATED_BODY()
};

/**
 * @brief Interface for single-agent reinforcement learning environments.
 * @details Implement this interface in your environment class to create a single-agent RL environment.
 * All methods are Blueprint Native Events, allowing implementation in either C++ or Blueprint.
 */
class SCHOLATRAINING_API ISingleAgentScholaEnvironment : public IBaseScholaEnvironment
{
	GENERATED_BODY()

public:
	// Interface Methods

	/**
	 * @brief Initialize the environment and define the agent's observation and action spaces.
	 * @param[out] OutAgentDefinition The interaction definition containing observation/action space information.
	 */
	UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "Schola|Environment")
	void InitializeEnvironment(FInteractionDefinition& OutAgentDefinition);

	/**
	 * @brief Reset the environment to its initial state.
	 * @param[out] OutAgentState The initial agent state after reset, including the first observation.
	 */
	UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "Schola|Environment")
	void Reset(FInitialAgentState& OutAgentState);

	/**
	 * @brief Execute one environment step with the given action.
	 * @param[in] InAction The action selected by the agent for this step.
	 * @param[out] OutAgentState The resulting agent state (observation, reward, done flags, info).
	 */
	UFUNCTION(BlueprintNativeEvent, Category = "Schola|Environment")
	void Step(const FInstancedStruct& InAction, FAgentState& OutAgentState);

	/**
	 * @brief Set the random seed for reproducible environment behavior.
	 * @param[in] Seed The random seed value.
	 */
	UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "Schola|Environment")
	void SeedEnvironment(int Seed);

	/**
	 * @brief Configure the environment with custom options.
	 * @param[in] Options Map of configuration option names to values.
	 */
	UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "Schola|Environment")
	void SetEnvironmentOptions(const TMap<FString, FString>& Options);

	// Compatibility Static Methods for TScholaEnvironment

	/**
	 * @brief Adapter method to convert single-agent initialization to multi-agent format.
	 * @param[in] Obj The environment object.
	 * @param[out] OutAgentDefinitions Map of agent IDs to definitions (will contain one entry "SingleAgent").
	 */
	static void Execute_InitializeEnvironment(UObject* Obj, TMap<FString, FInteractionDefinition>& OutAgentDefinitions)
	{
		FInteractionDefinition& Defn = OutAgentDefinitions.FindOrAdd(FString("SingleAgent"));
		ISingleAgentScholaEnvironment::Execute_InitializeEnvironment(Obj, Defn);
	}

	/**
	 * @brief Adapter method to convert single-agent reset to multi-agent format.
	 * @param[in] Obj The environment object.
	 * @param[out] OutAgentState Map of agent IDs to states (will contain one entry "SingleAgent").
	 */
	static void Execute_Reset(UObject* Obj, TMap<FString, FInitialAgentState>& OutAgentState)
	{
		FInitialAgentState& AgentState = OutAgentState.FindOrAdd(FString("SingleAgent"));
		ISingleAgentScholaEnvironment::Execute_Reset(Obj, AgentState);
	}

	/**
	 * @brief Adapter method to convert single-agent step to multi-agent format.
	 * @param[in] Obj The environment object.
	 * @param[in] InActions Map of agent IDs to actions (expects one entry "SingleAgent").
	 * @param[out] OutAgentStates Map of agent IDs to states (will contain one entry "SingleAgent").
	 */
	static void Execute_Step(UObject* Obj, const TMap<FString, TInstancedStruct<FPoint>>& InActions, TMap<FString, FAgentState>& OutAgentStates)
	{
		const TInstancedStruct<FPoint>* Action = InActions.Find(FString("SingleAgent"));
		FAgentState&					AgentState = OutAgentStates.FindOrAdd(FString("SingleAgent"));
		if (Action)
		{
			ISingleAgentScholaEnvironment::Execute_Step(Obj, *reinterpret_cast<const FInstancedStruct*>(Action), AgentState);
		}
	}

};
