// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Containers/Map.h"
#include "Environment/ImitationEnvironmentInterface.h"
#include "ImitationDataTypes/ImitationAgentState.h"
#include "SingleAgentImitationEnvironmentInterface.generated.h"

/**
 * @brief UInterface for single-agent imitation learning environments.
 * @details Implement this interface in Blueprint or C++ to create environments
 * with a single agent that learns from expert demonstrations.
 */
UINTERFACE(Blueprintable, BlueprintType)
class SCHOLAIMITATION_API USingleAgentImitationScholaEnvironment : public UBaseImitationScholaEnvironment
{
	GENERATED_BODY()
};

/**
 * @brief Native interface for single-agent imitation learning environments.
 * @details Provides the contract for environments that manage a single agent.
 * Includes compatibility methods to bridge single-agent and multi-agent APIs.
 */
class SCHOLAIMITATION_API ISingleAgentImitationScholaEnvironment : public IBaseImitationScholaEnvironment
{
	GENERATED_BODY()
public:
	/**
	 * @brief Initialize the environment and define the agent's interaction space.
	 * @param[out] OutAgentDefinition The agent's interaction definition (observation/action spaces)
	 */
	UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "Schola|Environment")
	void InitializeEnvironment(FInteractionDefinition& OutAgentDefinition);

	/**
	 * @brief Set the random seed for reproducible environment behavior.
	 * @param[in] InSeed The random seed value
	 */
	UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "Schola|Environment")
	void SeedEnvironment(int InSeed);

	/**
	 * @brief Configure environment-specific options.
	 * @param[in] InOptions Map of option names to their string values
	 */
	UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "Schola|Environment")
	void SetEnvironmentOptions(const TMap<FString, FString>& InOptions);

	/**
	 * @brief Reset the environment and return the agent's initial state.
	 * @param[out] OutAgentState The agent's initial state
	 */
	UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category = "Schola|Environment")
	void Reset(FInitialAgentState& OutAgentState);

	/**
	 * @brief Execute one environment step and return the agent's imitation state.
	 * @param[out] OutAgentState The agent's imitation state (observations, expert actions, rewards)
	 */
	UFUNCTION(BlueprintNativeEvent, Category = "Schola|Environment")
	void Step(FImitationAgentState& OutAgentState);

	// Compatibility Static Methods for TScholaEnvironment

	/**
	 * @brief Compatibility wrapper to adapt single-agent InitializeEnvironment to multi-agent API.
	 * @param[in] Obj The environment object
	 * @param[out] OutAgentDefinitions Map populated with a "SingleAgent" entry
	 */
	static void Execute_InitializeEnvironment(UObject* Obj, TMap<FString, FInteractionDefinition>& OutAgentDefinitions)
	{
		FInteractionDefinition& Defn = OutAgentDefinitions.FindOrAdd(FString("SingleAgent"));
		ISingleAgentImitationScholaEnvironment::Execute_InitializeEnvironment(Obj, Defn);
	}

	/**
	 * @brief Compatibility wrapper to adapt single-agent Reset to multi-agent API.
	 * @param[in] Obj The environment object
	 * @param[out] OutAgentState Map populated with a "SingleAgent" entry
	 */
	static void Execute_Reset(UObject* Obj, TMap<FString, FInitialAgentState>& OutAgentState)
	{
		FInitialAgentState& AgentState = OutAgentState.FindOrAdd(FString("SingleAgent"));
		ISingleAgentImitationScholaEnvironment::Execute_Reset(Obj, AgentState);
	}

	/**
	 * @brief Compatibility wrapper to adapt single-agent Step to multi-agent API.
	 * @param[in] Obj The environment object
	 * @param[out] OutAgentStates Map populated with a "SingleAgent" entry
	 */
	static void Execute_Step(UObject* Obj, TMap<FString, FImitationAgentState>& OutAgentStates)
	{
		FImitationAgentState& AgentState = OutAgentStates.FindOrAdd(FString("SingleAgent"));
		AgentState.Empty();
		ISingleAgentImitationScholaEnvironment::Execute_Step(Obj, AgentState);
	}
};
