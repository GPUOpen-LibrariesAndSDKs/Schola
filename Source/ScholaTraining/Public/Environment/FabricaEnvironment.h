// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Environment/CppOnlyMultiAgentEnvironmentInterface.h"
#include "TrainingDataTypes/AgentState.h"
#include "TrainingDataTypes/FabricaAgentState.h"
#include "Common/InteractionDefinition.h"
#include "StructUtils/InstancedStruct.h"
#include "FabricaEnvironment.generated.h"

/**
 * @brief Reserved prefix for reward-component keys written by Fabrica-generated reward code.
 * @details Values should be string-encoded scalars. AFabricaEnvironment passes
 *          FAgentState::Info to generated reward code and sets Reward to the sum of
 *          entries whose keys use the component prefix.
 */
namespace UFabricaRewardInfo
{
	/** Default prefix; CLI may document overriding via generated constants if needed. */
	inline const TCHAR* DefaultComponentPrefix = TEXT("fabrica_r:");

	/** Info key for the scalar task-success metric copied from FFabricaAgentState::TaskSuccessMetric. */
	inline const TCHAR* DefaultTaskSuccessInfoPrefix = TEXT("fabrica_ts");

	SCHOLATRAINING_API const FString& GetComponentPrefix();

	SCHOLATRAINING_API const FString& GetTaskSuccessInfoPrefix();
}

/**
 * @brief Base class for Fabrica-driven Schola environments.
 * @details Owns the ICppOnlyMultiAgentEnvironment lifecycle: user code implements protected hooks
 *          (C++ subclasses override the OnUser hook Implementation methods, e.g. OnUserReset_Implementation;
 *          Blueprint subclasses implement the corresponding events).
 *          User hooks are single-agent (one definition, one initial state, one action / FFabricaAgentState);
 *          InitializeEnvironment / Reset / Step adapt to per-agent TMaps using GetFabricaSingleAgentId().
 *          Fabrica-generated overrides in ``*.fabrica.gen.cpp`` supply
 *          FabricaGeneratedInit / FabricaGeneratedRewardForAgent.
 *          The base never requires the user to call generated entrypoints manually.
 */
UCLASS(Abstract)
class SCHOLATRAINING_API AFabricaEnvironment : public AActor, public ICppOnlyMultiAgentEnvironment
{
	GENERATED_BODY()

public:
	AFabricaEnvironment();

	// ========== ICppOnlyMultiAgentEnvironment ==========
	using AActor::Reset;

	virtual void InitializeEnvironment(TMap<FString, FInteractionDefinition>& OutAgentDefinitions) override;
	virtual void SeedEnvironment(int Seed) override;
	virtual void SetEnvironmentOptions(const TMap<FString, FString>& Options) override;
	virtual void Reset(TMap<FString, FInitialAgentState>& OutAgentState) override;
	virtual void Step(const TMap<FString, FInstancedStruct>& InActions, TMap<FString, FAgentState>& OutAgentStates) override;

	virtual void BeginPlay() override;

	/** Agent id used in TMaps passed to ICppOnlyMultiAgentEnvironment (trainers must use this key for actions). */
	virtual const FString& GetFabricaSingleAgentId() const;

protected:
	/** brief Actors resolved during FabricaGeneratedInit for use by generated reward code. */
	UPROPERTY()
	TMap<FString, TObjectPtr<AActor>> FabricaTrackedActors;

	// ----- User hooks (C++: override *_Implementation; Blueprint: implement event on subclass) -----

	/** Define the sole agent's observation and action spaces. */
	UFUNCTION(BlueprintNativeEvent, Category = "Schola|Fabrica")
	void OnUserInitializeEnvironment(FInteractionDefinition& OutAgentDefinition);
	virtual void OnUserInitializeEnvironment_Implementation(FInteractionDefinition& OutAgentDefinition);

	/** Apply per-environment seeding. */
	UFUNCTION(BlueprintNativeEvent, Category = "Schola|Fabrica")
	void OnUserSeedEnvironment(int Seed);
	virtual void OnUserSeedEnvironment_Implementation(int Seed);

	/** Apply key/value options from the trainer. */
	UFUNCTION(BlueprintNativeEvent, Category = "Schola|Fabrica")
	void OnUserSetEnvironmentOptions(const TMap<FString, FString>& Options);
	virtual void OnUserSetEnvironmentOptions_Implementation(const TMap<FString, FString>& Options);

	/** Populate initial observations (and optional non-Fabrica Info) for the sole agent. */
	UFUNCTION(BlueprintNativeEvent, Category = "Schola|Fabrica")
	void OnUserReset(FInitialAgentState& OutAgentState);
	virtual void OnUserReset_Implementation(FInitialAgentState& OutAgentState);

	/**
	 * @brief Apply the sole agent's action, advance simulation, and fill FFabricaAgentState (observations, totals, optional task success).
	 * @details AFabricaEnvironment::Step converts to FAgentState (TaskSuccessMetric -> Info key fabrica_ts),
	 *          passes Info to FabricaGeneratedRewardForAgent for fabrica_r: components, then sets Reward
	 *          from their sum. InAction may be invalid if the trainer sent no action for GetFabricaSingleAgentId().
	 */
	UFUNCTION(BlueprintNativeEvent, Category = "Schola|Fabrica")
	void OnUserStep(const FInstancedStruct& InAction, FFabricaAgentState& OutFabricaAgentState);
	virtual void OnUserStep_Implementation(const FInstancedStruct& InAction, FFabricaAgentState& OutFabricaAgentState);

	/** Optional hook after native BeginPlay (Super already ran). */
	UFUNCTION(BlueprintNativeEvent, Category = "Schola|Fabrica")
	void OnUserBeginPlay();
	virtual void OnUserBeginPlay_Implementation();

	// ----- Fabrica-generated overrides (bodies in *.fabrica.gen.cpp) -----

	/** Resolve tracked actors / caches; invoked exactly once during InitializeEnvironment (after FabricaTrackedActors is cleared). */
	virtual void FabricaGeneratedInit();

	/** Append fabrica_r: reward-component entries to AgentState::Info; the base class sets Reward from their sum. */
	virtual void FabricaGeneratedRewardForAgent(const FString& AgentId, TMap<FString, FString>& RewardComponents);

	/** Clears FabricaTrackedActors immediately before generated init runs during InitializeEnvironment; extend in subclass if additional bookkeeping is required. */
	virtual void ClearFabricaTrackedActors();
};
