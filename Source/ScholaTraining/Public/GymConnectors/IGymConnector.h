// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Interface.h"
#include "TrainingDataTypes/TrainingUpdate.h"
#include "GymConnectors/AutoResetTypeEnum.h"
#include "TrainingDataTypes/TrainingState.h"
#include "IGymConnector.generated.h"



/**
 * @brief Blueprint-accessible interface for connectors between Unreal Engine and gym environments.
 * @details This interface defines the contract for objects that manage communication
 * between Unreal Engine environments and external training systems.
 */
UINTERFACE(BlueprintType)
class UGymConnector : public UInterface
{
	GENERATED_BODY()
};

/**
 * @brief Interface for connectors between Unreal Engine and gym-compatible training systems.
 * @details Defines the core functionality required for connecting Unreal environments to external
 * training frameworks like Stable Baselines 3 or Ray RLlib. Implementations handle environment
 * collection, state synchronization, and action distribution.
 */
class SCHOLATRAINING_API IGymConnector
{
	GENERATED_BODY()
public:
	
	/**
	 * @brief Collect all available environments in the current world.
	 * @details Scans the level for environment objects and prepares them for training.
	 */
	virtual void CollectEnvironments() PURE_VIRTUAL(IGymConnector::CollectEnvironments, return; );
	
	/**
	 * @brief Resolve and retrieve the next environment state update.
	 * @return Pointer to the training state update containing actions or reset commands.
	 * @details This may block or return immediately depending on the connector implementation.
	 */
	virtual FTrainingStateUpdate* ResolveEnvironmentStateUpdate() PURE_VIRTUAL(IGymConnector::ResolveEnvironmentStateUpdate, return nullptr;);
	
	/**
	 * @brief Reset specified environments to their initial states.
	 * @param[in] InEnvironmentIds Array of environment IDs to reset.
	 * @param[out] OutState The resulting training state after reset.
	 */
	virtual void ResetEnvironments(const TArray<int>& InEnvironmentIds, FTrainingState& OutState) PURE_VIRTUAL(IGymConnector::ResetEnvironments, return; );
	
	/**
	 * @brief Update environments based on the provided state update.
	 * @param[in] StateUpdate The update containing actions or reset commands.
	 * @param[out] OutState The resulting training state after applying the update.
	 */
	virtual void UpdateEnvironments(const FTrainingStateUpdate& StateUpdate, FTrainingState& OutState) PURE_VIRTUAL(IGymConnector::UpdateEnvironments, return;);
	
	/**
	 * @brief Enable the connector to start processing.
	 */
	virtual void Enable() PURE_VIRTUAL(IGymConnector::Enable, return; );

	/**
	 * @brief Check if the connector is ready to start training.
	 * @return True if the connector can begin operation.
	 */
	virtual bool CheckForStart() PURE_VIRTUAL(IGymConnector::CheckForStart, return true;);
	
	/**
	 * @brief Submit the state of environments after a reset operation.
	 * @param[in] States The training state containing post-reset observations.
	 * @param[in] InEnvironmentIds Array of environment IDs that were reset.
	 */
	virtual void SubmitPostResetState(const FTrainingState& States, const TArray<int>& InEnvironmentIds) PURE_VIRTUAL(IGymConnector::SubmitPostResetState, return; );
	
};
