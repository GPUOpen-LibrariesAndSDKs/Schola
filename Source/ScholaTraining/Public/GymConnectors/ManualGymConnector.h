// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "GymConnectors/AutoResetTypeEnum.h"
#include "TrainingDataTypes/TrainingDefinition.h"
#include "TrainingDataTypes/TrainingState.h"
#include "GymConnectors/AbstractGymConnector.h"
#include "ManualGymConnector.generated.h"


/**
 * @brief Manual gym connector for testing and debugging without external training processes.
 * @details Provides a gym connector interface that can be controlled directly from C++ or Blueprint.
 * Useful for testing environment implementations or creating custom training loops in Unreal.
 */
UCLASS(Blueprintable)
class SCHOLATRAINING_API UManualGymConnector : public UAbstractGymConnector
{
	GENERATED_BODY()

public:

	/** The auto-reset behavior for this connector. */
	UPROPERTY(EditAnywhere, Category = "Schola|Gym Connector")
	EAutoResetType AutoResetType = EAutoResetType::SameStep;

	/** The next update to be applied (set via ManualStep or ManualReset). */
	UPROPERTY()
	FTrainingStateUpdate NextUpdate = FTrainingStateUpdate();
	
	/**
	 * @brief Constructor for the manual gym connector. Initializes with default values.
	 */
	UManualGymConnector() {};

	/**
	 * @brief Initialize the connector (no-op for manual connector).
	 * @param[in] AgentDefinitions The training definitions.
	 */
	void Init(const FTrainingDefinition& AgentDefinitions)
	{
		//Do Nothing
	}

	// Methods for processing the environment state(s)

	/**
	 * @brief Submit state (no-op for manual connector).
	 * @param[in] InTrainingState The training state to submit.
	 */
	void SubmitState(const FTrainingState& InTrainingState)
	{
		//Do Nothing
	}

	/**
	 * @brief Submit state with initial state (no-op for manual connector).
	 * @param[in] InTrainingState The training state to submit.
	 * @param[in] InInitialAgentStates The initial states to submit.
	 */
	void SubmitStateWithInitialState(const FTrainingState& InTrainingState, const FInitialState& InInitialAgentStates)
	{
		//Do Nothing
	}

	/**
	 * @brief Submit initial state (no-op for manual connector).
	 * @param[in] InInitialAgentStates The initial states to submit.
	 */
	void SubmitInitialState(const FInitialState& InInitialAgentStates)
	{
		//Do Nothing
	}

	/**
	 * @brief Resolve the next environment state update.
	 * @return Pointer to the NextUpdate member set via ManualStep or ManualReset.
	 */
	FTrainingStateUpdate* ResolveEnvironmentStateUpdate()
	{
		return &(this->NextUpdate);
	}

	/**
	 * @brief Check if the connector is ready to start.
	 * @return Always returns true for manual connectors.
	 */
	bool CheckForStart()
	{
		this->SetStatus(EConnectorStatus::Running);
		return true;
	}

	/**
	 * @brief Manually execute a step with the given actions.
	 * @param[in] InActions Array of action maps (one per environment), each mapping agent IDs to actions.
	 * @param[out] OutInitialState The initial states for any environments that reset.
	 * @param[out] OutTrainingState The resulting training state after the step.
	 */
	void ManualStep(const TArray<TMap<FString,TInstancedStruct<FPoint>>>& InActions, FInitialState& OutInitialState, FTrainingState& OutTrainingState)
	{
		this->NextUpdate.UpdateType = ETrainingUpdateType::STEP;

		FTrainingStep& StepConfig = this->NextUpdate.StepData;
		StepConfig.EnvSteps.Empty(this->Environments.Num());
		for (int i = 0; i < this->Environments.Num(); i++)
		{
			StepConfig.EnvSteps.Emplace(InActions[i]);
		}

		this->Step();
		//This is not very efficient but this is for debugging and testing so it is ok
		OutInitialState.EnvironmentStates = this->InitialState.EnvironmentStates;
		OutTrainingState.EnvironmentStates = this->TrainingState.EnvironmentStates;

	}

	/**
	 * @brief Manually reset specified environments with optional seeds and options.
	 * @param[in] Seeds Map of environment IDs to random seeds.
	 * @param[in] OutEnvOptions Map of environment IDs to their configuration options.
	 * @param[out] OutInitialState The initial states after reset.
	 */
	void ManualReset(const TMap<int, int>& Seeds, const TMap<int, TMap<FString, FString>>& OutEnvOptions, FInitialState& OutInitialState)
	{
		this->NextUpdate.UpdateType = ETrainingUpdateType::RESET;
		FTrainingReset& ResetConfig = this->NextUpdate.ResetData;
		ResetConfig.Environments.Empty(this->Environments.Num());

		for (int i = 0; i < this->Environments.Num(); i++)
		{
			FEnvReset& EnvReset = ResetConfig.Environments.FindOrAdd(i);
			EnvReset.Options.Empty();
			if (Seeds.Contains(i))
			{
				EnvReset.Seed = Seeds[i];
				EnvReset.bHasSeed = true;
			}
			else
			{
				EnvReset.Seed = 0;
				EnvReset.bHasSeed = false;
			}

			if (OutEnvOptions.Contains(i))
			{
				EnvReset.Options = OutEnvOptions[i];
			}
		}
		// This will always be a reset step so we can ignore the training state
		this->Step();
		OutInitialState.EnvironmentStates = this->InitialState.EnvironmentStates;
	}

	/**
	 * @brief Close the connector and stop processing.
	 */
	void Close()
	{
		this->Status = EConnectorStatus::Closed;
	}

	/**
	 * @brief Get the auto-reset behavior for this connector.
	 * @return The configured auto-reset type.
	 */
	EAutoResetType GetAutoResetType() { return AutoResetType;};

};
