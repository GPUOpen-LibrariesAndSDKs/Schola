// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "GymConnectors/IGymConnector.h"
#include "GymConnectors/AutoResetTypeEnum.h"
#include "TrainingDataTypes/TrainingDefinition.h"
#include "TrainingDataTypes/TrainingState.h"
#include "AbstractGymConnector.generated.h"

class IScholaEnvironment;

/**
 * @brief Enum class representing the operational status of the connector.
 * @details Tracks the current state of communication with the training system.
 */
UENUM(BlueprintType)
enum class EConnectorStatus : uint8
{
	/** The connector is currently running and processing training steps. */
	Running	   UMETA(DisplayName = "Running"),
	/** The connector has been closed and is no longer processing. */
	Closed	   UMETA(DisplayName = "Closed"),
	/** The connector encountered an error during operation. */
	Error	   UMETA(DisplayName = "Error"),
	/** The connector has not started yet and is waiting to begin. */
	NotStarted UMETA(DisplayName = "NotStarted"),
};

/**
 * @brief Delegate broadcast when the connector starts successfully.
 */
DECLARE_MULTICAST_DELEGATE(FConnectorStartedSignature);

/**
 * @brief Delegate broadcast when the connector closes.
 */
DECLARE_MULTICAST_DELEGATE(FConnectorClosedSignature);

/**
 * @brief Delegate broadcast when the connector encounters an error.
 */
DECLARE_MULTICAST_DELEGATE(FConnectorErrorSignature);

/**
 * @brief An abstract class for connectors between Unreal Engine and gym environments.
 * @details class provides the basic structure and functionality for connecting Unreal Engine environments
 * 	with external gym environments for training purposes.
 */
UCLASS(Blueprintable, Abstract)
class SCHOLATRAINING_API UAbstractGymConnector : public UObject
{
	GENERATED_BODY()

public:
	
	/** Delegate broadcast when the connector starts successfully. */
	FConnectorStartedSignature OnConnectorStarted;

	/** Delegate broadcast when the connector closes. */
	FConnectorClosedSignature  OnConnectorClosed;

	/** Delegate broadcast when the connector encounters an error. */
	FConnectorErrorSignature   OnConnectorError;

	/** The environments that are currently being used for training by this connector. */
	TArray<IScholaEnvironment*> Environments = TArray<IScholaEnvironment*>();

	/** The current training state containing observations, rewards, and done flags for all environments. */
	UPROPERTY()
	FTrainingState TrainingState = FTrainingState();

	/** The initial state of environments at the start of the last episode. */
	UPROPERTY()
	FInitialState InitialState = FInitialState();

	/** Definition of the training session including environment and agent specifications. */
	UPROPERTY()
	FTrainingDefinition TrainingDefinition = FTrainingDefinition();

	/** The current operational status of the connector. */
	UPROPERTY(VisibleAnywhere, Category = "Schola|Gym Connector")
	EConnectorStatus Status = EConnectorStatus::Running;

	/** Flag indicating if this is the first step after initialization. */
	UPROPERTY(VisibleAnywhere, Category = "Schola|Gym Connector")
	bool bFirstStep = true;

	/** Flag indicating if the environments should be run in parallel. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (InlineEditConditionToggle), Category = "Schola|Gym Connector")
	bool bRunEnvironmentsInParallel = false;

	/** The minimum number of environments to process in each parallel batch. If bigger than the number of environments parallelization may be disabled.*/
	UPROPERTY(EditAnywhere,BlueprintReadWrite, meta = (ClampMin = "1", EditCondition = "bRunEnvironmentsInParallel"), Category = "Schola|Gym Connector")
	int32 MinParallelBatchSize = 1;

	/** Critical section for protecting access to InitialState.EnvironmentStates during parallel operations. */
	FCriticalSection InitialStateCriticalSection;
	
	/**
	 * @brief Constructor for the abstract gym connector. Initializes the connector with default values.
	 */
	UAbstractGymConnector();

	/**
	 * @brief Destructor for the gym connector.
	 */
	~UAbstractGymConnector();
	
	/**
	 * @brief Initialize this gym connector with training definitions.
	 * @param[in] AgentDefinitions The definitions of the agents and environments that will be trained.
	 * @details Sets up communication channels and prepares the connector for training.
	 * Override this in derived classes to implement specific initialization logic.
	 */
	virtual void Init(const FTrainingDefinition& AgentDefinitions);

	/**
	 * @brief Initialize this gym connector by collecting and setting up environments.
	 * @param[in] InEnvironments Array of environment interfaces to train.
	 * @details Gathers environment definitions and calls the other Init function.
	 */
	UFUNCTION(BlueprintCallable, Category="Schola|Gym Connector")
	void Init(const TArray<TScriptInterface<IBaseScholaEnvironment>>& InEnvironments);

	/**
	 * @brief Set the operational status of the connector.
	 * @param[in] NewStatus The new status to set.
	 * @details Also broadcasts appropriate delegates when status changes.
	 */
	void SetStatus(EConnectorStatus NewStatus);

	// Methods for processing the environment state(s)

	/**
	 * @brief Submit the current training state to the external training system.
	 * @param[in] InTrainingState The state containing observations, rewards, and done flags.
	 * @details Must be implemented by derived classes to handle state submission.
	 */
	virtual void SubmitState(const FTrainingState& InTrainingState) PURE_VIRTUAL(UAbstractGymConnector::SubmitEnvironmentStates, return; );

	/**
	 * @brief Submit both training state and initial state (for environments that reset).
	 * @param[in] InTrainingState The current training state.
	 * @param[in] InInitialAgentStates The initial states for reset environments.
	 * @details Must be implemented by derived classes.
	 */
	virtual void SubmitStateWithInitialState(const FTrainingState& InTrainingState, const FInitialState& InInitialAgentStates) PURE_VIRTUAL(UAbstractGymConnector::SubmitEnvironmentStates, return;);

	/**
	 * @brief Submit only the initial state (for reset-only operations).
	 * @param[in] InInitialAgentStates The initial states for reset environments.
	 * @details Must be implemented by derived classes.
	 */
	virtual void SubmitInitialState(const FInitialState& InInitialAgentStates) PURE_VIRTUAL(UAbstractGymConnector::SubmitEnvironmentStates, return;);

	// Methods for handling different types of updates

	/**
	 * @brief Handle a step update by applying actions to environments.
	 * @param[in] InState The step data containing actions for each environment.
	 * @param[out] OutTrainingState The resulting training state after the step.
	 * @param[out] OutInitialState The initial states for any environments that reset.
	 */
	void HandleStep(const FTrainingStep& InState, FTrainingState& OutTrainingState, FInitialState& OutInitialState);

	/**
	 * @brief Handle a reset update by resetting specified environments.
	 * @param[in] InReset The reset data specifying which environments to reset.
	 * @param[out] OutTrainingState The resulting training state (empty for pure resets).
	 * @param[out] OutInitialState The initial states after reset.
	 */
	void HandleReset(const FTrainingReset& InReset, FTrainingState& OutTrainingState, FInitialState& OutInitialState);

	/**
	 * @brief Resolve the environment state update. Useful for connections that operate off of futures, or otherwise require synchronization.
	 * @return The resolved environment state update.
	 * @note This function should be implemented by a derived class.
	 */
	virtual FTrainingStateUpdate* ResolveEnvironmentStateUpdate() PURE_VIRTUAL(UAbstractGymConnector::ResolveEnvironmentStateUpdate, return nullptr;);

	/**
	 * @brief Update the connector status based on a state update.
	 * @param[in] StateUpdate The state update to base the new status on.
	 */
	void UpdateConnectorStatus(const FTrainingStateUpdate& StateUpdate);

	/**
	 * @brief Check if the connector is ready to start.
	 * @return True if the connector is ready to start.
	 * @note This function should be implemented by a derived class.
	 */
	virtual bool CheckForStart() PURE_VIRTUAL(UAbstractGymConnector::CheckForStart, return true;);

	/**
	 * @brief Get if the connector is running.
	 * @return True if the connector is running.
	 */
	bool IsRunning() { return this->Status == EConnectorStatus::Running; };

	/**
	 * @brief Get if the connector is closed.
	 * @return True if the connector is closed.
	 */
	bool IsNotStarted() { return Status == EConnectorStatus::NotStarted || Status == EConnectorStatus::Closed; };
	
	/**
	 * @brief Collect all environment objects in the current world.
	 * @param[out] OutCollectedEnvironments Array of found environment interfaces.
	 * @details Searches the level for all objects implementing IBaseScholaEnvironment.
	 */
	UFUNCTION()
	void CollectEnvironments(TArray<TScriptInterface<IBaseScholaEnvironment>>& OutCollectedEnvironments);

	/**
	 * @brief Prepare environments for training by initializing and storing them.
	 * @param[in] InEnvironments Array of environment interfaces to prepare.
	 * @details Converts environment interfaces to the internal type-erased format.
	 */
	void PrepareEnvironments(const TArray<TScriptInterface<IBaseScholaEnvironment>>& InEnvironments);

	/**
	 * @brief Get the IDs of environments that have completed their episodes.
	 * @param[in] InState The training state to check for completed environments.
	 * @param[out] OutCompletedEnvironmentIds Array of environment IDs that are done.
	 */
	void GetCompletedEnvironmentIds(const FTrainingState& InState, TArray<int>& OutCompletedEnvironmentIds);

	/**
	 * @brief Execute one training step across all environments.
	 * @details Retrieves the next action/reset update and applies it to environments.
	 * Internal state is updated in place.
	 */
	UFUNCTION(BlueprintCallable, Category = "Schola|Gym Connector")
	void Step();

	/**
	 * @brief Execute one training step and retrieve the resulting states.
	 * @param[out] OutTrainingState The training state after the step.
	 * @param[out] OutInitialState The initial states for any reset environments.
	 */
	void Step(FTrainingState& OutTrainingState, FInitialState& OutInitialState);

	/**
	 * @brief Get the auto-reset behavior for this connector.
	 * @return The auto-reset type (default: SameStep).
	 * @details Controls when environments automatically reset after episode completion.
	 */
	virtual EAutoResetType GetAutoResetType() { return EAutoResetType::SameStep;};

};
