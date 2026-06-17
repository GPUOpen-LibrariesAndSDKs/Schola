// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "TrainingDataTypes/TrainingDefinition.h"
#include "TrainingDataTypes/TrainingState.h"
#include "ImitationDataTypes/ImitationState.h"
#include "AbstractImitationConnector.generated.h"

class IImitationScholaEnvironment;

/**
 * @brief Enum class representing the status of the connector.
 */
UENUM(BlueprintType)
enum class EImitationConnectorStatus : uint8
{
	Running	   UMETA(DisplayName = "Running"),    /** The connector is currently running */
	Closed	   UMETA(DisplayName = "Closed"),     /** The connector has been closed */
	Error	   UMETA(DisplayName = "Error"),      /** The connector encountered an error */
	NotStarted UMETA(DisplayName = "NotStarted"), /** The connector has not started yet */
};

/** @brief Multicast delegate broadcast when the connector starts. */
DECLARE_MULTICAST_DELEGATE(FImitationConnectorStartedSignature);

/** @brief Multicast delegate broadcast when the connector closes. */
DECLARE_MULTICAST_DELEGATE(FImitationConnectorClosedSignature);

/** @brief Multicast delegate broadcast when the connector encounters an error. */
DECLARE_MULTICAST_DELEGATE(FImitationConnectorErrorSignature);

/**
 * @brief An abstract class for connectors between Unreal Engine and imitation learning environments.
 * @details This class provides the basic structure and functionality for connecting Unreal Engine environments
 * 	with external expert policies for imitation learning purposes.
 */
UCLASS(Blueprintable, Abstract)
class SCHOLAIMITATION_API UAbstractImitationConnector : public UObject
{
	GENERATED_BODY()

public:
	
	/** Delegate broadcast when the connector starts. */
	FImitationConnectorStartedSignature OnConnectorStarted;

	/** Delegate broadcast when the connector closes. */
	FImitationConnectorClosedSignature  OnConnectorClosed;

	/** Delegate broadcast when the connector encounters an error. */
	FImitationConnectorErrorSignature   OnConnectorError;

	/** Array of type-erased environment pointers currently used for imitation. */
	TArray<IImitationScholaEnvironment*> Environments = TArray<IImitationScholaEnvironment*>();

	/** Training definition containing environment and agent definitions. */
	UPROPERTY()
	FTrainingDefinition TrainingDefinition = FTrainingDefinition();

	/** Current status of the connector. */
	UPROPERTY(VisibleAnywhere, Category = "Schola|Imitation Connector")
	EImitationConnectorStatus Status = EImitationConnectorStatus::Running;

	/** Flag indicating whether this is the first step (requires reset). */
	UPROPERTY(VisibleAnywhere, Category = "Schola|Imitation Connector")
	bool bFirstStep = true;
	
	/**
	 * @brief Constructor for the abstract imitation connector.
	 * @details Initializes the connector with default values.
	 */
	UAbstractImitationConnector();

	/**
	 * @brief Destructor for the abstract imitation connector.
	 * @details Cleans up environment pointers.
	 */
	~UAbstractImitationConnector();
	
	/**
	 * @brief Initialize this imitation connector with training definitions.
	 * @param[in] AgentDefinitions The definitions of the agents that will be used for imitation
	 * @note This function should be overridden by derived classes to set up services.
	 */
	virtual void Init(const FTrainingDefinition& AgentDefinitions);

	/**
	 * @brief Initialize this imitation connector by collecting environments and building training definitions.
	 * @param[in] InEnvironments Array of environment interfaces to use for imitation
	 * @note This function collects environment definitions and calls the other Init overload.
	 */
	UFUNCTION(BlueprintCallable, Category="Schola|Imitation Connector")
	void Init(const TArray<TScriptInterface<IBaseImitationScholaEnvironment>>& InEnvironments);

	/**
	 * @brief Set the status of the connector and broadcast appropriate delegates.
	 * @param[in] NewStatus The new status to set
	 */
	void SetStatus(EImitationConnectorStatus NewStatus);

	/**
	 * @brief Submit imitation state to external source with initial state information.
	 * @param[in] InState The imitation training state containing observations, actions, and rewards
	 * @param[in] InInitialState The initial state after any resets
	 * @note Must be implemented by derived classes.
	 */
	virtual void SubmitStateWithInitialState(const FImitationTrainingState& InState, const FInitialState& InInitialState) PURE_VIRTUAL(UAbstractImitationConnector::SubmitStateWithInitialState, return;);
	
	/**
	 * @brief Submit imitation state to external source without initial state information.
	 * @param[in] InState The imitation training state containing observations, actions, and rewards
	 * @note Must be implemented by derived classes.
	 */
	virtual void SubmitState(const FImitationTrainingState& InState) PURE_VIRTUAL(UAbstractImitationConnector::SubmitState, return;);

	/**
	 * @brief Check if the connector is ready to start processing.
	 * @return True if the connector is ready to start
	 * @note Must be implemented by derived classes.
	 */
	virtual bool CheckForStart() PURE_VIRTUAL(UAbstractImitationConnector::CheckForStart, return true;);

	/**
	 * @brief Check if the connector is currently running.
	 * @return True if the connector status is Running
	 */
	bool IsRunning() { return this->Status == EImitationConnectorStatus::Running; };

	/**
	 * @brief Check if the connector has not started or is closed.
	 * @return True if the connector status is NotStarted or Closed
	 */
	bool IsNotStarted() { return Status == EImitationConnectorStatus::NotStarted || Status == EImitationConnectorStatus::Closed; };
	
	/**
	 * @brief Collect all imitation environment actors from the current world.
	 * @param[out] OutCollectedEnvironments Array populated with found environment interfaces
	 */
	UFUNCTION()
	void CollectEnvironments(TArray<TScriptInterface<IBaseImitationScholaEnvironment>>& OutCollectedEnvironments);

	/**
	 * @brief Prepare environments for imitation by wrapping them in type-erased interfaces.
	 * @param[in] InEnvironments Array of environment interfaces to prepare
	 */
	void PrepareEnvironments(const TArray<TScriptInterface<IBaseImitationScholaEnvironment>>& InEnvironments);

	/**
	 * @brief Execute one step of imitation learning across all environments.
	 * @details Collects expert demonstrations from environments and submits them to the external source.
	 */
	UFUNCTION(BlueprintCallable, Category = "Schola|Imitation Connector")
	void Step();

protected:
	/**
	 * @brief Handle a single step of imitation learning.
	 * @param[out] OutTrainingState The updated training state after stepping
	 * @param[out] OutInitialState The initial state if any resets occurred
	 * @details This is a protected helper method for processing imitation steps in derived classes.
	 */
	void HandleImitationStep(FImitationTrainingState& OutTrainingState, FInitialState& OutInitialState);
};

