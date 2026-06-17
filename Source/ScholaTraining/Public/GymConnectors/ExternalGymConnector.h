// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "CoreMinimal.h"
#include "Async/Future.h"
#include "GymConnectors/AbstractGymConnector.h"
#include "TrainingSettings/ExternalGymConnectorSettings.h"
#include "ExternalGymConnector.generated.h"

/**
 * @brief Abstract class for connectors that communicate with external training processes.
 * @details Provides asynchronous communication with external training systems running in separate processes.
 * Derived classes implement specific communication protocols (e.g., sockets, HTTP).
 */
UCLASS(Abstract)
class SCHOLATRAINING_API UExternalGymConnector : public UAbstractGymConnector
{
	GENERATED_BODY()

public:

	/** Settings for external communication (timeouts, etc.). */
	UPROPERTY()
	FExternalGymConnectorSettings ExternalSettings;

	/** Flag to track if this is the first step after connection/reconnection. */
	bool bFirstStep = true;

	/**
	 * @brief Default constructor for external gym connectors.
	 */
	UExternalGymConnector();

	/**
	 * @brief Request the next training state update from the external process.
	 * @return A future that will eventually contain the training update (actions or resets).
	 * @details This function is asynchronous and returns immediately. The future resolves when
	 * the external process provides the next action or reset command. Must be implemented by derived classes.
	 */
	virtual TFuture<FTrainingStateUpdate*> RequestStateUpdate()
		PURE_VIRTUAL(UExternalGymConnector::RequestBatchedDecision, return TFuture<FTrainingStateUpdate*>(););

	/**
	 * @brief Resolve the environment state update, blocking if necessary.
	 * @return Pointer to the resolved training state update.
	 * @details Waits for the async request to complete and returns the result.
	 */
	FTrainingStateUpdate* ResolveEnvironmentStateUpdate() override;

	
};
