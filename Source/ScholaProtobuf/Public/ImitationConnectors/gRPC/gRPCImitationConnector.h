// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "GymConnectors/gRPC/CommunicationManager.h"
#include "ImitationConnectors/AbstractImitationConnector.h"
#include "ProtobufBackends/gRPC/ConsumerRPCBackend.h"
#include "ProtobufBackends/gRPC/ProducerRPCBackend.h"
#include "GymConnectors/gRPC/RPCServerSettings.h"
#include "ImitationSettings/ImitationScriptSettings.h"

THIRD_PARTY_INCLUDES_START
#include "ScholaProtobufMacroGuardBegin.h"
#include "ImitationConnector.pb.h"
#include "ImitationConnector.grpc.pb.h"
#include "ImitationState.pb.h"
#include "ScholaProtobufMacroGuardEnd.h"
THIRD_PARTY_INCLUDES_END

#include "gRPCImitationConnector.generated.h"

using Schola::InitialState;
using Schola::ImitationConnectorService;
using Schola::ImitationConnectorStartRequest;
using Schola::ImitationConnectorStartResponse;
using Schola::ImitationStateRequest;
using Schola::ImitationDefinitionRequest;
using Schola::TrainingDefinition;
using Schola::ImitationState;

/**
 * @brief A connection to an external imitation learning expert policy implemented via gRPC.
 * @details This connector communicates with external expert policies (e.g., in Python) 
 * to receive observations, actions, and rewards for imitation learning.
 */
UCLASS(EditInlineNew)
class SCHOLAPROTOBUF_API URPCImitationConnector : public UAbstractImitationConnector
{
	GENERATED_BODY()
private:
	
	/** The service that will handle the initial ImitationConnectorStartRequest. */
	IConsumerBackend<ImitationConnectorStartRequest>* StartRequestService;
	
	/** The service that will handle imitation state requests. */
	IProducerBackend<Schola::ImitationState>* ImitationStateService;
	
	/** The service that will handle publishing training definitions. */
	IProducerBackend<Schola::TrainingDefinition>* TrainingDefinitionService;

	/** The communication manager that will handle the gRPC server. */
	UPROPERTY()
	TObjectPtr<UCommunicationManager> CommunicationManager;

	/** Cached training definition for reconnection scenarios */
	FTrainingDefinition CachedTrainingDefn;

public:
	
	/** Settings */

	UPROPERTY(EditAnywhere, Category = "Script Settings")
	bool bRunScriptOnPlay = false;

	/** Listen address and port for the imitation gRPC service. */
	UPROPERTY(EditAnywhere, Category = "Schola|gRPC")
	FRPCServerSettings ServerSettings;

	/** The settings for the python script to be launched */
	UPROPERTY(EditAnywhere, meta = (EditCondition = "bRunScriptOnPlay", ShowOnlyInnerProperties), Category = "Script Settings")
	FImitationScriptSettings ScriptSettings;

	/** Runtime handle for the imitation collection script when bRunScriptOnPlay is enabled. */
	UPROPERTY()
	FLaunchableScript Script;

	/** End of Settings */

	/**
	 * @brief Create a new URPCImitationConnector.
	 */
	URPCImitationConnector();


	/**
	 * @brief Initialize the gRPC connector with training definitions.
	 * @param[in] TrainingDefns The training definitions for all agents
	 */
	virtual void Init(const FTrainingDefinition& TrainingDefns) override;

	/**
	 * @brief Check if the connector received a start request and is ready to begin.
	 * @return True if the connector is ready to start.
	 */
	virtual bool CheckForStart() override;

	void SubmitStateWithInitialState(const FImitationTrainingState& InState, const FInitialState& InInitialState);
	void SubmitState(const FImitationTrainingState& InState);
};

