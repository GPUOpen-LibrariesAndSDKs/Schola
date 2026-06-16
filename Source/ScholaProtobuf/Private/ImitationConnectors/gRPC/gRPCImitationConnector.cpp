// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ImitationConnectors/gRPC/gRPCImitationConnector.h"
#include "Environment/ImitationEnvironmentInterface.h"
#include "ProtobufUtils/ProtobufDeserializer.h"
#include "ProtobufUtils/ProtobufSerializer.h"
#include "Misc/Optional.h"
#include "LogScholaProtobuf.h"
#include "Misc/CommandLine.h"
#include "TrainingUtils/ArgBuilder.h"

URPCImitationConnector::URPCImitationConnector() 
{
}

void URPCImitationConnector::Init(const FTrainingDefinition& TrainingDefns)
{
	TRACE_CPUPROFILER_EVENT_SCOPE_STR("ScholaProtobuf: gRPCImitationConnector Init");
	// Cache the training definitions for reconnection scenarios
	this->CachedTrainingDefn = TrainingDefns;
	
	// Check if server is already initialized (reconnection scenario)
	bool bIsReconnection = (this->CommunicationManager != nullptr);
	
	if (!bIsReconnection)
	{
		// First-time initialization: create server and services
		std::shared_ptr<ImitationConnectorService::AsyncService> Service = 
			std::make_shared<ImitationConnectorService::AsyncService>();
		
		this->CommunicationManager = NewObject<UCommunicationManager>();
		
		int32 Port = 0;
		if (!FParse::Value(FCommandLine::Get(), TEXT("ScholaPort="), Port))
		{
			// Parse failed so we fall back to the default
			Port = this->ServerSettings.Port;
		}
		this->CommunicationManager->Initialize(Port, this->ServerSettings.Address);

		// Create the imitation state service (Consumer - we receive states from expert)
		ImitationStateService = this->CommunicationManager->CreateProducerBackend<
			ImitationConnectorService::AsyncService, 
			ImitationStateRequest, 
			Schola::ImitationState>(
				&ImitationConnectorService::AsyncService::RequestRequestState, 
				Service);

		// Create the training definition service (Producer - we publish definitions)
		TrainingDefinitionService = this->CommunicationManager->CreateProducerBackend<
			ImitationConnectorService::AsyncService, 
			ImitationDefinitionRequest, 
			Schola::TrainingDefinition>(
				&ImitationConnectorService::AsyncService::RequestRequestTrainingDefinition, 
				Service);

		// Create the start request service (Polling - we poll for start requests)
		StartRequestService = this->CommunicationManager->CreatePollingBackend<
			ImitationConnectorService::AsyncService, 
			ImitationConnectorStartRequest, 
			ImitationConnectorStartResponse>(
				&ImitationConnectorService::AsyncService::RequestStartImitationConnector, 
				Service);

		// Initialize all services
		this->ImitationStateService->Initialize();
		this->TrainingDefinitionService->Initialize();
		this->StartRequestService->Initialize();

		// Set up cleanup on connector close
		this->OnConnectorClosed.AddLambda([this]() {
			UE_LOGFMT(LogScholaProtobuf, Verbose, "URPCImitationConnector::Init(): Imitation Connector Closed");
		});
		
		this->CommunicationManager->StartBackends();

		// Use the config setting, but we can override the config value by passing ScholaDisableScript on the command line
		if (this->bRunScriptOnPlay && !FParse::Param(FCommandLine::Get(), TEXT("ScholaDisableScript")))
		{
			FScriptArgBuilder ArgBuilder = FScriptArgBuilder();
			this->ScriptSettings.GetArgs(ArgBuilder);
			this->ServerSettings.GetArgs(ArgBuilder);
			FString Args = ArgBuilder.Build();

			this->Script = this->ScriptSettings.GetLaunchableScript();
			Script.AppendArgs(Args);
			Script.LaunchScript();
		}
		
		// Publish the training definition
		this->TrainingDefinitionService->Publish(TrainingDefns);
	}
	else
	{
		// Reconnection scenario: server already running, prepare for new client
		UE_LOGFMT(LogScholaProtobuf, Verbose, "URPCImitationConnector::Init(): Server already initialized, preparing for new connection");
		
		// Reset connection state for new client
		this->SetStatus(EImitationConnectorStatus::NotStarted);
		
		// Clear any stale start requests from the previous connection
		TOptional<ImitationConnectorStartRequest> DrainRequest;
		while (true)
		{
			this->StartRequestService->Poll(DrainRequest);
			if (!DrainRequest.IsSet())
			{
				break;
			}
			UE_LOGFMT(LogScholaProtobuf, Verbose, "URPCImitationConnector::Init(): Drained stale start request from previous connection");
		}
		
		UE_LOGFMT(LogScholaProtobuf, Verbose, "URPCImitationConnector::Init(): Reset ImitationStateService for new connection");
	}
}

bool URPCImitationConnector::CheckForStart()
{
	TOptional<ImitationConnectorStartRequest> OptionalStartRequest = TOptional<ImitationConnectorStartRequest>();
	this->StartRequestService->Poll(OptionalStartRequest);
	
	if (OptionalStartRequest.IsSet())
	{
		const ImitationConnectorStartRequest& StartRequest = OptionalStartRequest.GetValue();
		
		// Apply any environment settings (seeds, options) from the start request
		for (int i = 0; i < Environments.Num(); i++)
		{
			auto it = StartRequest.environments().find(i);
			if (it != StartRequest.environments().end())
			{
				const Schola::EnvironmentSettings& EnvSettings = it->second;
				
				if (EnvSettings.has_seed())
				{
					Environments[i]->SeedEnvironment(EnvSettings.seed());
					UE_LOGFMT(LogScholaProtobuf, Verbose, "URPCImitationConnector::CheckForStart(): Environment {0} seeded with {1}", i, EnvSettings.seed());
				}
				
				if (EnvSettings.options_size() > 0)
				{
					TMap<FString, FString> Options;
					for (const auto& OptPair : EnvSettings.options())
					{
						Options.Add(FString(OptPair.first.c_str()), FString(OptPair.second.c_str()));
					}
					Environments[i]->SetEnvironmentOptions(Options);
					UE_LOGFMT(LogScholaProtobuf, Verbose, "URPCImitationConnector::CheckForStart(): Environment {0} has {1} options supplied", i, EnvSettings.options_size());
				}
			}
		}
		
		// Reset the first-step flag for the new connection
		this->bFirstStep = true;
		
		this->SetStatus(EImitationConnectorStatus::Running);
		
		// Republish training definitions for the new client
		this->TrainingDefinitionService->Publish(this->CachedTrainingDefn);
		UE_LOGFMT(LogScholaProtobuf, Verbose, "URPCImitationConnector::CheckForStart(): Republished TrainingDefinition for new client");
	}
	
	return this->Status == EImitationConnectorStatus::Running;
}

void URPCImitationConnector::SubmitStateWithInitialState(const FImitationTrainingState& InTrainingState, const FInitialState& InInitialState)
{
	// Not implemented for gRPC connector
	Schola::ImitationState StateMsg;
	ProtobufSerializer::ToProto(InTrainingState, StateMsg.mutable_training_state());
	ProtobufSerializer::ToProto(InInitialState, StateMsg.mutable_initial_state());
	this->ImitationStateService->Publish(StateMsg);
}

void URPCImitationConnector::SubmitState(const FImitationTrainingState& InTrainingState)
{
	Schola::ImitationState StateMsg;
	ProtobufSerializer::ToProto(InTrainingState, StateMsg.mutable_training_state());
	this->ImitationStateService->Publish(StateMsg);
}