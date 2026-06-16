// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#include "GymConnectors/AbstractGymConnector.h"
#include "Environment/EnvironmentInterface.h"
#include "Environment/CppOnlyMultiAgentEnvironmentInterface.h"
#include "Environment/MultiAgentEnvironmentInterface.h"
#include "Environment/SingleAgentEnvironmentInterface.h"
#include "Kismet/GameplayStatics.h"
#include "LogScholaTraining.h"
#include "Async/ParallelFor.h"
#include "Engine/World.h"

UAbstractGymConnector::UAbstractGymConnector()
{
}

UAbstractGymConnector::~UAbstractGymConnector()
{
	for (IScholaEnvironment* Env : Environments)
	{
		delete Env;
	}
	Environments.Empty();
}

void UAbstractGymConnector::Init(const FTrainingDefinition& AgentDefinitions) {}

void UAbstractGymConnector::Init(const TArray<TScriptInterface<IBaseScholaEnvironment>>& InEnvironments)
{
	if (InEnvironments.Num() == 0)
	{
		UE_LOGFMT(LogScholaTraining, Warning, "UAbstractGymConnector::Init(): No Environments provided to Gym Connector");
		return;
	}

	this->Status = EConnectorStatus::NotStarted;
	this->PrepareEnvironments(InEnvironments);

	// Add a bunch of defaulted values
	this->TrainingState.EnvironmentStates.AddDefaulted(Environments.Num());
	this->TrainingDefinition.EnvironmentDefinitions.AddDefaulted(Environments.Num());

	for (int i = 0; i < Environments.Num(); i++)
	{
		Environments[i]->InitializeEnvironment(this->TrainingDefinition.EnvironmentDefinitions[i].AgentDefinitions);
	}

	this->Init(this->TrainingDefinition);
}

bool AllAgentsCompleted(const FEnvironmentState& EnvState)
{
	for (const TPair<FString, FAgentState>& AgentStatePairs : EnvState.AgentStates)
	{
		const FAgentState& AgentState = AgentStatePairs.Value;

		// Return false if anything
		if (!(AgentState.bTerminated || AgentState.bTruncated))
		{
			return false;
		}
	}
	return true;
}

void UAbstractGymConnector::PrepareEnvironments(const TArray<TScriptInterface<IBaseScholaEnvironment>>& InEnvironments)
{
	this->Environments.Empty();
	for (const TScriptInterface<IBaseScholaEnvironment>& TempEnv : InEnvironments)
	{
		if (TempEnv.GetObject()->GetClass()->ImplementsInterface(UCppOnlyMultiAgentEnvironment::StaticClass()))
		{
			this->Environments.Add(new TScholaEnvironment<ICppOnlyMultiAgentEnvironment>(TempEnv.GetObject()));
			UE_LOG(LogScholaTraining, Verbose, TEXT("Collected StateTree Environment %s"), *TempEnv.GetObject()->GetName());
		}
		else if (TempEnv.GetObject()->GetClass()->ImplementsInterface(UMultiAgentScholaEnvironment::StaticClass()))
		{
			this->Environments.Add(new TScholaEnvironment<IMultiAgentScholaEnvironment>(TempEnv.GetObject()));
			UE_LOGFMT(LogScholaTraining, Verbose, "UAbstractGymConnector::PrepareEnvironments(): Collected MultiAgent Environment {0}", TempEnv.GetObject()->GetName());
		}
		else if (TempEnv.GetObject()->GetClass()->ImplementsInterface(USingleAgentScholaEnvironment::StaticClass()))
		{
			this->Environments.Add(new TScholaEnvironment<ISingleAgentScholaEnvironment>(TempEnv.GetObject()));
			UE_LOGFMT(LogScholaTraining, Verbose, "UAbstractGymConnector::PrepareEnvironments(): Collected SingleAgent Environment {0}", TempEnv.GetObject()->GetName());
		}
		else
		{
			UE_LOGFMT(LogScholaTraining, Warning, "UAbstractGymConnector::PrepareEnvironments(): Collected Environment {0} does not implement a known Schola Environment Interface", TempEnv.GetObject()->GetName());
		}
	}
}

void UAbstractGymConnector::CollectEnvironments(TArray<TScriptInterface<IBaseScholaEnvironment>>& OutCollectedEnvironments)
{
	TArray<AActor*> TempEnvArray;
	UGameplayStatics::GetAllActorsWithInterface(GetWorld(), UBaseScholaEnvironment::StaticClass(), TempEnvArray);

	for (AActor* TempEnv : TempEnvArray)
	{
		OutCollectedEnvironments.Add(TScriptInterface<IBaseScholaEnvironment>(TempEnv));
		UE_LOGFMT(LogScholaTraining, Log, "UAbstractGymConnector::CollectEnvironments(): Collected Environments {0}", TempEnv->GetName());
	}
}

void UAbstractGymConnector::SetStatus(EConnectorStatus NewStatus)
{
	if (NewStatus == EConnectorStatus::Running)
	{
		this->OnConnectorStarted.Broadcast();
	}
	else if (NewStatus == EConnectorStatus::Closed)
	{
		this->OnConnectorClosed.Broadcast();
	}
	else if (NewStatus == EConnectorStatus::Error)
	{
		this->OnConnectorError.Broadcast();
	}
	this->Status = NewStatus;
}

void UAbstractGymConnector::UpdateConnectorStatus(const FTrainingStateUpdate& Decision)
{
	if (Decision.IsError())
	{
		this->SetStatus(EConnectorStatus::Error);
		UE_LOGFMT(LogScholaTraining, Warning, "UAbstractGymConnector::UpdateConnectorStatus(): Gym Connector Error");
	}
	else if (Decision.IsClosed())
	{
		this->SetStatus(EConnectorStatus::Closed);
		UE_LOGFMT(LogScholaTraining, Log, "UAbstractGymConnector::UpdateConnectorStatus(): Gym Connector Closed");
	}
}

void UAbstractGymConnector::HandleStep(const FTrainingStep& InStep, FTrainingState& OutTrainingState, FInitialState& OutInitialState)
{
	TRACE_CPUPROFILER_EVENT_SCOPE_STR("Schola: Gym Connector Step");
	// Note this function assumes that OutTrainingState is populated
	// If this isn't the case then there will be errors here.
	OutInitialState.EnvironmentStates.Empty();

	switch (this->GetAutoResetType())
	{
		case EAutoResetType::Disabled: // No Auto Reset
		{
			ParallelFor(TEXT("Parallel Environment Step"), InStep.EnvSteps.Num(), MinParallelBatchSize, [this, &InStep, &OutTrainingState](int EnvIdx) {
				TRACE_CPUPROFILER_EVENT_SCOPE_STR("Schola: Environment Step");
				const FEnvStep&	   EnvStep = InStep.EnvSteps[EnvIdx];
				FEnvironmentState& StateRef = OutTrainingState.EnvironmentStates[EnvIdx];
				// Don't step completed environments
				if (StateRef.IsEnvironmentActive())
				{
					Environments[EnvIdx]->Step(EnvStep.Actions, StateRef.AgentStates);
					
					if (AllAgentsCompleted(StateRef))
					{
						StateRef.MarkCompleted();
					}
				} }, this->bRunEnvironmentsInParallel ? EParallelForFlags::None : EParallelForFlags::ForceSingleThread);

			this->SubmitState(OutTrainingState);
			break;
		}
		case EAutoResetType::SameStep: // Auto Reset Same Step
		{
			ParallelFor(TEXT("Parallel Environment Step"), InStep.EnvSteps.Num(), MinParallelBatchSize, [this, &InStep, &OutTrainingState, &OutInitialState](int EnvIdx) {
				TRACE_CPUPROFILER_EVENT_SCOPE_STR("Schola: Environment Step");
				const FEnvStep&	   EnvStep = InStep.EnvSteps[EnvIdx];
				FEnvironmentState& StateRef = OutTrainingState.EnvironmentStates[EnvIdx];
				
				Environments[EnvIdx]->Step(EnvStep.Actions, StateRef.AgentStates);

				if (AllAgentsCompleted(StateRef))
				{
					FInitialEnvironmentState* EnvState;
					{
						FScopeLock Lock(&InitialStateCriticalSection);
						EnvState = &OutInitialState.EnvironmentStates.Emplace(EnvIdx);
					}

					Environments[EnvIdx]->Reset(EnvState->AgentStates);
				} }, this->bRunEnvironmentsInParallel ? EParallelForFlags::None : EParallelForFlags::ForceSingleThread);

			this->SubmitStateWithInitialState(OutTrainingState, OutInitialState);

			break;
		}
		case EAutoResetType::NextStep: // Auto Reset Next Step
		{
			ParallelFor(TEXT("Parallel Environment Step"), InStep.EnvSteps.Num(), MinParallelBatchSize, [this, &InStep, &OutTrainingState](int EnvIdx) {
				TRACE_CPUPROFILER_EVENT_SCOPE_STR("Schola: Environment Step");
				FEnvironmentState& EnvState = OutTrainingState.EnvironmentStates[EnvIdx];
				// Environment was completed at the end of last step so we do the Next Step reset.
				if (EnvState.IsEnvironmentCompleted())
				{
					TMap<FString, FInitialAgentState> InitialAgentStates;

					// We reset the environment, and mark it as active again.
					Environments[EnvIdx]->Reset(InitialAgentStates);
					EnvState.MarkActive();

					for (TPair<FString,FInitialAgentState>& InitialAgentStatePair : InitialAgentStates)
					{
						FAgentState& AgentState = EnvState.AgentStates.Add(InitialAgentStatePair.Key);
						AgentState.Observations = InitialAgentStatePair.Value.Observations;
						AgentState.Reward = 0.0f;
						AgentState.Info = InitialAgentStatePair.Value.Info;
						//Truncated and Terminated default to False so we don't need to set here
					}
				}
				else
				{
					// Environment is not completed so we just step it normally
					const FEnvStep&	   EnvStep = InStep.EnvSteps[EnvIdx];
					FEnvironmentState& StateRef = OutTrainingState.EnvironmentStates[EnvIdx];
					Environments[EnvIdx]->Step(EnvStep.Actions, StateRef.AgentStates);
					// Mark the Environment as completed if all agents finished on this step.
					// By tracking it after the step we handle the initial step properly.
					if (AllAgentsCompleted(StateRef))
					{
						StateRef.MarkCompleted();
					}

				} }, this->bRunEnvironmentsInParallel ? EParallelForFlags::None : EParallelForFlags::ForceSingleThread);
			this->SubmitState(OutTrainingState);
		}
	}
}

void UAbstractGymConnector::HandleReset(const FTrainingReset& InReset, FTrainingState& OutTrainingState, FInitialState& OutInitialState)
{
	TRACE_CPUPROFILER_EVENT_SCOPE_STR("Schola: Gym Connector Reset");
	OutInitialState.EnvironmentStates.Empty();
	// Apply any seeds or options to the environments first
	for (const TPair<int, FEnvReset>& EnvResetPair : InReset.Environments)
	{
		const FEnvReset& EnvReset = EnvResetPair.Value;
		if (EnvReset.bHasSeed)
		{
			Environments[EnvResetPair.Key]->SeedEnvironment(EnvReset.Seed);
			UE_LOGFMT(LogScholaTraining, Verbose, "UAbstractGymConnector::HandleReset(): Environment {0} seeded", EnvResetPair.Key);
		}

		if (EnvReset.Options.Num() > 0)
		{
			Environments[EnvResetPair.Key]->SetEnvironmentOptions(EnvReset.Options);
			UE_LOGFMT(LogScholaTraining, Verbose, "UAbstractGymConnector::HandleReset(): Environment {0} has {1} options supplied", EnvResetPair.Key, EnvResetPair.Value.Options.Num());
		}
	}

	// Reset all the environments.
	for (int EnvId = 0; EnvId < this->Environments.Num(); EnvId++)
	{
		FInitialEnvironmentState& InitialEnvState = OutInitialState.EnvironmentStates.Emplace(EnvId);
		Environments[EnvId]->Reset(InitialEnvState.AgentStates);
		// Reset and configure the environments
		OutTrainingState.EnvironmentStates[EnvId].AgentStates.Empty();
		OutTrainingState.EnvironmentStates[EnvId].MarkActive();
	}
	// Send the initial states back to wherever they need to go.
	this->SubmitInitialState(OutInitialState);
}

void UAbstractGymConnector::GetCompletedEnvironmentIds(const FTrainingState& InState, TArray<int>& OutCompletedEnvironmentIds)
{
	for (int i = 0; i < Environments.Num(); i++)
	{
		if (AllAgentsCompleted(InState.EnvironmentStates[i]))
		{
			OutCompletedEnvironmentIds.Add(i);
		}
	}
}

void UAbstractGymConnector::Step(FTrainingState& OutTrainingState, FInitialState& OutInitialState)
{
	TRACE_CPUPROFILER_EVENT_SCOPE_STR("Schola: Gym Connector Tick");
	if (this->IsNotStarted())
	{
		this->bFirstStep = true;
		bool bStarted = this->CheckForStart();
	}

	// Action Phase: We take any actions or Reset the Environment
	{
		if (this->IsRunning())
		{
			FTrainingStateUpdate* StateUpdate = this->ResolveEnvironmentStateUpdate();
			// Maybe there was nothing to resolve
			if (!StateUpdate)
			{
				// TODO figure out what to do here. Maybe close the connector?
				UE_LOGFMT(LogScholaTraining, Warning, "UAbstractGymConnector::Step(): No State Update resolved. Skipping this step.");
			}
			else if (StateUpdate->IsStep())
			{
				// Step
				UE_LOGFMT(LogScholaTraining, VeryVerbose, "UAbstractGymConnector::Step(): Step Received. Stepping Environments.");
				this->HandleStep(StateUpdate->GetStep(), OutTrainingState, OutInitialState);
			}
			else if (StateUpdate->IsReset())
			{
				// Reset
				UE_LOGFMT(LogScholaTraining, Verbose, "UAbstractGymConnector::Step(): Reset Received. Resetting Environments.");
				this->HandleReset(StateUpdate->GetReset(), OutTrainingState, OutInitialState);
			}
			else if (StateUpdate->IsClosed())
			{
				UE_LOGFMT(LogScholaTraining, Log, "UAbstractGymConnector::Step(): Close Received. Closing Connection.");
				this->SetStatus(EConnectorStatus::Closed);
			}
			else if (StateUpdate->IsError())
			{
				UE_LOGFMT(LogScholaTraining, Warning, "UAbstractGymConnector::Step(): Error Received. Closing Connections.");
				this->SetStatus(EConnectorStatus::Error);
			}
		}
	}

	bFirstStep = false;
}

void UAbstractGymConnector::Step()
{
	this->Step(this->TrainingState, this->InitialState);
}
