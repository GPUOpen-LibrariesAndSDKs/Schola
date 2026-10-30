// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Environment/StateTreeTrainingEnvironment.h"
#include "Tasks/StateTreeTask_StepInference.h"
#include "Evaluators/StateTreeEvaluator_RLDecision.h"
#include "Components/ScholaStateTreeComponent.h"
#include "StateTree.h"
#include "StateTreeTypes.h"
#include "Blueprint/StateTreeTaskBlueprintBase.h"
#include "Blueprint/StateTreeEvaluatorBlueprintBase.h"
#include "Components/StateTreeComponent.h"
#include "LogScholaStateTree.h"
#include "Points/Point.h"
#include "Points/DiscretePoint.h"
#include "Agent/AgentInterface.h"
#include "Common/InstancedStructUtils.h"
#include "Engine/World.h" // Include for UWorld

AStateTreeTrainingEnvironment::AStateTreeTrainingEnvironment()
{
	PrimaryActorTick.bCanEverTick = false;
}

UScholaStateTreeComponent* AStateTreeTrainingEnvironment::FindStateTreeComponent() const
{
	return FindComponentByClass<UScholaStateTreeComponent>();
}

// --- Task Registration ---

void AStateTreeTrainingEnvironment::RegisterTask(UStateTreeTask_StepInference* Task)
{
	if (!Task)
	{
		return;
	}

	FString AgentId = Task->GetAgentId().ToLower();

	if (TObjectPtr<UStateTreeTask_StepInference>* ExistingTask = RegisteredTasks.Find(AgentId))
	{
		if (*ExistingTask == Task)
		{
			return;
		}
		UE_LOGFMT(LogScholaStateTree, Error, "AStateTreeTrainingEnvironment::RegisterTask(): Duplicate agent ID '{AgentId}'! Each state must have a unique name.", AgentId);
		return;
	}

	RegisteredTasks.Add(AgentId, Task);
	PendingDoneAgents.Remove(AgentId);
	ActiveAgents.Add(AgentId);
	AgentsActiveThisEpisode.Add(AgentId);

	if (!CachedDefinitions.Contains(AgentId))
	{
		FInteractionDefinition Defn;
		IAgent::Execute_Define(Task, Defn);
		CachedDefinitions.Add(AgentId, Defn);
	}

	UE_LOGFMT(LogScholaStateTree, Verbose, "AStateTreeTrainingEnvironment::RegisterTask(): '{AgentId}' registered", AgentId);
}

void AStateTreeTrainingEnvironment::UnregisterTask(UStateTreeTask_StepInference* Task)
{
	if (!Task)
	{
		return;
	}

	FString AgentId = Task->GetAgentId().ToLower();

	if (RegisteredTasks.Contains(AgentId))
	{
		RegisteredTasks.Remove(AgentId);
		PendingDoneAgents.Add(AgentId);
		UE_LOGFMT(LogScholaStateTree, Verbose, "AStateTreeTrainingEnvironment::UnregisterTask(): '{AgentId}' unregistered and marked pending done", AgentId);
	}
}

// --- Evaluator Registration ---

void AStateTreeTrainingEnvironment::RegisterEvaluator(UStateTreeEvaluator_RLDecision* Evaluator)
{
	if (!Evaluator)
	{
		return;
	}

	FString AgentId = Evaluator->GetAgentId().ToLower();

	if (RegisteredEvaluators.Contains(AgentId))
	{
		UE_LOGFMT(LogScholaStateTree, Error, "AStateTreeTrainingEnvironment::RegisterEvaluator(): Duplicate agent ID '{AgentId}'! Each evaluator must have a unique name.", AgentId);
		return;
	}

	RegisteredEvaluators.Add(AgentId, Evaluator);
	PendingDoneAgents.Remove(AgentId);
	ActiveAgents.Add(AgentId);
	AgentsActiveThisEpisode.Add(AgentId);

	if (!CachedDefinitions.Contains(AgentId))
	{
		FInteractionDefinition Defn;
		IAgent::Execute_Define(Evaluator, Defn);
		CachedDefinitions.Add(AgentId, Defn);
	}

	UE_LOGFMT(LogScholaStateTree, Verbose, "AStateTreeTrainingEnvironment::RegisterEvaluator(): '{AgentId}' registered", AgentId);
}

void AStateTreeTrainingEnvironment::UnregisterEvaluator(UStateTreeEvaluator_RLDecision* Evaluator)
{
	if (!Evaluator)
	{
		return;
	}

	// Evaluators stay active until episode ends - just clear stale action
	FString AgentId = Evaluator->GetAgentId().ToLower();
	CurrentActions.Remove(AgentId);
}

// --- Agent Activation API ---

void AStateTreeTrainingEnvironment::ActivateAgent(const FString& AgentId)
{
	FString NormalizedId = AgentId.ToLower();
	PendingDoneAgents.Remove(NormalizedId);
	ActiveAgents.Add(NormalizedId);
}

void AStateTreeTrainingEnvironment::DeactivateAgent(const FString& AgentId)
{
	FString NormalizedId = AgentId.ToLower();
	if (ActiveAgents.Remove(NormalizedId) > 0)
	{
		CurrentActions.Remove(NormalizedId);
		PendingDoneAgents.Add(NormalizedId);
	}
}

bool AStateTreeTrainingEnvironment::IsAgentActive(const FString& AgentId) const
{
	return ActiveAgents.Contains(AgentId.ToLower());
}

int32 AStateTreeTrainingEnvironment::GetBranchAction(const FString& EvaluatorAgentId) const
{
	const FInstancedStruct* Action = CurrentActions.Find(EvaluatorAgentId.ToLower());
	if (Action && Action->IsValid())
	{
		if (const FDiscretePoint* DiscretePoint = Action->GetPtr<FDiscretePoint>())
		{
			return DiscretePoint->Value;
		}
	}
	return -1;
}

// --- ICppOnlyMultiAgentEnvironment Implementation ---

void AStateTreeTrainingEnvironment::InitializeEnvironment(TMap<FString, FInteractionDefinition>& OutAgentDefinitions)
{
	// Clear all state
	ActiveAgents.Empty();
	CurrentActions.Empty();
	PendingDoneAgents.Empty();
	RegisteredTasks.Empty();
	RegisteredEvaluators.Empty();
	CachedDefinitions.Empty();
	CachedTaskHelpers.Empty();
	CachedEvaluatorHelpers.Empty();

	UScholaStateTreeComponent* STComp = FindStateTreeComponent();
	if (!STComp)
	{
		UE_LOG(LogScholaStateTree, Error, TEXT("AStateTreeTrainingEnvironment::InitializeEnvironment(): No ScholaStateTreeComponent found - use ScholaStateTreeComponent instead of StateTreeComponent"));
		return;
	}

	const UStateTree* StateTree = STComp->GetStateTree();
	if (!StateTree)
	{
		UE_LOG(LogScholaStateTree, Error, TEXT("AStateTreeTrainingEnvironment::InitializeEnvironment(): No StateTree set on StateTreeComponent"));
		return;
	}

	TConstArrayView<FCompactStateTreeState> States = StateTree->GetStates();
	const FInstancedStructContainer&		Nodes = StateTree->GetNodes();

	// Discover tasks from states
	for (int32 StateIndex = 0; StateIndex < States.Num(); ++StateIndex)
	{
		const FCompactStateTreeState& State = States[StateIndex];
		FString						  StateName = State.Name.ToString().ToLower();

		for (uint8 TaskOffset = 0; TaskOffset < State.TasksNum; ++TaskOffset)
		{
			int32 NodeIndex = State.TasksBegin + TaskOffset;
			if (NodeIndex >= Nodes.Num())
			{
				continue;
			}

			FConstStructView NodeView = Nodes[NodeIndex];

			if (const FStateTreeBlueprintTaskWrapper* TaskWrapper = NodeView.GetPtr<const FStateTreeBlueprintTaskWrapper>())
			{
				TSubclassOf<UStateTreeTaskBlueprintBase> TaskClass = TaskWrapper->TaskClass;
				if (TaskClass && TaskClass->IsChildOf(UStateTreeTask_StepInference::StaticClass()))
				{
					FString AgentId = StateName;

					if (AgentId.IsEmpty())
					{
						UE_LOGFMT(LogScholaStateTree, Error, "AStateTreeTrainingEnvironment::InitializeEnvironment(): State at index {StateIndex} has no name", StateIndex);
						continue;
					}

					if (CachedDefinitions.Contains(AgentId))
					{
						UE_LOGFMT(LogScholaStateTree, Error, "AStateTreeTrainingEnvironment::InitializeEnvironment(): Duplicate state name '{AgentId}'", AgentId);
						continue;
					}

					// Create a per-environment helper instance (not CDO) to avoid mutating shared state
					UStateTreeTask_StepInference* TaskHelper = NewObject<UStateTreeTask_StepInference>(this, TaskClass);
					if (TaskHelper)
					{
						FInteractionDefinition Defn;
						IAgent::Execute_Define(TaskHelper, Defn);
						CachedDefinitions.Add(AgentId, Defn);
						CachedTaskHelpers.Add(AgentId, TaskHelper);
						UE_LOGFMT(LogScholaStateTree, Verbose, "AStateTreeTrainingEnvironment::InitializeEnvironment(): Found task '{AgentId}'", AgentId);
					}
				}
			}
		}
	}

	// Discover evaluators
	for (int32 NodeIndex = 0; NodeIndex < Nodes.Num(); ++NodeIndex)
	{
		FConstStructView NodeView = Nodes[NodeIndex];

		if (const FStateTreeBlueprintEvaluatorWrapper* EvalWrapper = NodeView.GetPtr<const FStateTreeBlueprintEvaluatorWrapper>())
		{
			TSubclassOf<UStateTreeEvaluatorBlueprintBase> EvalClass = EvalWrapper->EvaluatorClass;
			if (EvalClass && EvalClass->IsChildOf(UStateTreeEvaluator_RLDecision::StaticClass()))
			{
				FString AgentId = EvalWrapper->Name.ToString().ToLower();

				if (AgentId.IsEmpty())
				{
					UE_LOGFMT(LogScholaStateTree, Error, "AStateTreeTrainingEnvironment::InitializeEnvironment(): Evaluator at index {NodeIndex} has no name", NodeIndex);
					continue;
				}

				if (CachedDefinitions.Contains(AgentId))
				{
					UE_LOGFMT(LogScholaStateTree, Error, "AStateTreeTrainingEnvironment::InitializeEnvironment(): Duplicate evaluator name '{AgentId}'", AgentId);
					continue;
				}

				// Create a per-environment helper instance (not CDO) to avoid mutating shared state
				UStateTreeEvaluator_RLDecision* EvalHelper = NewObject<UStateTreeEvaluator_RLDecision>(this, EvalClass);
				if (EvalHelper)
				{
					FInteractionDefinition Defn;
					IAgent::Execute_Define(EvalHelper, Defn);
					CachedDefinitions.Add(AgentId, Defn);
					CachedEvaluatorHelpers.Add(AgentId, EvalHelper);
					UE_LOGFMT(LogScholaStateTree, Verbose, "AStateTreeTrainingEnvironment::InitializeEnvironment(): Found evaluator '{AgentId}'", AgentId);
				}
			}
		}
	}

	OutAgentDefinitions = CachedDefinitions;
	UE_LOGFMT(LogScholaStateTree, Verbose, "AStateTreeTrainingEnvironment::InitializeEnvironment(): Found {NumAgents} agents", OutAgentDefinitions.Num());
}

void AStateTreeTrainingEnvironment::SeedEnvironment(int Seed)
{
	RandomStream.Initialize(Seed);
}

void AStateTreeTrainingEnvironment::SetEnvironmentOptions(const TMap<FString, FString>& Options)
{
	if (const FString* MaxSteps = Options.Find(TEXT("max_steps")))
	{
		MaxStepsPerEpisode = FCString::Atoi(**MaxSteps);
	}
}

void AStateTreeTrainingEnvironment::Reset(TMap<FString, FInitialAgentState>& OutAgentState)
{
	// StateTree operations must happen on the game thread
	checkf(IsInGameThread(), TEXT("AStateTreeTrainingEnvironment::Reset(): Must be called on game thread. StateTree environments cannot be stepped in parallel - set bRunEnvironmentsInParallel=false on the GymConnector."));
	if (!IsInGameThread())
	{
		UE_LOG(LogScholaStateTree, Error, TEXT("AStateTreeTrainingEnvironment::Reset(): Must be called on game thread. StateTree environments cannot be stepped in parallel."));
		return;
	}

	CurrentStep = 0;
	ActiveAgents.Empty();
	CurrentActions.Empty();
	PendingDoneAgents.Empty();
	EpisodeDoneAgents.Empty();
	AgentsActiveThisEpisode.Empty();
	RegisteredTasks.Empty();
	RegisteredEvaluators.Empty();

	OnEpisodeReset();

	// Start StateTree
	UScholaStateTreeComponent* STComp = FindStateTreeComponent();
	if (STComp)
	{
		STComp->SetComponentTickEnabled(false);

		if (STComp->IsRunning())
		{
			STComp->StopLogic(TEXT("Training Reset"));
		}
		STComp->StartLogic();

		if (!STComp->IsRunning())
		{
			UE_LOG(LogScholaStateTree, Error, TEXT("AStateTreeTrainingEnvironment::Reset(): StartLogic failed"));
			return;
		}

		float DeltaSeconds = 1.0f / 60.0f; // Fallback to ~60 Hz if world delta is unavailable
		if (UWorld* World = GetWorld())
		{
			const float WorldDelta = World->GetDeltaSeconds();
			if (WorldDelta > 0.0f)
			{
				DeltaSeconds = WorldDelta;
			}
		}

		// Ensure DeltaSeconds is initialized properly
		STComp->TickComponent(DeltaSeconds, ELevelTick::LEVELTICK_All, nullptr);
	}
	else
	{
		UE_LOG(LogScholaStateTree, Error, TEXT("AStateTreeTrainingEnvironment::Reset(): No StateTreeComponent found"));
		return;
	}

	AActor* ContextActor = this;

	// Reset all tasks via helper instances
	for (const auto& Pair : CachedTaskHelpers)
	{
		const FString&				  AgentId = Pair.Key;
		UStateTreeTask_StepInference* TaskHelper = Pair.Value;

		if (TaskHelper)
		{
			FInstancedStruct Obs;
			TaskHelper->ResetForEpisode(ContextActor, Obs);

			if (Obs.IsValid() && Obs.GetPtr<FPoint>())
			{
				FInitialAgentState& State = OutAgentState.Add(AgentId);
				State.Observations = ToTypedInstancedStruct<FPoint>(Obs);
				AgentsActiveThisEpisode.Add(AgentId);
				ActiveAgents.Add(AgentId);
			}
			else
			{
				UE_LOGFMT(LogScholaStateTree, Warning, "AStateTreeTrainingEnvironment::Reset(): Task '{AgentId}' returned invalid observation", AgentId);
			}
		}
	}

	// Reset all evaluators via helper instances
	for (const auto& Pair : CachedEvaluatorHelpers)
	{
		const FString&					AgentId = Pair.Key;
		UStateTreeEvaluator_RLDecision* EvalHelper = Pair.Value;

		if (EvalHelper)
		{
			FInstancedStruct Obs;
			EvalHelper->ResetForEpisode(ContextActor, Obs);

			if (Obs.IsValid() && Obs.GetPtr<FPoint>())
			{
				FInitialAgentState& State = OutAgentState.Add(AgentId);
				State.Observations = ToTypedInstancedStruct<FPoint>(Obs);
				AgentsActiveThisEpisode.Add(AgentId);
				ActiveAgents.Add(AgentId);
			}
			else
			{
				UE_LOGFMT(LogScholaStateTree, Warning, "AStateTreeTrainingEnvironment::Reset(): Evaluator '{AgentId}' returned invalid observation", AgentId);
			}
		}
	}

	UE_LOGFMT(LogScholaStateTree, Verbose, "AStateTreeTrainingEnvironment::Reset(): Returning {NumAgents} agent states", OutAgentState.Num());
}

void AStateTreeTrainingEnvironment::Step(const TMap<FString, FInstancedStruct>& InActions, TMap<FString, FAgentState>& OutAgentStates)
{
	// StateTree operations must happen on the game thread
	checkf(IsInGameThread(), TEXT("AStateTreeTrainingEnvironment::Step(): Must be called on game thread. StateTree environments cannot be stepped in parallel - set bRunEnvironmentsInParallel=false on the GymConnector."));
	if (!IsInGameThread())
	{
		UE_LOG(LogScholaStateTree, Error, TEXT("AStateTreeTrainingEnvironment::Step(): Must be called on game thread. StateTree environments cannot be stepped in parallel."));
		return;
	}

	CurrentStep++;
	;

	// Store actions for active agents
	for (const auto& ActionPair : InActions)
	{
		FString NormalizedId = ActionPair.Key.ToLower();

		if (EpisodeDoneAgents.Contains(NormalizedId))
		{
			continue;
		}

		if (ActiveAgents.Contains(NormalizedId))
		{
			CurrentActions.Add(NormalizedId, ActionPair.Value);
		}
	}

	// Apply actions to evaluators first (they determine branch selection)
	for (const auto& ActionPair : InActions)
	{
		FString NormalizedId = ActionPair.Key.ToLower();

		if (EpisodeDoneAgents.Contains(NormalizedId))
		{
			continue;
		}

		if (TObjectPtr<UStateTreeEvaluator_RLDecision>* EvaluatorPtr = RegisteredEvaluators.Find(NormalizedId))
		{
			if (*EvaluatorPtr)
			{
				IAgent::Execute_Act(*EvaluatorPtr, ActionPair.Value);
			}
		}
	}

	// Apply actions to tasks
	for (const auto& ActionPair : InActions)
	{
		FString NormalizedId = ActionPair.Key.ToLower();

		if (EpisodeDoneAgents.Contains(NormalizedId))
		{
			continue;
		}

		if (RegisteredEvaluators.Contains(NormalizedId))
		{
			continue;
		}

		if (TObjectPtr<UStateTreeTask_StepInference>* TaskPtr = RegisteredTasks.Find(NormalizedId))
		{
			if (*TaskPtr)
			{
				IAgent::Execute_Act(*TaskPtr, ActionPair.Value);
			}
		}
	}

	// Tick StateTree
	if (UScholaStateTreeComponent* STComp = FindStateTreeComponent())
	{
		const float DeltaSeconds = (GetWorld() != nullptr) ? GetWorld()->GetDeltaSeconds() : 0.0f;
		STComp->TickComponent(DeltaSeconds, ELevelTick::LEVELTICK_All, nullptr);
	}

	// Check termination
	bool bTerminated = false;
	bool bTruncated = false;
	IsEpisodeOver(bTerminated, bTruncated);
	bTruncated = bTruncated || (CurrentStep >= MaxStepsPerEpisode);

	if (bTerminated || bTruncated)
	{
		// Episode ending - terminate all agents
		AActor* ContextActor = this;

		// Terminate all tasks
		for (const auto& Pair : CachedTaskHelpers)
		{
			const FString&				  AgentId = Pair.Key;
			UStateTreeTask_StepInference* TaskHelper = Pair.Value;

			if (EpisodeDoneAgents.Contains(AgentId))
			{
				continue;
			}

			if (TaskHelper)
			{
				FInstancedStruct Obs;
				float			 Reward = 0.0f;

				if (TObjectPtr<UStateTreeTask_StepInference>* TaskPtr = RegisteredTasks.Find(AgentId))
				{
					if (*TaskPtr)
					{
						IAgent::Execute_Observe(*TaskPtr, Obs);
						(*TaskPtr)->ComputeReward(Reward);
					}
				}

				if (!Obs.IsValid() || !Obs.GetPtr<FPoint>())
				{
					TaskHelper->ResetForEpisode(ContextActor, Obs);
				}

				if (Obs.IsValid() && Obs.GetPtr<FPoint>())
				{
					FAgentState& State = OutAgentStates.Add(AgentId);
					State.Observations = ToTypedInstancedStruct<FPoint>(Obs);
					State.Reward = Reward;
					State.bTerminated = bTerminated;
					State.bTruncated = bTruncated;
					EpisodeDoneAgents.Add(AgentId);
				}
			}
		}

		// Terminate all evaluators
		for (const auto& Pair : CachedEvaluatorHelpers)
		{
			const FString&					AgentId = Pair.Key;
			UStateTreeEvaluator_RLDecision* EvalHelper = Pair.Value;

			if (EpisodeDoneAgents.Contains(AgentId))
			{
				continue;
			}

			if (EvalHelper)
			{
				FInstancedStruct Obs;
				float			 Reward = 0.0f;

				if (TObjectPtr<UStateTreeEvaluator_RLDecision>* EvalPtr = RegisteredEvaluators.Find(AgentId))
				{
					if (*EvalPtr)
					{
						IAgent::Execute_Observe(*EvalPtr, Obs);
						(*EvalPtr)->ComputeReward(Reward);
					}
				}

				if (!Obs.IsValid() || !Obs.GetPtr<FPoint>())
				{
					EvalHelper->ResetForEpisode(ContextActor, Obs);
				}

				if (Obs.IsValid() && Obs.GetPtr<FPoint>())
				{
					FAgentState& State = OutAgentStates.Add(AgentId);
					State.Observations = ToTypedInstancedStruct<FPoint>(Obs);
					State.Reward = Reward;
					State.bTerminated = bTerminated;
					State.bTruncated = bTruncated;
					EpisodeDoneAgents.Add(AgentId);
				}
			}
		}

		// Agents that exited this step still need a terminal state for RLlib.
		for (const FString& DoneAgentId : PendingDoneAgents)
		{
			if (!EpisodeDoneAgents.Contains(DoneAgentId))
			{
				FAgentState& State = OutAgentStates.FindOrAdd(DoneAgentId);
				State.bTerminated = true;
				State.bTruncated = false;
				EpisodeDoneAgents.Add(DoneAgentId);
			}
			ActiveAgents.Remove(DoneAgentId);
			CurrentActions.Remove(DoneAgentId);
		}
		PendingDoneAgents.Empty();

		UE_LOGFMT(LogScholaStateTree, Verbose, "AStateTreeTrainingEnvironment::Step(): Step {CurrentStep} - Episode ended, returning {NumAgents} agents", CurrentStep, OutAgentStates.Num());
		return;
	}

	// Normal step - collect from active agents only

	// Evaluators
	for (const auto& Pair : RegisteredEvaluators)
	{
		const FString&					AgentId = Pair.Key;
		UStateTreeEvaluator_RLDecision* Evaluator = Pair.Value;

		if (EpisodeDoneAgents.Contains(AgentId) || !ActiveAgents.Contains(AgentId))
		{
			continue;
		}

		if (Evaluator)
		{
			FInstancedStruct Obs;
			IAgent::Execute_Observe(Evaluator, Obs);

			if (Obs.IsValid() && Obs.GetPtr<FPoint>())
			{
				FAgentState& State = OutAgentStates.Add(AgentId);
				State.Observations = ToTypedInstancedStruct<FPoint>(Obs);

				float Reward = 0.0f;
				Evaluator->ComputeReward(Reward);
				State.Reward = Reward;
				State.bTerminated = false;
				State.bTruncated = false;
			}
		}
	}

	// Tasks
	for (const auto& Pair : RegisteredTasks)
	{
		const FString&				  AgentId = Pair.Key;
		UStateTreeTask_StepInference* Task = Pair.Value;

		if (EpisodeDoneAgents.Contains(AgentId) || !ActiveAgents.Contains(AgentId) || PendingDoneAgents.Contains(AgentId))
		{
			continue;
		}

		if (Task)
		{
			FInstancedStruct Obs;
			IAgent::Execute_Observe(Task, Obs);

			if (Obs.IsValid() && Obs.GetPtr<FPoint>())
			{
				FAgentState& State = OutAgentStates.Add(AgentId);
				State.Observations = ToTypedInstancedStruct<FPoint>(Obs);

				float Reward = 0.0f;
				Task->ComputeReward(Reward);
				State.Reward = Reward;
				State.bTerminated = false;
				State.bTruncated = false;
			}
		}
	}

	// Agents that deactivated or unregistered this step are terminated once.
	for (const FString& DoneAgentId : PendingDoneAgents)
	{
		if (!EpisodeDoneAgents.Contains(DoneAgentId))
		{
			FAgentState& State = OutAgentStates.FindOrAdd(DoneAgentId);
			State.bTerminated = true;
			State.bTruncated = false;
			EpisodeDoneAgents.Add(DoneAgentId);
		}
		ActiveAgents.Remove(DoneAgentId);
		CurrentActions.Remove(DoneAgentId);
	}
	PendingDoneAgents.Empty();

	UE_LOGFMT(LogScholaStateTree, Verbose, "AStateTreeTrainingEnvironment::Step(): Step {CurrentStep} - Returning {NumAgents} agents", CurrentStep, OutAgentStates.Num());
}

// --- Blueprint Event Default Implementations ---

void AStateTreeTrainingEnvironment::IsEpisodeOver_Implementation(bool& OutTerminated, bool& OutTruncated)
{
	OutTerminated = false;
	OutTruncated = false;
}

void AStateTreeTrainingEnvironment::OnEpisodeReset_Implementation()
{
}
