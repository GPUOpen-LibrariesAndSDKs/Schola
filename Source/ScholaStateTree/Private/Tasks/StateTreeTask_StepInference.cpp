// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Tasks/StateTreeTask_StepInference.h"
#include "StateTreeHelpers.h"
#include "StateTreeExecutionContext.h"
#include "StateTree.h"
#include "StateTreeTypes.h"
#include "Environment/StateTreeTrainingEnvironment.h"
#include "LogScholaStateTree.h"
#include "GameFramework/Actor.h"
#include "EngineUtils.h"
#include "Common/InstancedStructUtils.h"

// ============================================================================
// IMPLEMENTATION
// ============================================================================

UStateTreeTask_StepInference::UStateTreeTask_StepInference(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
	bShouldCallTick = true;
}

FString UStateTreeTask_StepInference::GetAgentId() const
{
	// Return cached agent ID (resolved in ResolveAgentId)
	return CachedAgentId;
}

void UStateTreeTask_StepInference::ResolveAgentId(FStateTreeExecutionContext& Context)
{
	// Get STATE name (not node name) - this is the editable name in the StateTree editor
	// IMPORTANT: Normalize to lowercase for consistent matching with InitializeEnvironment
	FStateTreeStateHandle StateHandle = Context.GetCurrentlyProcessedState();
	if (const FCompactStateTreeState* State = Context.GetStateFromHandle(StateHandle))
	{
		CachedAgentId = State->Name.ToString().ToLower();
	}

	if (CachedAgentId.IsEmpty())
	{
		UE_LOG(LogScholaStateTree, Error, TEXT("UStateTreeTask_StepInference::ResolveAgentId(): State has no name! Set a unique name for the state in the StateTree editor."));
		CachedAgentId = TEXT("unnamed_task");
	}
}

EStateTreeRunStatus UStateTreeTask_StepInference::EnterState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition)
{
	// Resolve agent ID from node name (must be done first while context is valid)
	ResolveAgentId(Context);
	FString AgentId = GetAgentId();
	UE_LOGFMT(LogScholaStateTree, Verbose, "UStateTreeTask_StepInference::EnterState(): AgentId='{AgentId}'", AgentId);

	// Reset state
	bInitialized = false;
	bTrainingMode = false;
	TrainingEnvironment = nullptr;
	CachedActor = nullptr;
	ObservationBuffer.Reset();
	ActionBuffer.Reset();
	CurrentState = nullptr;
	NextState = nullptr;

	// Cache the context actor
	AActor* ContextActor = StateTreeHelpers::GetActorFromContext(Context);
	if (!ContextActor)
	{
		UE_LOG(LogScholaStateTree, Error, TEXT("UStateTreeTask_StepInference::EnterState(): No actor found in State Tree context"));
		return EStateTreeRunStatus::Failed;
	}
	CachedActor = ContextActor;

	// Auto-detect mode: check for training environment
	if (UWorld* World = ContextActor->GetWorld())
	{
		AStateTreeTrainingEnvironment* Env = FindTrainingEnvironment(World);
		if (Env)
		{
			// Training mode: register with the environment
			TrainingEnvironment = Env;
			bTrainingMode = true;
			Env->RegisterTask(this);
			bInitialized = true;

			UE_LOGFMT(LogScholaStateTree, Verbose, "UStateTreeTask_StepInference::EnterState(): Training mode - registered agent '{AgentId}'", AgentId);
		}
	}

	// Inference mode: initialize policy
	if (!bTrainingMode)
	{
		if (!InitializePolicy())
		{
			UE_LOG(LogScholaStateTree, Error, TEXT("UStateTreeTask_StepInference::EnterState(): Failed to initialize NNE policy"));
			return EStateTreeRunStatus::Failed;
		}
		bInitialized = true;

		UE_LOG(LogScholaStateTree, Verbose, TEXT("UStateTreeTask_StepInference::EnterState(): Inference mode - policy initialized"));
	}

	Super::EnterState(Context, Transition);

	// Inference mode: optionally do a single step now
	if (!bTrainingMode && bCompleteAfterSingleStep)
	{
		return PerformInferenceStep(0.0f);
	}

	return EStateTreeRunStatus::Running;
}

void UStateTreeTask_StepInference::ExitState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition)
{
	FString AgentId = GetAgentId();
	UE_LOGFMT(LogScholaStateTree, Verbose, "UStateTreeTask_StepInference::ExitState(): AgentId='{AgentId}'", AgentId);

	Super::ExitState(Context, Transition);

	// Unregister from training environment
	if (bTrainingMode && TrainingEnvironment.IsValid())
	{
		TrainingEnvironment->UnregisterTask(this);
		// DO NOT clear state here - we need CachedActor for final observation collection!
		// State will be cleared when task is actually removed from RegisteredTasks
		UE_LOG(LogScholaStateTree, Verbose, TEXT("UStateTreeTask_StepInference::ExitState(): Training mode - state preserved for final observation"));
		return;
	}

	// Inference mode: clean up normally
	TrainingEnvironment = nullptr;
	CachedActor = nullptr;
	CurrentState = nullptr;
	NextState = nullptr;
	bInitialized = false;
	bTrainingMode = false;
}

EStateTreeRunStatus UStateTreeTask_StepInference::Tick(FStateTreeExecutionContext& Context, const float DeltaTime)
{
	EStateTreeRunStatus ParentStatus = Super::Tick(Context, DeltaTime);

	// Training mode: nothing to do on tick, environment drives everything
	if (bTrainingMode)
	{
		return EStateTreeRunStatus::Running;
	}

	// Inference mode: run inference step
	if (!bInitialized)
	{
		UE_LOG(LogScholaStateTree, Error, TEXT("UStateTreeTask_StepInference::Tick(): Tick called but task not initialized"));
		return EStateTreeRunStatus::Failed;
	}

	EStateTreeRunStatus InferenceStatus = PerformInferenceStep(DeltaTime);
	if (InferenceStatus == EStateTreeRunStatus::Failed)
	{
		return InferenceStatus;
	}
	return ParentStatus;
}

EStateTreeRunStatus UStateTreeTask_StepInference::PerformInferenceStep(float DeltaTime)
{
	IPolicy* PolicyInterface = Cast<IPolicy>(Policy);
	if (!PolicyInterface)
	{
		UE_LOG(LogScholaStateTree, Error, TEXT("UStateTreeTask_StepInference::PerformInferenceStep(): Policy is invalid"));
		return EStateTreeRunStatus::Failed;
	}

	// Collect observations via IAgent interface
	FInstancedStruct RawObs;
	IAgent::Execute_Observe(const_cast<UStateTreeTask_StepInference*>(this), RawObs);

	if (RawObs.IsValid() && RawObs.GetPtr<FPoint>())
	{
		ObservationBuffer = ToTypedInstancedStruct<FPoint>(RawObs);
	}
	else
	{
		UE_LOG(LogScholaStateTree, Error, TEXT("UStateTreeTask_StepInference::PerformInferenceStep(): Observe did not return valid observation"));
		return EStateTreeRunStatus::Failed;
	}

	// Run inference
	if (!PolicyInterface->Think(ObservationBuffer, CurrentState, ActionBuffer, NextState))
	{
		UE_LOG(LogScholaStateTree, Error, TEXT("UStateTreeTask_StepInference::PerformInferenceStep(): Policy inference failed"));
		return EStateTreeRunStatus::Failed;
	}
	Swap(CurrentState, NextState);

	// Apply actions via IAgent interface
	FInstancedStruct ActionStruct = ToUntypedInstancedStruct(ActionBuffer);
	IAgent::Execute_Act(const_cast<UStateTreeTask_StepInference*>(this), ActionStruct);

	if (bCompleteAfterSingleStep)
	{
		return EStateTreeRunStatus::Succeeded;
	}
	return EStateTreeRunStatus::Running;
}

bool UStateTreeTask_StepInference::InitializePolicy()
{
	IPolicy* PolicyInterface = Cast<IPolicy>(Policy);
	if (!PolicyInterface)
	{
		UE_LOG(LogScholaStateTree, Error, TEXT("UStateTreeTask_StepInference::InitializePolicy(): No Policy specified or Policy does not implement IPolicy"));
		return false;
	}

	FInteractionDefinition InteractionDef;
	IAgent::Execute_Define(const_cast<UStateTreeTask_StepInference*>(this), InteractionDef);

	if (!InteractionDef.ObsSpaceDefn.IsValid())
	{
		UE_LOG(LogScholaStateTree, Error, TEXT("UStateTreeTask_StepInference::InitializePolicy(): Define did not set ObsSpaceDefn"));
		return false;
	}
	if (!InteractionDef.ActionSpaceDefn.IsValid())
	{
		UE_LOG(LogScholaStateTree, Error, TEXT("UStateTreeTask_StepInference::InitializePolicy(): Define did not set ActionSpaceDefn"));
		return false;
	}

	if (!PolicyInterface->Init(InteractionDef))
	{
		UE_LOG(LogScholaStateTree, Error, TEXT("UStateTreeTask_StepInference::InitializePolicy(): Failed to initialize policy"));
		return false;
	}

	if (!PolicyInterface->CreateInitialState(this, CurrentState) || !PolicyInterface->CreateInitialState(this, NextState))
	{
		UE_LOG(LogScholaStateTree, Error, TEXT("UStateTreeTask_StepInference::InitializePolicy(): Failed to create initial policy state"));
		return false;
	}

	UE_LOGFMT(LogScholaStateTree, Verbose, "UStateTreeTask_StepInference::InitializePolicy(): Policy initialized ({PolicyClass})", Policy->GetClass()->GetName());
	return true;
}

AStateTreeTrainingEnvironment* UStateTreeTask_StepInference::FindTrainingEnvironment(UWorld* World) const
{
	if (!World)
	{
		return nullptr;
	}

	// Check if the owner IS the training environment
	if (AActor* Owner = CachedActor.Get())
	{
		if (AStateTreeTrainingEnvironment* Env = Cast<AStateTreeTrainingEnvironment>(Owner))
		{
			return Env;
		}
	}

	// Search the world
	for (TActorIterator<AStateTreeTrainingEnvironment> It(World); It; ++It)
	{
		return *It;
	}

	return nullptr;
}

// ========== Default Implementations ==========

void UStateTreeTask_StepInference::ComputeReward_Implementation(float& OutReward)
{
	OutReward = 0.0f;
}

void UStateTreeTask_StepInference::Define_Implementation(FInteractionDefinition& OutDefinition)
{
	UE_LOG(LogScholaStateTree, Verbose, TEXT("UStateTreeTask_StepInference::Define_Implementation(): Base called - override in Blueprint"));
}

void UStateTreeTask_StepInference::Observe_Implementation(FInstancedStruct& OutObservation)
{
	UE_LOG(LogScholaStateTree, Verbose, TEXT("UStateTreeTask_StepInference::Observe_Implementation(): Base called - override in Blueprint"));
}

void UStateTreeTask_StepInference::Act_Implementation(const FInstancedStruct& InAction)
{
	// Override in Blueprint to handle the action
}

void UStateTreeTask_StepInference::ResetForEpisode_Implementation(AActor* ContextActor, FInstancedStruct& OutObservation)
{
	// Default implementation: reset policy state and call Observe with the provided context
	// Blueprint can override to reset per-episode state
	CachedActor = ContextActor;
	if (CurrentState)
	{
		CurrentState->Reset();
	}
	IAgent::Execute_Observe(this, OutObservation);
}