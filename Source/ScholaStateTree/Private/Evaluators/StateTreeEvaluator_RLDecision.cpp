// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "Evaluators/StateTreeEvaluator_RLDecision.h"
#include "RLDecisionHelpers.h"
#include "StateTreeHelpers.h"
#include "StateTreeExecutionContext.h"
#include "StateTree.h"
#include "StateTreeNodeBase.h"
#include "Environment/StateTreeTrainingEnvironment.h"
#include "LogScholaStateTree.h"
#include "GameFramework/Actor.h"
#include "Points/DiscretePoint.h"
#include "Points/DictPoint.h"
#include "Points/BoxPoint.h"
#include "Spaces/DiscreteSpace.h"
#include "Common/InstancedStructUtils.h"
#include "EngineUtils.h"

// ============================================================================
// RLDecisionHelpers Implementation
// ============================================================================

int32 RLDecisionHelpers::ExtractBranchIndexFromAction(
	const TInstancedStruct<FPoint>& ActionBuffer,
	int32&							InOutNumBranches,
	TArray<float>&					OutBranchScores,
	float&							OutConfidence)
{
	if (!ActionBuffer.IsValid())
	{
		return -1;
	}

	// Case 1: Direct discrete point
	if (const FDiscretePoint* DiscretePoint = ActionBuffer.GetPtr<FDiscretePoint>())
	{
		int32 BranchIdx = DiscretePoint->Value;

		// Validate branch index against configured number of branches
		if (InOutNumBranches <= 0 || BranchIdx < 0 || BranchIdx >= InOutNumBranches)
		{
			return -1;
		}

		OutBranchScores.SetNum(InOutNumBranches);
		for (int32 i = 0; i < InOutNumBranches; ++i)
		{
			OutBranchScores[i] = (i == BranchIdx) ? 1.0f : 0.0f;
		}
		OutConfidence = 1.0f;
		return BranchIdx;
	}

	// Case 2: Dict point - first discrete entry
	if (const FDictPoint* DictPoint = ActionBuffer.GetPtr<FDictPoint>())
	{
		for (const auto& Entry : DictPoint->Points)
		{
			if (const FDiscretePoint* InnerDiscrete = Entry.Value.GetPtr<FDiscretePoint>())
			{
				int32 BranchIdx = InnerDiscrete->Value;

				// Validate branch index against configured number of branches
				if (InOutNumBranches <= 0 || BranchIdx < 0 || BranchIdx >= InOutNumBranches)
				{
					return -1;
				}

				OutBranchScores.SetNum(InOutNumBranches);
				for (int32 i = 0; i < InOutNumBranches; ++i)
				{
					OutBranchScores[i] = (i == BranchIdx) ? 1.0f : 0.0f;
				}
				OutConfidence = 1.0f;
				return BranchIdx;
			}
		}
	}

	// Case 3: Box point - interpret as logits/scores, take argmax
	if (const FBoxPoint* BoxPoint = ActionBuffer.GetPtr<FBoxPoint>())
	{
		if (BoxPoint->Values.Num() > 0)
		{
			int32 BestIdx = 0;
			float BestVal = BoxPoint->Values[0];

			OutBranchScores.SetNum(BoxPoint->Values.Num());
			for (int32 i = 0; i < BoxPoint->Values.Num(); ++i)
			{
				OutBranchScores[i] = BoxPoint->Values[i];
				if (BoxPoint->Values[i] > BestVal)
				{
					BestVal = BoxPoint->Values[i];
					BestIdx = i;
				}
			}
			OutConfidence = BestVal;
			InOutNumBranches = BoxPoint->Values.Num();
			return BestIdx;
		}
	}

	return -1;
}

int32 RLDecisionHelpers::GetNumBranchesFromActionSpace(const TInstancedStruct<FSpace>& ActionSpaceDefn)
{
	if (const FDiscreteSpace* DiscreteSpace = ActionSpaceDefn.GetPtr<FDiscreteSpace>())
	{
		return DiscreteSpace->High;
	}

	return 0;
}

// ============================================================================
// UStateTreeEvaluator_RLDecision Implementation
// ============================================================================

UStateTreeEvaluator_RLDecision::UStateTreeEvaluator_RLDecision(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
}

FString UStateTreeEvaluator_RLDecision::GetAgentId() const
{
	// Return cached agent ID (resolved in ResolveAgentId)
	return CachedAgentId;
}

void UStateTreeEvaluator_RLDecision::ResolveAgentId(FStateTreeExecutionContext& Context)
{
	// Get node name from the StateTree
	// IMPORTANT: Normalize to lowercase for consistent matching with InitializeEnvironment
#if ENGINE_MAJOR_VERSION > 5 || (ENGINE_MAJOR_VERSION == 5 && ENGINE_MINOR_VERSION >= 6)
	const UStateTree*		StateTree = Context.GetStateTree();
	const FStateTreeIndex16 NodeIndex = Context.GetCurrentlyProcessedNodeIndex();

	if (StateTree && NodeIndex.IsValid())
	{
		FConstStructView NodeView = StateTree->GetNode(NodeIndex.Get());
		if (const FStateTreeNodeBase* NodeBase = NodeView.GetPtr<const FStateTreeNodeBase>())
		{
			CachedAgentId = NodeBase->Name.ToString().ToLower();
		}
	}
#endif

	if (CachedAgentId.IsEmpty())
	{
		UE_LOG(LogScholaStateTree, Error, TEXT("UStateTreeEvaluator_RLDecision::ResolveAgentId(): Evaluator node has no name! Set a unique name in the StateTree editor."));
		CachedAgentId = TEXT("unnamed_evaluator");
	}
}

void UStateTreeEvaluator_RLDecision::TreeStart(FStateTreeExecutionContext& Context)
{
	// Resolve agent ID from node name (must be done first while context is valid)
	ResolveAgentId(Context);
	FString AgentId = GetAgentId();
	UE_LOGFMT(LogScholaStateTree, Verbose, "UStateTreeEvaluator_RLDecision::TreeStart(): AgentId='{AgentId}'", AgentId);

	// Reset state
	bInitialized = false;
	bTrainingMode = false;
	TrainingEnvironment = nullptr;
	CachedActor = nullptr;
	SelectedBranch = -1;
	NumBranches = 0;
	BranchScores.Empty();
	Confidence = 0.0f;
	bLastInferenceSucceeded = false;
	TimeSinceLastEvaluation = 0.0f;
	CurrentState = nullptr;
	NextState = nullptr;

	// Find the context actor
	CachedActor = StateTreeHelpers::GetActorFromContext(Context);
	if (!CachedActor.IsValid())
	{
		UE_LOG(LogScholaStateTree, Error, TEXT("UStateTreeEvaluator_RLDecision::TreeStart(): No actor found in State Tree context"));
		return;
	}

	// Auto-detect mode: Policy set = inference, otherwise training
	if (Policy)
	{
		// Inference mode: initialize policy
		if (!InitializePolicy())
		{
			UE_LOG(LogScholaStateTree, Error, TEXT("UStateTreeEvaluator_RLDecision::TreeStart(): Failed to initialize NNE policy"));
			return;
		}
		bInitialized = true;
		UE_LOG(LogScholaStateTree, Verbose, TEXT("UStateTreeEvaluator_RLDecision::TreeStart(): Inference mode - policy initialized"));
	}
	else
	{
		// Training mode: find environment
		if (UWorld* World = CachedActor->GetWorld())
		{
			AStateTreeTrainingEnvironment* Env = FindTrainingEnvironment(World);
			if (Env)
			{
				TrainingEnvironment = Env;
				bTrainingMode = true;
				Env->RegisterEvaluator(this);
				bInitialized = true;

				// Get NumBranches from the action space definition
				FInteractionDefinition Defn;
				IAgent::Execute_Define(this, Defn);
				NumBranches = RLDecisionHelpers::GetNumBranchesFromActionSpace(Defn.ActionSpaceDefn);

				if (NumBranches <= 0)
				{
					// Invalid or unsupported action space definition for training mode
					BranchScores.Reset();
					SelectedBranch = -1;
					Env->UnregisterEvaluator(this);
					Confidence = 0.0f;
					bInitialized = false;

					UE_LOGFMT(LogScholaStateTree, Error, "UStateTreeEvaluator_RLDecision::TreeStart(): Invalid action space for training mode on agent '{AgentId}' (NumBranches={NumBranches}). Expected a discrete action space with at least one branch.", AgentId, NumBranches);
					return;
				}
				BranchScores.Init(0.0f, NumBranches);

				// Default to branch 0 until Python provides an action
				// This allows StateTree to select an initial state during Reset
				SelectedBranch = 0;
				Confidence = 1.0f;
				BranchScores[SelectedBranch] = 1.0f;

				UE_LOGFMT(LogScholaStateTree, Verbose, "UStateTreeEvaluator_RLDecision::TreeStart(): Training mode - registered agent '{AgentId}' (branch={SelectedBranch}, NumBranches={NumBranches})", AgentId, SelectedBranch, NumBranches);
			}
			else
			{
				UE_LOG(LogScholaStateTree, Error, TEXT("UStateTreeEvaluator_RLDecision::TreeStart(): No Policy and no training environment found"));
				return;
			}
		}
	}

	Super::TreeStart(Context);

	// Inference mode: perform initial decision
	if (!bTrainingMode && bInitialized)
	{
		PerformDecision();
	}

	UE_LOGFMT(LogScholaStateTree, Verbose, "UStateTreeEvaluator_RLDecision::TreeStart(): Complete, branch={SelectedBranch}, confidence={Confidence}",
		SelectedBranch, Confidence);
}

void UStateTreeEvaluator_RLDecision::TreeStop(FStateTreeExecutionContext& Context)
{
	FString AgentId = GetAgentId();
	UE_LOGFMT(LogScholaStateTree, Verbose, "UStateTreeEvaluator_RLDecision::TreeStop(): AgentId='{AgentId}'", AgentId);

	Super::TreeStop(Context);

	// In training mode, do NOT clear state - environment controls lifecycle via Reset
	// Only unregister (which is also a no-op now, but call it for logging)
	if (bTrainingMode && TrainingEnvironment.IsValid())
	{
		TrainingEnvironment->UnregisterEvaluator(this);
		// DO NOT clear state - evaluator must remain functional until Reset
		UE_LOG(LogScholaStateTree, Verbose, TEXT("UStateTreeEvaluator_RLDecision::TreeStop(): Training mode - state preserved until Reset"));
		return;
	}

	// Inference mode: clean up normally
	TrainingEnvironment = nullptr;
	CachedActor = nullptr;
	bInitialized = false;
	bTrainingMode = false;
	SelectedBranch = -1;
	BranchScores.Empty();
	CurrentState = nullptr;
	NextState = nullptr;
}

void UStateTreeEvaluator_RLDecision::Tick(FStateTreeExecutionContext& Context, const float DeltaTime)
{
	Super::Tick(Context, DeltaTime);

	if (!bInitialized)
	{
		return;
	}

	// Training mode: environment drives decisions via Act()
	if (bTrainingMode)
	{
		return;
	}

	// Inference mode: re-evaluate if configured
	if (!bReevaluateEveryTick)
	{
		return;
	}

	// Throttle re-evaluation
	if (ReevaluationInterval > 0.0f)
	{
		TimeSinceLastEvaluation += DeltaTime;
		if (TimeSinceLastEvaluation < ReevaluationInterval)
		{
			return;
		}
		TimeSinceLastEvaluation = 0.0f;
	}

	PerformDecision();
}

void UStateTreeEvaluator_RLDecision::PerformDecision()
{
	IPolicy* PolicyInterface = Cast<IPolicy>(Policy);
	if (!PolicyInterface)
	{
		UE_LOG(LogScholaStateTree, Error, TEXT("UStateTreeEvaluator_RLDecision::PerformDecision(): Policy is invalid"));
		bLastInferenceSucceeded = false;
		return;
	}

	// Collect observations via IAgent interface
	FInstancedStruct RawObs;
	IAgent::Execute_Observe(const_cast<UStateTreeEvaluator_RLDecision*>(this), RawObs);
	if (RawObs.IsValid() && RawObs.GetPtr<FPoint>())
	{
		ObservationBuffer = ToTypedInstancedStruct<FPoint>(RawObs);
	}
	else
	{
		UE_LOG(LogScholaStateTree, Warning, TEXT("UStateTreeEvaluator_RLDecision::PerformDecision(): Observe returned invalid observation"));
		bLastInferenceSucceeded = false;
		return;
	}

	// Run inference
	if (!PolicyInterface->Think(ObservationBuffer, CurrentState, ActionBuffer, NextState))
	{
		UE_LOG(LogScholaStateTree, Error, TEXT("UStateTreeEvaluator_RLDecision::PerformDecision(): Policy inference failed"));
		bLastInferenceSucceeded = false;
		return;
	}
	Swap(CurrentState, NextState);

	bLastInferenceSucceeded = true;

	// Extract branch selection
	int32 NewBranch = ExtractBranchIndex(ActionBuffer);
	if (NewBranch >= 0)
	{
		SelectedBranch = NewBranch;
		UE_LOGFMT(LogScholaStateTree, Verbose, "UStateTreeEvaluator_RLDecision::PerformDecision(): Selected branch {NewBranch}", NewBranch);
	}
	else
	{
		UE_LOG(LogScholaStateTree, Error, TEXT("UStateTreeEvaluator_RLDecision::PerformDecision(): Failed to extract branch index from action"));
		bLastInferenceSucceeded = false;
	}
}

int32 UStateTreeEvaluator_RLDecision::ExtractBranchIndex(const TInstancedStruct<FPoint>& InActionBuffer)
{
	return RLDecisionHelpers::ExtractBranchIndexFromAction(
		InActionBuffer,
		NumBranches,
		BranchScores,
		Confidence);
}

bool UStateTreeEvaluator_RLDecision::InitializePolicy()
{
	IPolicy* PolicyInterface = Cast<IPolicy>(Policy);
	if (!PolicyInterface)
	{
		UE_LOG(LogScholaStateTree, Error, TEXT("UStateTreeEvaluator_RLDecision::InitializePolicy(): Policy not set or does not implement IPolicy"));
		return false;
	}

	// Get interaction definition from IAgent interface
	FInteractionDefinition InteractionDef;
	IAgent::Execute_Define(const_cast<UStateTreeEvaluator_RLDecision*>(this), InteractionDef);

	if (!InteractionDef.ObsSpaceDefn.IsValid())
	{
		UE_LOG(LogScholaStateTree, Error, TEXT("UStateTreeEvaluator_RLDecision::InitializePolicy(): Define did not set ObsSpaceDefn"));
		return false;
	}
	if (!InteractionDef.ActionSpaceDefn.IsValid())
	{
		UE_LOG(LogScholaStateTree, Error, TEXT("UStateTreeEvaluator_RLDecision::InitializePolicy(): Define did not set ActionSpaceDefn"));
		return false;
	}

	if (!PolicyInterface->Init(InteractionDef))
	{
		UE_LOG(LogScholaStateTree, Error, TEXT("UStateTreeEvaluator_RLDecision::InitializePolicy(): Failed to initialize policy"));
		return false;
	}

	if (!PolicyInterface->CreateInitialState(this, CurrentState) || !PolicyInterface->CreateInitialState(this, NextState))
	{
		UE_LOG(LogScholaStateTree, Error, TEXT("UStateTreeEvaluator_RLDecision::InitializePolicy(): Failed to create initial policy state"));
		return false;
	}

	NumBranches = RLDecisionHelpers::GetNumBranchesFromActionSpace(InteractionDef.ActionSpaceDefn);
	BranchScores.SetNum(NumBranches);

	UE_LOGFMT(LogScholaStateTree, Verbose, "UStateTreeEvaluator_RLDecision::InitializePolicy(): Policy initialized, {NumBranches} branches", NumBranches);
	return true;
}

AStateTreeTrainingEnvironment* UStateTreeEvaluator_RLDecision::FindTrainingEnvironment(UWorld* World) const
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

void UStateTreeEvaluator_RLDecision::ComputeReward_Implementation(float& OutReward)
{
	OutReward = 0.0f;
}

void UStateTreeEvaluator_RLDecision::Define_Implementation(FInteractionDefinition& OutDefinition)
{
	UE_LOG(LogScholaStateTree, Verbose, TEXT("UStateTreeEvaluator_RLDecision::Define_Implementation(): Base called - override in Blueprint"));
}

void UStateTreeEvaluator_RLDecision::Observe_Implementation(FInstancedStruct& OutObservation)
{
	UE_LOG(LogScholaStateTree, Verbose, TEXT("UStateTreeEvaluator_RLDecision::Observe_Implementation(): Base called - override in Blueprint"));
}

void UStateTreeEvaluator_RLDecision::Act_Implementation(const FInstancedStruct& InAction)
{
	// Training mode: extract branch from action provided by environment
	if (bTrainingMode)
	{
		int32 NewBranch = ExtractBranchIndex(ToTypedInstancedStruct<FPoint>(InAction));
		if (NewBranch >= 0)
		{
			SelectedBranch = NewBranch;
			bLastInferenceSucceeded = true;
			UE_LOGFMT(LogScholaStateTree, Verbose, "UStateTreeEvaluator_RLDecision::Act_Implementation(): Training action -> branch {NewBranch}", NewBranch);
		}
	}
}

void UStateTreeEvaluator_RLDecision::ResetForEpisode_Implementation(AActor* ContextActor, FInstancedStruct& OutObservation)
{
	// Default implementation: set context and call Observe
	// Blueprint can override to reset per-episode state (e.g., branch selection logic)
	CachedActor = ContextActor;
	SelectedBranch = -1; // Reset branch selection for new episode
	if (CurrentState)
	{
		CurrentState->Reset();
	}
	IAgent::Execute_Observe(this, OutObservation);
}
