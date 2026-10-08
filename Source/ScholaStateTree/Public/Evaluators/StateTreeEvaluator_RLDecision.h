// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Blueprint/StateTreeEvaluatorBlueprintBase.h"
#include "StateTreeExecutionContext.h"
#include "Policies/PolicyInterface.h"
#include "Agent/AgentInterface.h"
#include "StateTreeEvaluator_RLDecision.generated.h"

class AStateTreeTrainingEnvironment;

/**
 * @brief Unified State Tree Evaluator for RL-driven branch selection.
 *
 * Auto-detects mode based on whether Policy is provided:
 * - **Inference**: Uses the provided NNE policy to select branches
 * - **Training**: Registers with environment, which provides branch selection
 *
 * ## Blueprint Interface (override in subclass):
 * - **Define**: Define observation and action spaces
 * - **Observe**: Collect observations
 * - **ComputeReward**: Return per-agent reward (training only)
 *
 * ## Output Properties (bindable to conditions):
 * - SelectedBranch: The branch index chosen (0, 1, 2, ...)
 * - Confidence: How confident the policy is (0.0 - 1.0)
 * - NumBranches: Total number of possible branches
 *
 * @see FStateTreeCondition_RLBranch
 */
UCLASS(Blueprintable, BlueprintType, meta = (DisplayName = "RL Decision Evaluator"))
class SCHOLASTATETREE_API UStateTreeEvaluator_RLDecision : public UStateTreeEvaluatorBlueprintBase, public IAgent
{
	GENERATED_BODY()

public:
	UStateTreeEvaluator_RLDecision(const FObjectInitializer& ObjectInitializer = FObjectInitializer::Get());

	// ========== Configuration ==========

	/** Get agent ID (resolved from evaluator node name). */
	UFUNCTION(BlueprintCallable, BlueprintPure, Category = "Schola|StateTree")
	FString GetAgentId() const;

	/** NNE policy for inference. Only used in inference mode. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Instanced, Category = "Reinforcement Learning", meta = (AllowedClasses = "/Script/ScholaNNE.NNEPolicy"))
	TObjectPtr<UObject> Policy;

	/** Re-evaluate every tick (vs once at start). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Reinforcement Learning")
	bool bReevaluateEveryTick = true;

	/** Minimum seconds between re-evaluations. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Reinforcement Learning", meta = (EditCondition = "bReevaluateEveryTick", ClampMin = "0.0"))
	float ReevaluationInterval = 0.0f;

	// ========== Output Properties ==========

	/** The selected branch index. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Output")
	int32 SelectedBranch = -1;

	/** Number of possible branches (derived from ActionSpaceDefn). */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Output")
	int32 NumBranches = 0;

	/** Confidence in selection (0.0 - 1.0). */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Output")
	float Confidence = 0.0f;

	/** Raw scores for each branch. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Output")
	TArray<float> BranchScores;

	/** Whether last inference/decision succeeded. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Output")
	bool bLastInferenceSucceeded = false;

	// ========== IAgent Interface (override in Blueprint) ==========

	virtual void Define_Implementation(FInteractionDefinition& OutDefinition) override;
	virtual void Observe_Implementation(FInstancedStruct& OutObservation) override;
	virtual void Act_Implementation(const FInstancedStruct& InAction) override;

	// ========== Training ==========

	/** Compute per-agent reward (training mode only). */
	UFUNCTION(BlueprintNativeEvent, Category = "Schola|StateTree")
	void ComputeReward(float& OutReward);

	/**
	 * Reset this agent for a new episode. Called by the environment on ALL agents
	 * at episode start. Override in Blueprint to reset any per-episode state.
	 * @param ContextActor The actor context for observations
	 * @param OutObservation The initial observation for this episode
	 */
	UFUNCTION(BlueprintNativeEvent, Category = "Schola|StateTree")
	void ResetForEpisode(AActor* ContextActor, FInstancedStruct& OutObservation);

	// ========== Helpers ==========

	/** Get the context actor. */
	UFUNCTION(BlueprintCallable, Category = "Schola|StateTree")
	AActor* GetContextActor() const { return CachedActor.Get(); }

	/** Check if running in training mode. */
	UFUNCTION(BlueprintCallable, BlueprintPure, Category = "Schola|StateTree")
	bool IsTrainingMode() const { return bTrainingMode; }

	/** Get number of branches. */
	UFUNCTION(BlueprintCallable, BlueprintPure, Category = "Schola|StateTree")
	int32 GetNumBranches() const { return NumBranches; }

	/** Set selected branch (for definition collection during init). */
	void SetSelectedBranch(int32 Branch)
	{
		SelectedBranch = Branch;
		Confidence = 1.0f;
	}

protected:
	virtual void TreeStart(FStateTreeExecutionContext& Context) override;
	virtual void TreeStop(FStateTreeExecutionContext& Context) override;
	virtual void Tick(FStateTreeExecutionContext& Context, const float DeltaTime) override;

	virtual EAgentStatus GetStatus_Implementation() override { return EAgentStatus::Running; }
	virtual void		 SetStatus_Implementation(EAgentStatus NewStatus) override {}

private:
	UPROPERTY(Transient)
	TWeakObjectPtr<AActor> CachedActor;

	UPROPERTY(Transient)
	TWeakObjectPtr<AStateTreeTrainingEnvironment> TrainingEnvironment;

	/** Cached agent ID resolved from node name or override */
	FString CachedAgentId;

	bool  bInitialized = false;
	bool  bTrainingMode = false;
	float TimeSinceLastEvaluation = 0.0f;

	TInstancedStruct<FPoint> ObservationBuffer;
	TInstancedStruct<FPoint> ActionBuffer;

	/** Policy state, advanced in place by each decision. Null for stateless policies. */
	UPROPERTY(Transient)
	TScriptInterface<IPolicyState> CurrentState;

	bool						   InitializePolicy();
	void						   PerformDecision();
	int32						   ExtractBranchIndex(const TInstancedStruct<FPoint>& InActionBuffer);
	AStateTreeTrainingEnvironment* FindTrainingEnvironment(UWorld* World) const;

	/** Resolve agent ID from evaluator node name */
	void ResolveAgentId(FStateTreeExecutionContext& Context);
};
