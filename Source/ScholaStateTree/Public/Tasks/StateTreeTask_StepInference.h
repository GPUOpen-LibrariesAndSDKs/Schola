// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Blueprint/StateTreeTaskBlueprintBase.h"
#include "StateTreeExecutionContext.h"
#include "Policies/PolicyInterface.h"
#include "Agent/AgentInterface.h"
#include "StateTreeTask_StepInference.generated.h"

class AStateTreeTrainingEnvironment;

/**
 * @brief Unified State Tree Task for RL inference and training.
 *
 * Auto-detects mode based on whether AStateTreeTrainingEnvironment exists:
 * - **Inference**: Runs NNE policy each tick (Observe -> Think -> Act)
 * - **Training**: Registers with environment, which drives Observe/Act
 *
 * ## Blueprint Interface (override in subclass):
 * - **Define**: Define observation and action spaces
 * - **Observe**: Collect observations
 * - **Act**: Apply an action
 * - **ComputeReward**: Return per-agent reward (training only)
 *
 * @see AStateTreeTrainingEnvironment
 */
UCLASS(Blueprintable, BlueprintType, meta = (DisplayName = "Step Inference Task"))
class SCHOLASTATETREE_API UStateTreeTask_StepInference : public UStateTreeTaskBlueprintBase, public IAgent
{
	GENERATED_BODY()

public:
	UStateTreeTask_StepInference(const FObjectInitializer& ObjectInitializer = FObjectInitializer::Get());

	// ========== Configuration ==========

	/** Get agent ID (resolved from state name). */
	UFUNCTION(BlueprintCallable, BlueprintPure, Category = "Schola|StateTree")
	FString GetAgentId() const;

	/** NNE policy for inference. Only used in inference mode. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Instanced, Category = "Reinforcement Learning", meta = (AllowedClasses = "/Script/ScholaNNE.NNEPolicy"))
	TObjectPtr<UObject> Policy;

	/** Complete task after single inference step (inference mode only). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Reinforcement Learning")
	bool bCompleteAfterSingleStep = false;

	// ========== IAgent Interface (override in Blueprint) ==========

	/** Define observation and action spaces. */
	virtual void Define_Implementation(FInteractionDefinition& OutDefinition) override;

	/** Collect observations. Use GetContextActor() to access actor data. */
	virtual void Observe_Implementation(FInstancedStruct& OutObservation) override;

	/** Apply an action. */
	virtual void Act_Implementation(const FInstancedStruct& InAction) override;

	// ========== Training ==========

	/** Compute per-agent reward (training mode only). */
	UFUNCTION(BlueprintNativeEvent, Category = "Schola|StateTree")
	void ComputeReward(float& OutReward);

	/**
	 * Reset this agent for a new episode. Called by the environment on ALL agents
	 * at episode start, even if this task's state isn't entered yet.
	 * Override in Blueprint to reset any per-episode state and return initial observation.
	 * @param ContextActor The actor context for observations
	 * @param OutObservation The initial observation for this episode
	 */
	UFUNCTION(BlueprintNativeEvent, Category = "Schola|StateTree")
	void ResetForEpisode(AActor* ContextActor, FInstancedStruct& OutObservation);

	// ========== Helpers ==========

	/** Get the context actor for use in Observe/Act. */
	UFUNCTION(BlueprintCallable, Category = "Schola|StateTree")
	AActor* GetContextActor() const { return CachedActor.Get(); }

	/** Check if running in training mode. */
	UFUNCTION(BlueprintCallable, BlueprintPure, Category = "Schola|StateTree")
	bool IsTrainingMode() const { return bTrainingMode; }

protected:
	virtual EStateTreeRunStatus EnterState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) override;
	virtual void				ExitState(FStateTreeExecutionContext& Context, const FStateTreeTransitionResult& Transition) override;
	virtual EStateTreeRunStatus Tick(FStateTreeExecutionContext& Context, const float DeltaTime) override;

	virtual EAgentStatus GetStatus_Implementation() override { return EAgentStatus::Running; }
	virtual void		 SetStatus_Implementation(EAgentStatus NewStatus) override {}

private:
	UPROPERTY(Transient)
	TWeakObjectPtr<AActor> CachedActor;

	UPROPERTY(Transient)
	TWeakObjectPtr<AStateTreeTrainingEnvironment> TrainingEnvironment;

	/** Cached agent ID resolved from node name or override */
	FString CachedAgentId;

	bool bInitialized = false;
	bool bTrainingMode = false;

	TInstancedStruct<FPoint> ObservationBuffer;
	TInstancedStruct<FPoint> ActionBuffer;

	/** Policy state read by the next inference step. Null for stateless policies. */
	UPROPERTY(Transient)
	TScriptInterface<IPolicyState> CurrentState;

	/** Policy state written by the next inference step, swapped with CurrentState on success. */
	UPROPERTY(Transient)
	TScriptInterface<IPolicyState> NextState;

	bool						   InitializePolicy();
	EStateTreeRunStatus			   PerformInferenceStep(float DeltaTime);
	AStateTreeTrainingEnvironment* FindTrainingEnvironment(UWorld* World) const;

	/** Resolve agent ID from state name */
	void ResolveAgentId(FStateTreeExecutionContext& Context);
};
