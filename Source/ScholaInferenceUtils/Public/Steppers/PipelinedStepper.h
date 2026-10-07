// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Steppers/StepperInterface.h"
#include "Agent/AgentInterface.h"
#include "Common/LogSchola.h"
#include "LogScholaInferenceUtils.h"
#include "PipelinedStepper.generated.h"

#define PIPELINE_STAGES 2

/**
 * @brief Pipelined stepper implementation for asynchronous inference.
 * 
 * A sophisticated stepper that pipelines agent observations and policy inference
 * across multiple frames to hide inference latency. Uses a two-stage pipeline:
 * 
 * Frame N:   Collect observations -> Dispatch async inference
 * Frame N+1: Apply actions from N   -> Collect observations -> Dispatch async inference
 * 
 * This allows the policy to perform inference on a background thread while the
 * game continues running, significantly reducing effective latency for expensive
 * inference operations. The stepper manages synchronization between the game
 * thread and inference threads.
 */
UCLASS(Blueprintable)
class SCHOLAINFERENCEUTILS_API UPipelinedStepper : public UObject, public IStepper
{
    GENERATED_BODY()

public:
    /**
     * @brief Initialize the pipelined stepper with agents and policy.
     * 
     * Sets up the pipeline frames and prepares for asynchronous operation.
     * 
     * @param[in] InAgents The array of agents to manage
     * @param[in] InPolicy The initialized policy to use for inference (must support batched operations)
     * @return true if initialization succeeded (agents and policy are valid), false otherwise
     */
    bool Init(const TArray<TScriptInterface<IAgent>>& InAgents, const TScriptInterface<IPolicy>& InPolicy) override
    {
        Agents.Reset();
        CurrentStates.Reset();
        Policy = InPolicy;
        TickCounter = 0;
        bDispatchInFlight = false;
        PendingStateResets.Reset();

        for (int i = 0; i < PIPELINE_STAGES; ++i)
        {
            Frames[i].Observations.Reset();
            Frames[i].Actions.Reset();
            Frames[i].DispatchedAgents.Reset();
            Frames[i].bActionsReady = false;
            Frames[i].bThinkInFlight = false;
        }
        if (InAgents.Num() == 0 || !Policy)
        {
            return false;
        }
        for (const TScriptInterface<IAgent>& Agent : InAgents)
        {
            if (!AddAgent(Agent))
            {
                Agents.Reset();
                CurrentStates.Reset();
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Start stepping an agent with the stepper's policy.
     *
     * Safe to call while inference is in flight: the agent gets its first action from the next dispatched inference.
     */
    UFUNCTION(BlueprintCallable, Category = "Schola|Stepper")
    bool AddAgent(const TScriptInterface<IAgent>& InAgent) override;

    /**
     * @brief Stop stepping an agent and release its policy state.
     *
     * Safe to call while inference is in flight: the agent receives no further actions, and its
     * state is kept alive until the in-flight inference completes.
     */
    UFUNCTION(BlueprintCallable, Category = "Schola|Stepper")
    bool RemoveAgent(const TScriptInterface<IAgent>& InAgent) override;

    /**
     * @brief Reset every agent's policy state to the start of an episode.
     * 
     * If inference is in flight, the reset is applied once it completes.
     */
    UFUNCTION(BlueprintCallable, Category = "Schola|Stepper")
    void ResetStates();

    /**
     * @brief Reset one agent's policy state to the start of an episode.
     * 
     * If inference is in flight, the reset is applied once it completes.
     * 
     * @param[in] AgentIndex Index of the agent in Agents
     */
    UFUNCTION(BlueprintCallable, Category = "Schola|Stepper")
    void ResetAgentState(int32 AgentIndex);

    /**
     * @brief Get each agent's current policy state.
     *
     * The states are updated in place on a background thread while inference is in flight,
     * so only read their contents once the dispatched inference has completed.
     * 
     * @return One state per agent, as read by the next dispatched inference (null entries for stateless policies)
     */
    const TArray<TScriptInterface<IPolicyState>>& GetCurrentStates() const { return CurrentStates; }

    /**
     * @brief Execute one step of the pipelined agent-policy loop.
     * 
     * Must be called every tick on the Game Thread. Performs:
     * - Applies actions from the previous frame (if ready)
     * - Collects observations from all agents
     * - Dispatches asynchronous inference if the policy is not busy and the previous result has been handled
     * 
     * The inference runs on a background thread and results are applied
     * in a subsequent frame once ready.
     */
    UFUNCTION(BlueprintCallable, Category = "Schola|Stepper")
    void Step();

    /**
     * @brief Override for object destruction.
     * 
     * Sets the shutdown flag to prevent background threads from accessing
     * the object after it has been destroyed.
     */
    virtual void BeginDestroy() override
    {
        bShuttingDown = true;
        Super::BeginDestroy();
    }

private:
    
    /** Array of agents managed by this stepper */
    UPROPERTY() 
    TArray<TScriptInterface<IAgent>> Agents;
    
    /** Policy used for inference */
    UPROPERTY() 
    TScriptInterface<IPolicy> Policy;

    /** Each agent's policy state, parallel to Agents and advanced in place by each dispatched inference. Null entries for stateless policies. */
    UPROPERTY()
    TArray<TScriptInterface<IPolicyState>> CurrentStates;

    /** States whose reset was requested while inference was in flight */
    UPROPERTY()
    TArray<TScriptInterface<IPolicyState>> PendingStateResets;

    /** States of agents removed while inference was in flight, kept alive until the in-flight inference has finished using them */
    UPROPERTY()
    TArray<TScriptInterface<IPolicyState>> RetiredStates;

    /** Incremented whenever an agent is added or removed, so actions computed for an older set of agents are routed by agent instead of by index */
    uint64 MembershipVersion = 0;

    /** Game Thread flag set from dispatch until the result is handled, so the Game Thread never touches state while it is being written */
    bool bDispatchInFlight = false;

    /**
     * @brief Frame data structure for pipeline stages.
     * 
     * Renamed to avoid collision with UE's internal FFrame used in UFunction thunks.
     * Contains observations, actions, and synchronization flags for one pipeline stage.
     */
    struct FPipelinedStepperFrame
    {
        /** Observations collected from agents for this frame */
        TArray<TInstancedStruct<FPoint>> Observations;
        
        /** Actions computed by the policy for this frame */
        TArray<TInstancedStruct<FPoint>> Actions;

        /** Agents the dispatched inference computes Actions for, in the same order */
        TArray<TWeakObjectPtr<UObject>> DispatchedAgents;

        /** MembershipVersion when this frame was dispatched */
        uint64 DispatchedMembershipVersion = 0;
        
        /** Flag indicating actions are ready to be applied */
        std::atomic<bool> bActionsReady = false;
        
        /** Flag indicating inference is currently in progress */
        std::atomic<bool> bThinkInFlight = false;
        
        /** Debug ID for tracking dispatch order */
		uint64 DebugDispatchId = 0; 
    };

    /** Pipeline frame buffers (double-buffered) */
    FPipelinedStepperFrame Frames[PIPELINE_STAGES];
    
    /** Current tick counter for frame indexing */
    uint64 TickCounter = 0;
    
    /** Flag indicating the stepper is being destroyed */
    std::atomic<bool> bShuttingDown = false;

    /**  Debug sequence counter for dispatch tracking */
	std::atomic<uint64> DebugDispatchSeq{0};

    /**
     * @brief Dispatch asynchronous inference for a frame.
     * 
     * Spawns a background task to run policy inference, then schedules a game
     * thread task to apply the results once complete.
     * 
     * @param[in] FrameIndex Index of the pipeline frame to process
     */
    void DispatchThink(int32 FrameIndex);

    /**
     * @brief Handle a finished inference on the Game Thread: apply pending resets and release retired states.
     */
    void CompleteThink();

    /** @return The index of InAgent in Agents, or INDEX_NONE if it is not managed by this stepper */
    int32 FindAgentIndex(const UObject* InAgent) const;

};
