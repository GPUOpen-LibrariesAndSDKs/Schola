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
        Agents = InAgents;
        Policy = InPolicy;
        TickCounter = 0;
        bDispatchInFlight = false;
        PendingStateResets.Reset();

        for (int i = 0; i < PIPELINE_STAGES; ++i)
        {
            Frames[i].Observations.Reset();
            Frames[i].Actions.Reset();
            Frames[i].bActionsReady = false;
            Frames[i].bThinkInFlight = false;
        }
        if (Agents.Num() == 0 || !Policy)
        {
            return false;
        }
        if (!CreateAgentStates(*Policy, this, Agents.Num(), CurrentStates)
            || !CreateAgentStates(*Policy, this, Agents.Num(), NextStates))
        {
            UE_LOGFMT(LogScholaInferenceUtils, Error, "UPipelinedStepper::Init(): Policy failed to create initial states");
            return false;
        }
        return true;
    }

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

    /** Each agent's policy state, read by the next dispatched inference. Null entries for stateless policies. */
    UPROPERTY()
    TArray<TScriptInterface<IPolicyState>> CurrentStates;

    /** Each agent's policy state written by the next dispatched inference, swapped with CurrentStates on success. */
    UPROPERTY()
    TArray<TScriptInterface<IPolicyState>> NextStates;

    /** Agents whose state reset was requested while inference was in flight */
    TSet<int32> PendingStateResets;

    /** Game Thread flag set from dispatch until the result is handled, so state is never read while being written */
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
     * @brief Handle a finished inference on the Game Thread: advance states and apply pending resets.
     * 
     * @param[in] bSuccess Whether the inference succeeded
     */
    void CompleteThink(bool bSuccess);

};
