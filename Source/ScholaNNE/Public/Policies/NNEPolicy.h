// Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "NNEModelData.h"
#include "Common/InteractionDefinition.h"
#include "Common/InstancedStructUtils.h"
#include "Common/BlueprintErrorUtils.h"
#include "NNEUtils/NNEWrappers.h"
#include "NNEUtils/NNEBuffer.h"
#include "Policies/PolicyInterface.h"
#include "Policies/ActionChunker.h"
#include "Policies/NNEPolicyState.h"
#include <atomic>
#include "NNEPolicy.generated.h"

/**
 * @brief Policy implementation that uses trained NNE models for decision-making
 * 
 * This policy class integrates Unreal Engine's Neural Network Engine (NNE) with the Schola
 * reinforcement learning framework, enabling trained neural network models to control agents.
 * Supports both CPU and GPU inference runtimes.
 *
 * Models whose action output holds several actions per inference, such as LeRobot ACT
 * with output [1, chunk_size, action_dim], are detected in Init. Each Think then plays
 * one action from the chunk as configured by ActionChunking.
 *
 * @note Text spaces (FTextSpace) are not yet supported for NNE inference. Supported spaces are
 * Box, Discrete, MultiDiscrete, MultiBinary, and Dict (and nested combinations thereof).
 */
UCLASS(Blueprintable, BlueprintType, EditInlineNew, DefaultToInstanced)
class SCHOLANNE_API UNNEPolicy : public UObject, public IPolicy
{
	GENERATED_BODY()

public:

	/** Defines the observation and action spaces for this policy */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Policy Definition")
	FInteractionDefinition PolicyDefinition;

	/** The neural network model data asset used for inference */
	UPROPERTY(EditAnywhere,BlueprintReadWrite,meta=(ExposeOnSpawn=true), Category="Policy Properties")
	TObjectPtr<UNNEModelData> ModelData;

	/** The runtime name for inference execution. Enable NNE runtime plugins (e.g., NNERuntimeORT) to see options */
	UPROPERTY(EditAnywhere,BlueprintReadWrite, meta = (GetOptions = "GetRuntimeNames",ExposeOnSpawn=true), Category="Policy Properties")
	FString RuntimeName;

	/** Maximum sequence length for state buffers with a sequence dimension */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ExposeOnSpawn = true, ClampMin = 1), Category = "Policy Properties")
	int MaxStateSequenceLength = 1;

	/** How a model that predicts action chunks, such as ACT, plays each chunk. Ignored for other models. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, AdvancedDisplay, Category = "Policy Properties")
	FActionChunkerSettings ActionChunking;

	/** Actions the model predicts per inference, found in Init. 1 for models that predict a single action. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, AdvancedDisplay, Category = "Policy Properties")
	int32 ActionChunkSize = 1;

	/**
	 * @brief Finds how many actions a model output holds
	 * @param[in] OutputShape Symbolic shape of the action output tensor. Only the leading batch dimension may be dynamic.
	 * @param[in] ActionDim Number of values in one action
	 * @return The number of actions in the output, 1 if it holds one action or has other dynamic dimensions, or 0 if it is larger than one action but not a whole number of actions
	 */
	static int32 GetActionChunkSize(TConstArrayView<int32> OutputShape, int32 ActionDim);

	/** @return True if Init found that the model predicts several actions per inference */
	bool IsActionChunked() const { return this->ChunkedAction.Num() > 0; }

	/**
	 * @brief Retrieves all available NNE runtime names
	 * @return Array of available runtime names that can be used for inference
	 */
	UFUNCTION(Category="Policy Utilities")
	TArray<FString> GetRuntimeNames() const;

	/**
	 * @brief Gets the singleton runtime instance with the specified name
	 * @param[in] SelectedRuntimeName The name of the runtime to retrieve
	 * @return Pointer to the runtime interface, or nullptr if not found
	 */
	IRuntimeInterface* GetRuntime(const FString& SelectedRuntimeName) const;

	/** Flag indicating whether the neural network model loaded successfully */
	UPROPERTY(VisibleAnywhere, Category="Policy Properties")
	bool bNetworkLoaded = false;

	/**
	 * @brief Blueprint-callable wrapper for the Think function
	 * @param[in] InObservations The observations to process (generic instanced struct)
	 * @param[in] InOutState The model state, advanced in place to the next state for recurrent models
	 * @param[out] OutAction The computed action (generic instanced struct)
	 * @return true if inference succeeded, false otherwise
	 */
	UFUNCTION(BlueprintCallable, Category = "Schola|Policy")
	bool Think(const FInstancedStruct& InObservations, const TScriptInterface<IPolicyState>& InOutState, FInstancedStruct& OutAction)
	{
		if (!InObservations.GetScriptStruct())
		{
			RaiseInvalidInstancedStructError(TEXT("NNEPolicy::Think"));
			return false;
		}

		if (!InObservations.GetScriptStruct()->IsChildOf(FPoint::StaticStruct()))
		{
			RaiseInstancedStructTypeMismatchError(InObservations, TEXT("FPoint"), TEXT("NNEPolicy::Think"));
			return false;
		}

		if (OutAction.GetScriptStruct() && !OutAction.GetScriptStruct()->IsChildOf(FPoint::StaticStruct()))
		{
			RaiseInstancedStructTypeMismatchError(OutAction, TEXT("FPoint"), TEXT("NNEPolicy::Think"));
			return false;
		}

		return this->Think(ToTypedInstancedStruct<FPoint>(InObservations), InOutState, ToTypedInstancedStruct<FPoint>(OutAction));
	}

	/**
	 * @brief Creates a zeroed UNNEPolicyState matching the model's state tensors
	 *
	 * For models that predict action chunks, the state's ActionChunker is also set up to hold the chunk being played.
	 *
	 * @param[out] OutState Receives the new state, with no buffers if the model is stateless
	 * @return true if the state was created, false if the policy is not initialized
	 */
	virtual bool CreateInitialState(TScriptInterface<IPolicyState>& OutState) const override;

	/**
	 * @brief Blueprint-callable wrapper for CreateInitialState
	 * @return A new zeroed UNNEPolicyState, or null if the policy is not initialized
	 */
	UFUNCTION(BlueprintCallable, Category = "Schola|Policy")
	TScriptInterface<IPolicyState> CreateInitialState() const
	{
		TScriptInterface<IPolicyState> State;
		this->CreateInitialState(State);
		return State;
	}

	/**
	 * @brief Runs neural network inference to compute an action and advance the recurrent state
	 *
	 * InOutState must be a UNNEPolicyState created by CreateInitialState. For recurrent
	 * models, the model writes its newest state into the pending slot of each state buffer,
	 * which is not part of the state yet. If inference succeeds, NotifyStateUpdate moves it
	 * into the state, so the state only changes on success. Stateless models use a
	 * UNNEPolicyState with no buffers.
	 *
	 * For models that predict action chunks, inference only runs when the state's chunk
	 * needs a new prediction, and OutActions is the next action from the chunk.
	 *
	 * @param[in] InObservations The observation point to process
	 * @param[in,out] InOutState The recurrent state from the previous step, advanced in place to the next state
	 * @param[out] OutActions The computed action point
	 * @return true if an action was written, false otherwise
	 */
	virtual bool Think(
		const TInstancedStruct<FPoint>&		  InObservations,
		const TScriptInterface<IPolicyState>& InOutState,
		TInstancedStruct<FPoint>&			  OutActions) override;

	/**
	 * @brief Initializes the policy with the given interaction definition
	 * @param[in] InPolicyDefinition The definition of observation and action spaces
	 * @return true if initialization succeeded, false otherwise
	 */
	UFUNCTION(BlueprintCallable, Category = "Schola|Policy")
	bool Init(const FInteractionDefinition& InPolicyDefinition) override;

	/** Buffer storing action data for neural network output */
	UPROPERTY(VisibleAnywhere,BlueprintReadOnly, Category="Policy Data")
	TInstancedStruct<FNNEPointBuffer> ActionBuffer;

	/** Buffer storing observation data for neural network input */
	UPROPERTY(VisibleAnywhere,BlueprintReadOnly, Category="Policy Data")
	TInstancedStruct<FNNEPointBuffer> ObservationBuffer;

	/** Layout of each state tensor pair in the model, read once in Init and used to allocate new states */
	UPROPERTY(VisibleAnywhere,BlueprintReadOnly, Category="Policy Data")
	TArray<FNNEStateLayout> StateLayouts;

	/**
	 * @brief Checks if an inference operation is currently running
	 * @return true if inference is in progress, false otherwise
	 */
	UFUNCTION(BlueprintCallable, Category="Policy Properties")
	bool IsInferenceBusy() const override;

protected:

	/**
	 * @brief Initializes the input tensor shapes for the model instance
	 * @param[in] InModelInstance The model instance to configure
	 * @return true if initialization succeeded, false otherwise
	 */
	bool InitInputTensorShapes(TSharedPtr<IModelInstanceRunSync> InModelInstance);

	/**
	 * @brief Allocates arrays for input and output tensor bindings
	 * @param[in] InModelInstance The model instance to allocate bindings for
	 * @return true if allocation succeeded, false otherwise
	 */
	bool AllocateBindingArrays(TSharedPtr<IModelInstanceRunSync> InModelInstance);

	/**
	 * @brief Finds the state tensors of a recurrent model and records the layout of each one
	 * @param[in] InModelInstance The model instance to initialize state for
	 * @return true if initialization succeeded, false otherwise
	 */
	bool InitStateBuffersAndBindings(TSharedPtr<IModelInstanceRunSync> InModelInstance);

	/**
	 * @brief Converts a policy state to a UNNEPolicyState
	 * @param[in] InState The state to convert
	 * @return The state as a UNNEPolicyState, or null if it is not one
	 */
	static UNNEPolicyState* AsNNEPolicyState(const TScriptInterface<IPolicyState>& InState);

	/**
	 * @brief Checks that a state object was created by this policy for the current model
	 * @param[in] InState The state to check
	 * @return true if InState has one buffer per model state tensor, with the same shape, state size and buffer size, and an ActionChunker sized for this model's action chunk
	 */
	bool IsStateCompatible(const UNNEPolicyState& InState) const;
	/**
	 * @brief Points the state input bindings at InOutState's state sequences and the state output bindings at its pending state vectors
	 * @param[in] InOutState The recurrent state from the previous step
	 * @return true if InOutState matches this model's state tensors and was bound, false otherwise
	 */
	bool BindState(UNNEPolicyState& InOutState);

	/**
	 * @brief Initializes buffers for non-state data (observations and actions)
	 * @param[in] InDefinition The interaction definition specifying buffer sizes
	 */
	void InitNonStateBuffers(const FInteractionDefinition& InDefinition);

	/**
	 * @brief Initializes tensor bindings for non-state data
	 * @param[in] InDefinition The interaction definition specifying binding structure
	 * @return true if initialization succeeded, false otherwise
	 */
	bool InitNonStateBindings(const FInteractionDefinition& InDefinition);

	/**
	 * @brief Detects an action output that holds several actions and binds it to ActionChunkBuffer
	 * @param[in] InModelInstance The model instance whose output tensors are checked
	 * @return true if the outputs match the action space, false if an action output is not a whole number of actions
	 */
	bool InitActionChunking(TSharedPtr<IModelInstanceRunSync> InModelInstance);

	/**
	 * @brief Sets up all buffers and bindings for the policy
	 * @param[in] PolicyDefinition The interaction definition
	 * @param[in] ModelInstance The model instance to set up for
	 * @return true if setup succeeded, false otherwise
	 */
	bool SetupBuffersAndBindings(const FInteractionDefinition& PolicyDefinition, TSharedPtr<IModelInstanceRunSync> ModelInstance);

	
private:
	/** The runtime interface used for neural network inference */
	TUniquePtr<IRuntimeInterface> Runtime;

	/** The model interface created by the runtime */
	TUniquePtr<IModelInterface> Model;

	/** The instantiated model ready for inference */
	TSharedPtr<IModelInstanceRunSync> ModelInstance;

	/** Array of input tensor bindings for passing data to the model */
	TArray<UE::NNE::FTensorBindingCPU> InputBindings;

	/** Array of output tensor bindings for receiving data from the model */
	TArray<UE::NNE::FTensorBindingCPU> OutputBindings;

	/** Input binding index of each state_in tensor, parallel to StateLayouts */
	TArray<int32> StateInputBindingIndices;

	/** Output binding index of each state_out tensor, parallel to StateLayouts */
	TArray<int32> StateOutputBindingIndices;

	/** State object currently referenced by the state tensor bindings */
	TWeakObjectPtr<UNNEPolicyState> BoundState;

	/** Model output of a chunked model, [chunk_size, action_dim], bound in place of the action buffer */
	TArray<float> ActionChunkBuffer;

	/** The memory in ActionBuffer that the chunked output was bound to, which receives one action from the chunk each step. Empty if the model is not chunked */
	TArrayView<float> ChunkedAction;

	/** Atomic flag preventing concurrent inference operations and buffer races */
	std::atomic<bool> bInferenceInFlight {false};
};
