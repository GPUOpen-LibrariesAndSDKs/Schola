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
#include <atomic>
#include "NNEPolicy.generated.h"

/**
 * @brief Policy implementation that uses trained NNE models for decision-making
 * 
 * This policy class integrates Unreal Engine's Neural Network Engine (NNE) with the Schola
 * reinforcement learning framework, enabling trained neural network models to control agents.
 * Supports both CPU and GPU inference runtimes.
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
	 * @param[out] OutAction The computed action (generic instanced struct)
	 * @return true if inference succeeded, false otherwise
	 */
	UFUNCTION(BlueprintCallable, Category = "Schola|Policy")
	bool Think(const FInstancedStruct& InObservations, FInstancedStruct& OutAction)
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

		return this->Think(ToTypedInstancedStruct<FPoint>(InObservations), ToTypedInstancedStruct<FPoint>(OutAction));
	}

	/**
	 * @brief Runs neural network inference to compute an action from observations
	 * @param[in] InObservations The observation point to process
	 * @param[out] OutActions The computed action point
	 * @return true if inference succeeded, false otherwise
	 */
	virtual bool Think(const TInstancedStruct<FPoint>& InObservations, TInstancedStruct<FPoint>& OutActions) override;

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

	/** Array of buffers storing recurrent state for sequence-based models */
	UPROPERTY(VisibleAnywhere,BlueprintReadOnly, Category="Policy Data")
	TArray<FNNEStateBuffer> StateBuffer;

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
	 * @brief Initializes state buffers and their bindings for recurrent models
	 * @param[in] InModelInstance The model instance to initialize state for
	 * @return true if initialization succeeded, false otherwise
	 */
	bool InitStateBuffersAndBindings(TSharedPtr<IModelInstanceRunSync> InModelInstance);

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

	/** Owned model ensuring the instance remains valid for the policy lifetime */
	TUniquePtr<IModelInterface> OwnedModel;

	/** Array of input tensor bindings for passing data to the model */
	TArray<UE::NNE::FTensorBindingCPU> InputBindings;

	/** Array of output tensor bindings for receiving data from the model */
	TArray<UE::NNE::FTensorBindingCPU> OutputBindings;

	/** Atomic flag preventing concurrent inference operations and buffer races */
	std::atomic<bool> bInferenceInFlight {false};
};
