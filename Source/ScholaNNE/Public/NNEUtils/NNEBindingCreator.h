// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "NNE.h"
#include "NNERuntimeCPU.h"
#include "NNERuntimeGPU.h"
#include "NNEModelData.h"
#include "NNERuntimeRunSync.h"
#include "Spaces/Space.h"
#include "NNEUtils/NNEWrappers.h"
#include "NNEUtils/NNEBufferVisitor.h"
#include "NNEUtils/NNEBuffer.h"

/**
 * @brief Visitor that creates NNE tensor bindings from buffers
 * 
 * This visitor traverses buffer structures and creates corresponding NNE tensor bindings
 * that can be used for neural network inference, matching buffers to tensor descriptors.
 */
struct SCHOLANNE_API FNNEBindingCreator : public FNNEBufferVisitor
{

	/** Reference to the buffer to create bindings from */
	TInstancedStruct<FNNEPointBuffer>& Buffer;

	/** Array of tensor bindings to populate */
	TArray<UE::NNE::FTensorBindingCPU>& Bindings;

	/** Error flag indicating if binding creation failed */
	bool bError = false;

	/** Current tensor index being processed */
	int Index = 0;
	
	/** Array of tensor descriptors from the neural network model */
	TConstArrayView<UE::NNE::FTensorDesc> TensorDescs;

	/**
	 * @brief Constructor that initializes the binding creator
	 * @param[in] InBuffer The buffer to create bindings from
	 * @param[in] InTensorDescs The tensor descriptors from the model
	 * @param[out] OutBindings The array to populate with tensor bindings
	 */
	FNNEBindingCreator(TInstancedStruct<FNNEPointBuffer>& InBuffer,
		TConstArrayView<UE::NNE::FTensorDesc>			  InTensorDescs,
		TArray<UE::NNE::FTensorBindingCPU>&				  OutBindings)
		: Buffer(InBuffer), Bindings(OutBindings), TensorDescs(InTensorDescs)
	{
	}

	virtual ~FNNEBindingCreator() = default;

	/**
	 * @brief Visit operation for dictionary buffers
	 * @param[in] InBuffer The dictionary buffer to create bindings from
	 */
	void operator()(const FNNEDictBuffer& InBuffer) override;

	/**
	 * @brief Visit operation for box buffers
	 * @param[in] InBuffer The box buffer to create bindings from
	 */
	void operator()(const FNNEBoxBuffer& InBuffer) override;

	/**
	 * @brief Visit operation for multi-binary buffers
	 * @param[in] InBuffer The multi-binary buffer to create bindings from
	 */
	void operator()(const FNNEMultiBinaryBuffer& InBuffer) override;

	/**
	 * @brief Visit operation for discrete buffers
	 * @param[in] InBuffer The discrete buffer to create bindings from
	 */
	void operator()(const FNNEDiscreteBuffer& InBuffer) override;

	/**
	 * @brief Visit operation for multi-discrete buffers
	 * @param[in] InBuffer The multi-discrete buffer to create bindings from
	 */
	void operator()(const FNNEMultiDiscreteBuffer& InBuffer) override;

	/**
	 * @brief Static utility function to create tensor bindings from a buffer
	 * @param[in] InSpace The space definition (currently unused but kept for consistency)
	 * @param[in] InBuffer The buffer to create bindings from
	 * @param[in] InTensorDescs The tensor descriptors from the model
	 * @param[out] OutBindings The array to populate with tensor bindings
	 * @return true if an error occurred, false on success
	 */
	inline static bool CreateBindings(const TInstancedStruct<FSpace>& InSpace, TInstancedStruct<FNNEPointBuffer>& InBuffer, TConstArrayView<UE::NNE::FTensorDesc> InTensorDescs, TArray<UE::NNE::FTensorBindingCPU>& OutBindings)
	{
		FNNEBindingCreator InputBindingVisitor(InBuffer, InTensorDescs, OutBindings);
		InBuffer.Get<FNNEPointBuffer>().Accept(InputBindingVisitor);
		return InputBindingVisitor.bError;
	}
};
