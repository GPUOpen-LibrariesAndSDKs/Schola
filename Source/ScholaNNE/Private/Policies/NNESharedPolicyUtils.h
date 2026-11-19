// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once

#include "CoreMinimal.h"
#include "NNEModelData.h"
#include "Common/InteractionDefinition.h"

/**
 * @brief Model-contract helpers shared by the NNE-backed ISharedPolicy implementations.
 */
namespace NNESharedPolicyUtils
{
	/**
	 * @brief Reads a model's tensor names and widths and derives the definition the batch provider needs.
	 *
	 * The model must have exactly one float [batch, width] input. Every output must be a float
	 * [batch, width] tensor with a unique name; the action space is a dict keyed by output name.
	 * A throwaway model instance is created to read the tensors and released before returning.
	 *
	 * @param[in] ModelData Imported ONNX model.
	 * @param[in] RuntimeName Registered NNE CPU runtime, such as NNERuntimeORTCpu.
	 * @param[out] OutDefinition Observation and action spaces matching the model's tensors.
	 * @param[out] OutError Reason for failure.
	 * @return True if the model satisfies the contract.
	 */
	bool BuildDefinition(UNNEModelData* ModelData, const FString& RuntimeName,
		FInteractionDefinition& OutDefinition, FString& OutError);
} // namespace NNESharedPolicyUtils
