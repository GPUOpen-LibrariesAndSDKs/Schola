// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Common/LogSchola.h"
#include "NNEUtils/NNEBufferVisitor.h"
#include "NNEUtils/NNEBuffer.h"
#include "Points/Point.h"
#include "Spaces/Space.h"
#include "StructUtils/InstancedStruct.h"

/**
 * @brief Visitor that converts NNE buffers into point representations
 * 
 * This visitor reads neural network output buffers and converts them into
 * Schola point structures, guided by space definitions to ensure proper interpretation.
 */
struct SCHOLANNE_API FNNEPointCreator : public FNNEBufferVisitor
{
	/** Reference to the output point being created */
	TInstancedStruct<FPoint>& OutputPoint;
	
	/** Space definition guiding the point creation */
	const TInstancedStruct<FSpace>& Space;

	/**
	 * @brief Constructor that initializes the point creator
	 * @param[out] OutPoint The point to be created from buffer data
	 * @param[in] InSpace The space definition that guides the conversion
	 */
	FNNEPointCreator(TInstancedStruct<FPoint>& OutPoint, const TInstancedStruct<FSpace>& InSpace)
		: OutputPoint(OutPoint), Space(InSpace)
	{
	}

	/**
	 * @brief Visit operation for dictionary buffers
	 * @param[in] InBuffer The dictionary buffer to convert to a point
	 */
	void operator()(const FNNEDictBuffer& InBuffer) override;

	/**
	 * @brief Visit operation for box buffers
	 * @param[in] InBuffer The box buffer to convert to a point
	 */
	void operator()(const FNNEBoxBuffer& InBuffer) override;

	/**
	 * @brief Visit operation for multi-binary buffers
	 * @param[in] InBuffer The multi-binary buffer to convert to a point
	 */
	void operator()(const FNNEMultiBinaryBuffer& InBuffer) override;

	/**
	 * @brief Visit operation for discrete buffers
	 * @param[in] InBuffer The discrete buffer to convert to a point
	 */
	void operator()(const FNNEDiscreteBuffer& InBuffer) override;

	/**
	 * @brief Visit operation for multi-discrete buffers
	 * @param[in] InBuffer The multi-discrete buffer to convert to a point
	 */
	void operator()(const FNNEMultiDiscreteBuffer& InBuffer) override;

	/**
	 * @brief Static utility function to create a point from a buffer
	 * @param[in] InBuffer The buffer containing the data to convert
	 * @param[out] OutPoint The point to be created
	 * @param[in] InSpace The space definition guiding the conversion
	 */
	inline static void CreatePoint(const TInstancedStruct<FNNEPointBuffer>& InBuffer, TInstancedStruct<FPoint>& OutPoint, const TInstancedStruct<FSpace>& InSpace)
	{
		FNNEPointCreator Creator(OutPoint, InSpace);
		InBuffer.Get<FNNEPointBuffer>().Accept(Creator);
	}

};
