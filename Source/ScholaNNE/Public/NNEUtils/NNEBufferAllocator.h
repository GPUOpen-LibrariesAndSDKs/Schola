// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "NNEUtils/NNEBuffer.h"
#include "Spaces/Space.h"
#include "Spaces/SpaceVisitor.h"

/**
 * @brief Visitor that allocates appropriate buffer types based on space definitions
 * 
 * This visitor traverses space definitions and creates corresponding buffer instances
 * with appropriate sizes and types for neural network inference.
 */
struct SCHOLANNE_API FNNEBufferAllocator : public ConstSpaceVisitor
{

	/** Reference to the buffer being allocated */
	TInstancedStruct<FNNEPointBuffer>& Buffer;

public:
	/**
	 * @brief Constructor that takes a reference to the output buffer
	 * @param[out] OutBuffer The buffer to be allocated based on space definition
	 */
	FNNEBufferAllocator(TInstancedStruct<FNNEPointBuffer>& OutBuffer)
		: Buffer(OutBuffer)
	{
	}

	/**
	 * @brief Visit operation for dictionary spaces
	 * @param[in] InSpace The dictionary space definition
	 */
	void operator()(const FDictSpace& InSpace) override;

	/**
	 * @brief Visit operation for box spaces
	 * @param[in] InSpace The box space definition
	 */
	void operator()(const FBoxSpace& InSpace) override;

	/**
	 * @brief Visit operation for multi-binary spaces
	 * @param[in] InSpace The multi-binary space definition
	 */
	void operator()(const FMultiBinarySpace& InSpace) override;

	/**
	 * @brief Visit operation for discrete spaces
	 * @param[in] InSpace The discrete space definition
	 */
	void operator()(const FDiscreteSpace& InSpace) override;

	/**
	 * @brief Visit operation for multi-discrete spaces
	 * @param[in] InSpace The multi-discrete space definition
	 */
	void operator()(const FMultiDiscreteSpace& InSpace) override;

	/**
	 * @brief Static utility function to allocate a buffer based on a space definition
	 * @param[in] InSpace The space definition to allocate a buffer for
	 * @param[out] OutBuffer The allocated buffer
	 */
	inline static void AllocateBuffer(const TInstancedStruct<FSpace>& InSpace, TInstancedStruct<FNNEPointBuffer>& OutBuffer)
	{
		FNNEBufferAllocator Creator(OutBuffer);
		InSpace.Get<FSpace>().Accept(Creator);
	}
};
