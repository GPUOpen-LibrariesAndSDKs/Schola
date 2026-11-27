// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#pragma once
#include "CoreMinimal.h"
#include "IREEUtils/IREEBuffer.h"

/**
 * @brief Visitor interface for processing different types of IREE buffers
 *
 * This interface implements the visitor pattern, allowing type-safe operations
 * on different buffer types without requiring runtime type checking.
 */
struct SCHOLAIREE_API FIREEBufferVisitor
{

public:
	/**
	 * @brief Visit operation for dictionary buffers
	 * @param[in] InBuffer The dictionary buffer to process
	 */
	virtual void operator()(const FIREEDictBuffer& InBuffer) = 0;

	/**
	 * @brief Visit operation for box buffers
	 * @param[in] InBuffer The box buffer to process
	 */
	virtual void operator()(const FIREEBoxBuffer& InBuffer) = 0;

	/**
	 * @brief Visit operation for multi-binary buffers
	 * @param[in] InBuffer The multi-binary buffer to process
	 */
	virtual void operator()(const FIREEMultiBinaryBuffer& InBuffer) = 0;

	/**
	 * @brief Visit operation for discrete buffers
	 * @param[in] InBuffer The discrete buffer to process
	 */
	virtual void operator()(const FIREEDiscreteBuffer& InBuffer) = 0;

	/**
	 * @brief Visit operation for multi-discrete buffers
	 * @param[in] InBuffer The multi-discrete buffer to process
	 */
	virtual void operator()(const FIREEMultiDiscreteBuffer& InBuffer) = 0;

	virtual ~FIREEBufferVisitor() = default;
};
