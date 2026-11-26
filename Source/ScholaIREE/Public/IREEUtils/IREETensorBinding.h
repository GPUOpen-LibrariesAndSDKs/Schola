// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"

/**
 * @brief Raw memory binding for one IREE input or output tensor.
 *
 * Points at host memory owned by a Schola IREE buffer. The tensor's name, dtype and shape come from
 * the matching FScholaIREETensorInfo in the model metadata.
 */
struct FIREETensorBinding
{
	void*	  Data = nullptr;
	uint64	  SizeInBytes = 0;
};
