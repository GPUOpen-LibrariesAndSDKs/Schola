// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

namespace schola {

// Identifies which alternative a Space or Point holds.
enum class SpaceKind {
	BOX,
	DISCRETE,
	MULTI_DISCRETE,
	MULTI_BINARY,
	DICT,
};

// Declared element type of Box values, in the same order as Proto/DType.proto. It is carried so that
// Python rebuilds the Box with its original dtype, such as uint8 for images. Values are stored as
// float in every case, matching the protobuf BoxPoint wire format.
enum class DType {
	FLOAT32,
	UINT8,
	UINT16,
	UINT32,
	UINT64,
	INT8,
	INT16,
	INT32,
	INT64,
	FLOAT16,
	FLOAT64,
	BOOL,
};

} // namespace schola
