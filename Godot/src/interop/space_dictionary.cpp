// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "interop/space_dictionary.h"

#include "core/spaces/space.h"

#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <cmath>

namespace schola {

namespace {

const char *DTYPE_NAMES[] = {
	"float32",
	"uint8",
	"uint16",
	"uint32",
	"uint64",
	"int8",
	"int16",
	"int32",
	"int64",
	"float16",
	"float64",
	"bool",
};

godot::Array bounds_to_array(const std::vector<float> &p_bounds) {
	godot::Array result;
	for (float bound : p_bounds) {
		result.push_back(std::isinf(bound) ? godot::Variant() : godot::Variant(bound));
	}
	return result;
}

template <typename T>
godot::Array ints_to_array(const std::vector<T> &p_values) {
	godot::Array result;
	for (T value : p_values) {
		result.push_back(static_cast<int64_t>(value));
	}
	return result;
}

} // namespace

godot::Dictionary space_to_dictionary(const Space &p_space) {
	godot::Dictionary result;
	if (const BoxSpace *box = p_space.get_if<BoxSpace>()) {
		result["type"] = "Box";
		result["low"] = bounds_to_array(box->low);
		result["high"] = bounds_to_array(box->high);
		result["shape"] = ints_to_array(box->shape);
		result["dtype"] = DTYPE_NAMES[static_cast<int>(box->dtype)];
	} else if (const DiscreteSpace *discrete = p_space.get_if<DiscreteSpace>()) {
		result["type"] = "Discrete";
		result["n"] = discrete->n;
	} else if (const MultiDiscreteSpace *multi_discrete = p_space.get_if<MultiDiscreteSpace>()) {
		result["type"] = "MultiDiscrete";
		result["nvec"] = ints_to_array(multi_discrete->nvec);
	} else if (const MultiBinarySpace *multi_binary = p_space.get_if<MultiBinarySpace>()) {
		result["type"] = "MultiBinary";
		result["n"] = multi_binary->n;
	} else {
		godot::Dictionary spaces;
		for (const auto &entry : p_space.get_if<DictSpace>()->entries) {
			spaces[godot::String(entry.first.c_str())] = space_to_dictionary(entry.second);
		}
		result["type"] = "Dict";
		result["spaces"] = spaces;
	}
	return result;
}

} // namespace schola
