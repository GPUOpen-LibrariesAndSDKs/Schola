// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "core/common/status.h"
#include "core/spaces/point.h"
#include "core/spaces/space_kind.h"

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace schola {

class Space;

// Continuous values with per-element bounds. A valid definition has at least one dimension, every
// dimension is positive, and low and high each hold product(shape) entries in row-major order.
// low[i] <= high[i] and neither bound is NaN. An unbounded side, including a bound omitted from the
// protobuf definition, is stored as negative or positive infinity. An empty protobuf shape is converted
// to {low.size()}, matching Python Schola.
struct BoxSpace {
	std::vector<float> low;
	std::vector<float> high;
	std::vector<int32_t> shape;
	DType dtype = DType::FLOAT32;
};

// A single choice in [0, n), where n is positive.
struct DiscreteSpace {
	int32_t n = 0;
};

// Independent choices; dimension i is in [0, nvec[i]). nvec is non-empty and every entry is positive.
struct MultiDiscreteSpace {
	std::vector<int32_t> nvec;
};

// n independent 0 or 1 values, where n is positive.
struct MultiBinarySpace {
	int32_t n = 0;
};

// Named subspaces with unique keys. Space(DictSpace) sorts entries by key in byte-wise std::string
// order, which matches gymnasium's sorted() on Python str keys. That order fixes the flattened layout,
// so it matches Python Schola regardless of declaration order or protobuf map order.
struct DictSpace {
	std::vector<std::pair<std::string, Space>> entries;
	// Returns a non-owning pointer to the subspace for p_key, or nullptr if the key is absent.
	// The pointer is invalidated when entries is modified.
	const Space *find(const std::string &p_key) const;
};

// An observation or action space definition. A Space owns all of its nested data, is safe to copy
// and move, and holds no references to Godot, protobuf, ONNX, or caller-owned storage.
class Space {
public:
	// Constructs an empty Box space.
	Space();
	Space(BoxSpace p_space);
	Space(DiscreteSpace p_space);
	Space(MultiDiscreteSpace p_space);
	Space(MultiBinarySpace p_space);
	// Sorts p_space.entries by key. See DictSpace.
	Space(DictSpace p_space);

	SpaceKind get_kind() const;
	// Returns a non-owning pointer to the held definition, or nullptr if the space holds a different kind.
	template <typename T>
	const T *get_if() const;

	// Checks that the definition itself is well-formed, recursing into Dict entries. Returns
	// INVALID_DATA for a malformed definition, such as a non-positive dimension or size, mismatched
	// Box bound sizes, a NaN bound, low > high, or a duplicate Dict key. The message identifies the
	// failing field and its Dict key path.
	Status check_definition() const;
	// Checks the complete point against this space without modifying, clipping, coercing, reordering,
	// filling, or discarding values. Assumes the definition already passed check_definition(). Returns
	// INVALID_ARGUMENT for a wrong kind or size, a NaN or out-of-bounds Box value, an out-of-range
	// Discrete or MultiDiscrete value, a MultiBinary value other than 0 or 1, or Dict keys that are
	// missing, extra, duplicated, or not sorted by key. The message identifies the invalid element index
	// and its Dict key path.
	Status validate(const Point &p_point) const;
	// Returns a point of this space's kind and size that passes validate(). Every value is zero, except
	// Box elements, which are zero clamped to [low[i], high[i]].
	Point make_point() const;
	// Returns the flattened size, matching Unreal Schola: Box is the element count, Discrete is n
	// (one-hot), MultiDiscrete is the sum of nvec, MultiBinary is n, and Dict is the sum of its entries.
	int64_t get_flattened_size() const;
	std::string to_string() const;

	// Compares kind and definition exactly. Dict entries are always sorted, so declaration order does
	// not affect equality.
	bool operator==(const Space &p_other) const;
	bool operator!=(const Space &p_other) const;

private:
	using Data = std::variant<BoxSpace, DiscreteSpace, MultiDiscreteSpace, MultiBinarySpace, DictSpace>;
	Data data;
};

template <typename T>
const T *Space::get_if() const {
	return std::get_if<T>(&data);
}

} // namespace schola
