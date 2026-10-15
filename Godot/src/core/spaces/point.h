// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "core/spaces/space_kind.h"

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace schola {

class Point;

// Box values in row-major order.
struct BoxPoint {
	std::vector<float> values;
};

// Selected index in [0, n).
struct DiscretePoint {
	int32_t value = 0;
};

// One selected index per dimension; values[i] is in [0, nvec[i]).
struct MultiDiscretePoint {
	std::vector<int32_t> values;
};

// One 0 or 1 value per dimension.
struct MultiBinaryPoint {
	std::vector<uint8_t> values;
};

// Entries are ordered. A valid point has exactly the keys of its DictSpace, in the same order, which
// is sorted by key (see DictSpace). Protobuf Dict maps are unordered, so a converter must build
// entries by walking the DictSpace entries in order and looking up each key in the map, rather than
// iterating the map.
struct DictPoint {
	std::vector<std::pair<std::string, Point>> entries;
	// Returns a non-owning pointer to the entry for p_key, or nullptr if the key is absent.
	// The pointer is invalidated when entries is modified.
	Point *find(const std::string &p_key);
	const Point *find(const std::string &p_key) const;
};

// A value that belongs to a Space. Use Space::validate() to check that a point fits its space.
// A Point owns all of its nested data, is safe to copy and move, and holds no references to Godot,
// protobuf, ONNX, or caller-owned storage.
class Point {
public:
	// Constructs an empty Box point.
	Point();
	Point(BoxPoint p_point);
	Point(DiscretePoint p_point);
	Point(MultiDiscretePoint p_point);
	Point(MultiBinaryPoint p_point);
	Point(DictPoint p_point);

	SpaceKind get_kind() const;
	// Returns a non-owning pointer to the held value, or nullptr if the point holds a different kind.
	template <typename T>
	T *get_if();
	template <typename T>
	const T *get_if() const;
	std::string to_string() const;

	// Compares kind and values exactly, including Dict entry order.
	bool operator==(const Point &p_other) const;
	bool operator!=(const Point &p_other) const;

private:
	using Data = std::variant<BoxPoint, DiscretePoint, MultiDiscretePoint, MultiBinaryPoint, DictPoint>;
	Data data;
};

template <typename T>
T *Point::get_if() {
	return std::get_if<T>(&data);
}

template <typename T>
const T *Point::get_if() const {
	return std::get_if<T>(&data);
}

} // namespace schola
