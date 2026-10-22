// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "core/spaces/point.h"

#include <sstream>

namespace schola {

namespace {

template <typename T>
std::string format_list(const std::vector<T> &p_values) {
	std::ostringstream stream;
	stream << "[";
	for (size_t i = 0; i < p_values.size(); ++i) {
		if (i > 0) {
			stream << ", ";
		}
		stream << +p_values[i];
	}
	stream << "]";
	return stream.str();
}

} // namespace

Point *DictPoint::find(const std::string &p_key) {
	for (auto &entry : entries) {
		if (entry.first == p_key) {
			return &entry.second;
		}
	}
	return nullptr;
}

const Point *DictPoint::find(const std::string &p_key) const {
	for (const auto &entry : entries) {
		if (entry.first == p_key) {
			return &entry.second;
		}
	}
	return nullptr;
}

Point::Point() :
		data(BoxPoint{}) {
}

Point::Point(BoxPoint p_point) :
		data(std::move(p_point)) {
}

Point::Point(DiscretePoint p_point) :
		data(p_point) {
}

Point::Point(MultiDiscretePoint p_point) :
		data(std::move(p_point)) {
}

Point::Point(MultiBinaryPoint p_point) :
		data(std::move(p_point)) {
}

Point::Point(DictPoint p_point) :
		data(std::move(p_point)) {
}

SpaceKind Point::get_kind() const {
	return static_cast<SpaceKind>(data.index());
}

std::string Point::to_string() const {
	if (const BoxPoint *box = get_if<BoxPoint>()) {
		return "Box(" + format_list(box->values) + ")";
	}
	if (const DiscretePoint *discrete = get_if<DiscretePoint>()) {
		return "Discrete(" + std::to_string(discrete->value) + ")";
	}
	if (const MultiDiscretePoint *multi_discrete = get_if<MultiDiscretePoint>()) {
		return "MultiDiscrete(" + format_list(multi_discrete->values) + ")";
	}
	if (const MultiBinaryPoint *multi_binary = get_if<MultiBinaryPoint>()) {
		return "MultiBinary(" + format_list(multi_binary->values) + ")";
	}
	std::string result = "Dict(";
	const DictPoint *dict = get_if<DictPoint>();
	for (size_t i = 0; i < dict->entries.size(); ++i) {
		if (i > 0) {
			result += ", ";
		}
		result += dict->entries[i].first + ": " + dict->entries[i].second.to_string();
	}
	return result + ")";
}

bool Point::operator==(const Point &p_other) const {
	if (get_kind() != p_other.get_kind()) {
		return false;
	}
	if (const BoxPoint *box = get_if<BoxPoint>()) {
		return box->values == p_other.get_if<BoxPoint>()->values;
	}
	if (const DiscretePoint *discrete = get_if<DiscretePoint>()) {
		return discrete->value == p_other.get_if<DiscretePoint>()->value;
	}
	if (const MultiDiscretePoint *multi_discrete = get_if<MultiDiscretePoint>()) {
		return multi_discrete->values == p_other.get_if<MultiDiscretePoint>()->values;
	}
	if (const MultiBinaryPoint *multi_binary = get_if<MultiBinaryPoint>()) {
		return multi_binary->values == p_other.get_if<MultiBinaryPoint>()->values;
	}
	const auto &entries = get_if<DictPoint>()->entries;
	const auto &other_entries = p_other.get_if<DictPoint>()->entries;
	if (entries.size() != other_entries.size()) {
		return false;
	}
	for (size_t i = 0; i < entries.size(); ++i) {
		if (entries[i].first != other_entries[i].first || entries[i].second != other_entries[i].second) {
			return false;
		}
	}
	return true;
}

bool Point::operator!=(const Point &p_other) const {
	return !(*this == p_other);
}

} // namespace schola
