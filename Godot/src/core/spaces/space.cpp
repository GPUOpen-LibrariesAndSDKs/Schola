// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "core/spaces/space.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>

namespace schola {

namespace {

const char *kind_name(SpaceKind p_kind) {
	switch (p_kind) {
		case SpaceKind::BOX:
			return "Box";
		case SpaceKind::DISCRETE:
			return "Discrete";
		case SpaceKind::MULTI_DISCRETE:
			return "MultiDiscrete";
		case SpaceKind::MULTI_BINARY:
			return "MultiBinary";
		case SpaceKind::DICT:
			return "Dict";
	}
	return "Unknown";
}

const char *dtype_name(DType p_dtype) {
	switch (p_dtype) {
		case DType::FLOAT32:
			return "float32";
		case DType::UINT8:
			return "uint8";
		case DType::UINT16:
			return "uint16";
		case DType::UINT32:
			return "uint32";
		case DType::UINT64:
			return "uint64";
		case DType::INT8:
			return "int8";
		case DType::INT16:
			return "int16";
		case DType::INT32:
			return "int32";
		case DType::INT64:
			return "int64";
		case DType::FLOAT16:
			return "float16";
		case DType::FLOAT64:
			return "float64";
		case DType::BOOL:
			return "bool";
	}
	return "unknown";
}

std::string at_path(const std::string &p_path, const std::string &p_message) {
	if (p_path.empty()) {
		return p_message;
	}
	return "at '" + p_path + "': " + p_message;
}

std::string join_path(const std::string &p_path, const std::string &p_key) {
	return p_path.empty() ? p_key : p_path + "." + p_key;
}

Status invalid_data(const std::string &p_path, const std::string &p_message) {
	return Status::error(StatusCode::INVALID_DATA, at_path(p_path, p_message));
}

Status invalid_argument(const std::string &p_path, const std::string &p_message) {
	return Status::error(StatusCode::INVALID_ARGUMENT, at_path(p_path, p_message));
}

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

std::string format_float(float p_value) {
	std::ostringstream stream;
	stream << p_value;
	return stream.str();
}

Status check_definition_at(const Space &p_space, const std::string &p_path);
Status validate_at(const Space &p_space, const Point &p_point, const std::string &p_path);

Status check_box(const BoxSpace &p_box, const std::string &p_path) {
	if (p_box.shape.empty()) {
		return invalid_data(p_path, "Box shape must have at least one dimension");
	}
	int64_t element_count = 1;
	for (size_t i = 0; i < p_box.shape.size(); ++i) {
		if (p_box.shape[i] <= 0) {
			return invalid_data(p_path, "Box shape[" + std::to_string(i) + "] is " + std::to_string(p_box.shape[i]) + " but must be positive");
		}
		element_count *= p_box.shape[i];
	}
	if (static_cast<int64_t>(p_box.low.size()) != element_count) {
		return invalid_data(p_path, "Box low has " + std::to_string(p_box.low.size()) + " values but shape " + format_list(p_box.shape) + " needs " + std::to_string(element_count));
	}
	if (static_cast<int64_t>(p_box.high.size()) != element_count) {
		return invalid_data(p_path, "Box high has " + std::to_string(p_box.high.size()) + " values but shape " + format_list(p_box.shape) + " needs " + std::to_string(element_count));
	}
	for (size_t i = 0; i < p_box.low.size(); ++i) {
		if (std::isnan(p_box.low[i])) {
			return invalid_data(p_path, "Box low[" + std::to_string(i) + "] is NaN");
		}
		if (std::isnan(p_box.high[i])) {
			return invalid_data(p_path, "Box high[" + std::to_string(i) + "] is NaN");
		}
		if (p_box.low[i] > p_box.high[i]) {
			return invalid_data(p_path, "Box low[" + std::to_string(i) + "] (" + format_float(p_box.low[i]) + ") is greater than high[" + std::to_string(i) + "] (" + format_float(p_box.high[i]) + ")");
		}
	}
	return Status::ok();
}

Status check_dict(const DictSpace &p_dict, const std::string &p_path) {
	for (size_t i = 1; i < p_dict.entries.size(); ++i) {
		if (p_dict.entries[i].first == p_dict.entries[i - 1].first) {
			return invalid_data(p_path, "Dict key '" + p_dict.entries[i].first + "' appears more than once");
		}
	}
	for (const auto &entry : p_dict.entries) {
		Status status = check_definition_at(entry.second, join_path(p_path, entry.first));
		if (!status.is_ok()) {
			return status;
		}
	}
	return Status::ok();
}

Status check_definition_at(const Space &p_space, const std::string &p_path) {
	if (const BoxSpace *box = p_space.get_if<BoxSpace>()) {
		return check_box(*box, p_path);
	}
	if (const DiscreteSpace *discrete = p_space.get_if<DiscreteSpace>()) {
		if (discrete->n <= 0) {
			return invalid_data(p_path, "Discrete n is " + std::to_string(discrete->n) + " but must be positive");
		}
		return Status::ok();
	}
	if (const MultiDiscreteSpace *multi_discrete = p_space.get_if<MultiDiscreteSpace>()) {
		if (multi_discrete->nvec.empty()) {
			return invalid_data(p_path, "MultiDiscrete nvec must not be empty");
		}
		for (size_t i = 0; i < multi_discrete->nvec.size(); ++i) {
			if (multi_discrete->nvec[i] <= 0) {
				return invalid_data(p_path, "MultiDiscrete nvec[" + std::to_string(i) + "] is " + std::to_string(multi_discrete->nvec[i]) + " but must be positive");
			}
		}
		return Status::ok();
	}
	if (const MultiBinarySpace *multi_binary = p_space.get_if<MultiBinarySpace>()) {
		if (multi_binary->n <= 0) {
			return invalid_data(p_path, "MultiBinary n is " + std::to_string(multi_binary->n) + " but must be positive");
		}
		return Status::ok();
	}
	return check_dict(*p_space.get_if<DictSpace>(), p_path);
}

Status validate_box(const BoxSpace &p_box, const BoxPoint &p_point, const std::string &p_path) {
	if (p_point.values.size() != p_box.low.size()) {
		return invalid_argument(p_path, "Box point has " + std::to_string(p_point.values.size()) + " values but the space expects " + std::to_string(p_box.low.size()));
	}
	for (size_t i = 0; i < p_point.values.size(); ++i) {
		const float value = p_point.values[i];
		if (std::isnan(value)) {
			return invalid_argument(p_path, "Box value at index " + std::to_string(i) + " is NaN");
		}
		if (value < p_box.low[i] || value > p_box.high[i]) {
			return invalid_argument(p_path, "Box value at index " + std::to_string(i) + " is " + format_float(value) + ", outside [" + format_float(p_box.low[i]) + ", " + format_float(p_box.high[i]) + "]");
		}
	}
	return Status::ok();
}

Status validate_dict(const DictSpace &p_dict, const DictPoint &p_point, const std::string &p_path) {
	std::set<std::string> seen;
	for (const auto &entry : p_point.entries) {
		if (!seen.insert(entry.first).second) {
			return invalid_argument(p_path, "Dict point key '" + entry.first + "' appears more than once");
		}
	}
	for (const auto &entry : p_dict.entries) {
		if (seen.count(entry.first) == 0) {
			return invalid_argument(p_path, "Dict point is missing key '" + entry.first + "'");
		}
	}
	for (const auto &entry : p_point.entries) {
		if (p_dict.find(entry.first) == nullptr) {
			return invalid_argument(p_path, "Dict point has unexpected key '" + entry.first + "'");
		}
	}
	for (size_t i = 0; i < p_dict.entries.size(); ++i) {
		if (p_point.entries[i].first != p_dict.entries[i].first) {
			return invalid_argument(p_path, "Dict point keys are not sorted: expected '" + p_dict.entries[i].first + "' at index " + std::to_string(i) + " but found '" + p_point.entries[i].first + "'");
		}
	}
	for (size_t i = 0; i < p_dict.entries.size(); ++i) {
		Status status = validate_at(p_dict.entries[i].second, p_point.entries[i].second, join_path(p_path, p_dict.entries[i].first));
		if (!status.is_ok()) {
			return status;
		}
	}
	return Status::ok();
}

Status validate_at(const Space &p_space, const Point &p_point, const std::string &p_path) {
	if (p_space.get_kind() != p_point.get_kind()) {
		return invalid_argument(p_path, std::string("expected a ") + kind_name(p_space.get_kind()) + " point but got a " + kind_name(p_point.get_kind()) + " point");
	}
	if (const BoxSpace *box = p_space.get_if<BoxSpace>()) {
		return validate_box(*box, *p_point.get_if<BoxPoint>(), p_path);
	}
	if (const DiscreteSpace *discrete = p_space.get_if<DiscreteSpace>()) {
		const int32_t value = p_point.get_if<DiscretePoint>()->value;
		if (value < 0 || value >= discrete->n) {
			return invalid_argument(p_path, "Discrete value " + std::to_string(value) + " is outside [0, " + std::to_string(discrete->n) + ")");
		}
		return Status::ok();
	}
	if (const MultiDiscreteSpace *multi_discrete = p_space.get_if<MultiDiscreteSpace>()) {
		const std::vector<int32_t> &values = p_point.get_if<MultiDiscretePoint>()->values;
		if (values.size() != multi_discrete->nvec.size()) {
			return invalid_argument(p_path, "MultiDiscrete point has " + std::to_string(values.size()) + " values but the space expects " + std::to_string(multi_discrete->nvec.size()));
		}
		for (size_t i = 0; i < values.size(); ++i) {
			if (values[i] < 0 || values[i] >= multi_discrete->nvec[i]) {
				return invalid_argument(p_path, "MultiDiscrete value at index " + std::to_string(i) + " is " + std::to_string(values[i]) + ", outside [0, " + std::to_string(multi_discrete->nvec[i]) + ")");
			}
		}
		return Status::ok();
	}
	if (const MultiBinarySpace *multi_binary = p_space.get_if<MultiBinarySpace>()) {
		const std::vector<uint8_t> &values = p_point.get_if<MultiBinaryPoint>()->values;
		if (static_cast<int64_t>(values.size()) != multi_binary->n) {
			return invalid_argument(p_path, "MultiBinary point has " + std::to_string(values.size()) + " values but the space expects " + std::to_string(multi_binary->n));
		}
		for (size_t i = 0; i < values.size(); ++i) {
			if (values[i] > 1) {
				return invalid_argument(p_path, "MultiBinary value at index " + std::to_string(i) + " is " + std::to_string(values[i]) + " but must be 0 or 1");
			}
		}
		return Status::ok();
	}
	return validate_dict(*p_space.get_if<DictSpace>(), *p_point.get_if<DictPoint>(), p_path);
}

bool box_equal(const BoxSpace &p_a, const BoxSpace &p_b) {
	return p_a.low == p_b.low && p_a.high == p_b.high && p_a.shape == p_b.shape && p_a.dtype == p_b.dtype;
}

} // namespace

const Space *DictSpace::find(const std::string &p_key) const {
	for (const auto &entry : entries) {
		if (entry.first == p_key) {
			return &entry.second;
		}
	}
	return nullptr;
}

Space::Space() :
		data(BoxSpace{}) {
}

Space::Space(BoxSpace p_space) :
		data(std::move(p_space)) {
}

Space::Space(DiscreteSpace p_space) :
		data(p_space) {
}

Space::Space(MultiDiscreteSpace p_space) :
		data(std::move(p_space)) {
}

Space::Space(MultiBinarySpace p_space) :
		data(p_space) {
}

Space::Space(DictSpace p_space) {
	std::stable_sort(p_space.entries.begin(), p_space.entries.end(), [](const auto &p_a, const auto &p_b) {
		return p_a.first < p_b.first;
	});
	data = std::move(p_space);
}

SpaceKind Space::get_kind() const {
	return static_cast<SpaceKind>(data.index());
}

Status Space::check_definition() const {
	return check_definition_at(*this, {});
}

Status Space::validate(const Point &p_point) const {
	return validate_at(*this, p_point, {});
}

Point Space::make_point() const {
	if (const BoxSpace *box = get_if<BoxSpace>()) {
		BoxPoint point;
		point.values.reserve(box->low.size());
		for (size_t i = 0; i < box->low.size(); ++i) {
			point.values.push_back(std::clamp(0.0f, box->low[i], box->high[i]));
		}
		return Point(std::move(point));
	}
	if (get_if<DiscreteSpace>()) {
		return Point(DiscretePoint{});
	}
	if (const MultiDiscreteSpace *multi_discrete = get_if<MultiDiscreteSpace>()) {
		return Point(MultiDiscretePoint{ std::vector<int32_t>(multi_discrete->nvec.size(), 0) });
	}
	if (const MultiBinarySpace *multi_binary = get_if<MultiBinarySpace>()) {
		return Point(MultiBinaryPoint{ std::vector<uint8_t>(static_cast<size_t>(std::max(multi_binary->n, 0)), 0) });
	}
	DictPoint point;
	for (const auto &entry : get_if<DictSpace>()->entries) {
		point.entries.emplace_back(entry.first, entry.second.make_point());
	}
	return Point(std::move(point));
}

int64_t Space::get_flattened_size() const {
	if (const BoxSpace *box = get_if<BoxSpace>()) {
		return static_cast<int64_t>(box->low.size());
	}
	if (const DiscreteSpace *discrete = get_if<DiscreteSpace>()) {
		return discrete->n;
	}
	if (const MultiDiscreteSpace *multi_discrete = get_if<MultiDiscreteSpace>()) {
		int64_t size = 0;
		for (int32_t n : multi_discrete->nvec) {
			size += n;
		}
		return size;
	}
	if (const MultiBinarySpace *multi_binary = get_if<MultiBinarySpace>()) {
		return multi_binary->n;
	}
	int64_t size = 0;
	for (const auto &entry : get_if<DictSpace>()->entries) {
		size += entry.second.get_flattened_size();
	}
	return size;
}

std::string Space::to_string() const {
	if (const BoxSpace *box = get_if<BoxSpace>()) {
		return "Box(low=" + format_list(box->low) + ", high=" + format_list(box->high) + ", shape=" + format_list(box->shape) + ", dtype=" + dtype_name(box->dtype) + ")";
	}
	if (const DiscreteSpace *discrete = get_if<DiscreteSpace>()) {
		return "Discrete(" + std::to_string(discrete->n) + ")";
	}
	if (const MultiDiscreteSpace *multi_discrete = get_if<MultiDiscreteSpace>()) {
		return "MultiDiscrete(" + format_list(multi_discrete->nvec) + ")";
	}
	if (const MultiBinarySpace *multi_binary = get_if<MultiBinarySpace>()) {
		return "MultiBinary(" + std::to_string(multi_binary->n) + ")";
	}
	std::string result = "Dict(";
	const DictSpace *dict = get_if<DictSpace>();
	for (size_t i = 0; i < dict->entries.size(); ++i) {
		if (i > 0) {
			result += ", ";
		}
		result += dict->entries[i].first + ": " + dict->entries[i].second.to_string();
	}
	return result + ")";
}

bool Space::operator==(const Space &p_other) const {
	if (get_kind() != p_other.get_kind()) {
		return false;
	}
	if (const BoxSpace *box = get_if<BoxSpace>()) {
		return box_equal(*box, *p_other.get_if<BoxSpace>());
	}
	if (const DiscreteSpace *discrete = get_if<DiscreteSpace>()) {
		return discrete->n == p_other.get_if<DiscreteSpace>()->n;
	}
	if (const MultiDiscreteSpace *multi_discrete = get_if<MultiDiscreteSpace>()) {
		return multi_discrete->nvec == p_other.get_if<MultiDiscreteSpace>()->nvec;
	}
	if (const MultiBinarySpace *multi_binary = get_if<MultiBinarySpace>()) {
		return multi_binary->n == p_other.get_if<MultiBinarySpace>()->n;
	}
	const auto &entries = get_if<DictSpace>()->entries;
	const auto &other_entries = p_other.get_if<DictSpace>()->entries;
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

bool Space::operator!=(const Space &p_other) const {
	return !(*this == p_other);
}

} // namespace schola
