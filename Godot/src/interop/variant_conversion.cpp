// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "interop/variant_conversion.h"

#include "core/common/status.h"
#include "core/spaces/point.h"
#include "core/spaces/space.h"

#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/string.hpp>

#include <string>

namespace schola {

namespace {

std::string type_name(const godot::Variant &p_value) {
	return godot::Variant::get_type_name(p_value.get_type()).utf8().get_data();
}

std::string at_path(const std::string &p_path, const std::string &p_message) {
	if (p_path.empty()) {
		return p_message;
	}
	return "at '" + p_path + "': " + p_message;
}

Status wrong_type(const std::string &p_path, const char *p_expected, const char *p_kind, const godot::Variant &p_value) {
	return Status::error(StatusCode::INVALID_ARGUMENT, at_path(p_path, std::string("expected ") + p_expected + " for a " + p_kind + " space but got " + type_name(p_value)));
}

Status wrong_size(const std::string &p_path, const char *p_kind, int64_t p_expected, int64_t p_actual) {
	return Status::error(StatusCode::INVALID_ARGUMENT, at_path(p_path, std::string(p_kind) + " space expects " + std::to_string(p_expected) + " values but got " + std::to_string(p_actual)));
}

Status convert(const godot::Variant &p_value, const Space &p_space, const std::string &p_path, Point &r_point) {
	if (const BoxSpace *box = p_space.get_if<BoxSpace>()) {
		if (p_value.get_type() != godot::Variant::PACKED_FLOAT32_ARRAY) {
			return wrong_type(p_path, "PackedFloat32Array", "Box", p_value);
		}
		const godot::PackedFloat32Array values = p_value;
		if (values.size() != static_cast<int64_t>(box->low.size())) {
			return wrong_size(p_path, "Box", static_cast<int64_t>(box->low.size()), values.size());
		}
		BoxPoint point;
		point.values.assign(values.ptr(), values.ptr() + values.size());
		r_point = Point(std::move(point));
		return Status::ok();
	}
	if (p_space.get_if<DiscreteSpace>()) {
		if (p_value.get_type() != godot::Variant::INT) {
			return wrong_type(p_path, "int", "Discrete", p_value);
		}
		r_point = Point(DiscretePoint{ static_cast<int32_t>(static_cast<int64_t>(p_value)) });
		return Status::ok();
	}
	if (const MultiDiscreteSpace *multi_discrete = p_space.get_if<MultiDiscreteSpace>()) {
		if (p_value.get_type() != godot::Variant::PACKED_INT32_ARRAY) {
			return wrong_type(p_path, "PackedInt32Array", "MultiDiscrete", p_value);
		}
		const godot::PackedInt32Array values = p_value;
		if (values.size() != static_cast<int64_t>(multi_discrete->nvec.size())) {
			return wrong_size(p_path, "MultiDiscrete", static_cast<int64_t>(multi_discrete->nvec.size()), values.size());
		}
		MultiDiscretePoint point;
		point.values.assign(values.ptr(), values.ptr() + values.size());
		r_point = Point(std::move(point));
		return Status::ok();
	}
	if (const MultiBinarySpace *multi_binary = p_space.get_if<MultiBinarySpace>()) {
		if (p_value.get_type() != godot::Variant::PACKED_BYTE_ARRAY) {
			return wrong_type(p_path, "PackedByteArray", "MultiBinary", p_value);
		}
		const godot::PackedByteArray values = p_value;
		if (values.size() != multi_binary->n) {
			return wrong_size(p_path, "MultiBinary", multi_binary->n, values.size());
		}
		MultiBinaryPoint point;
		point.values.assign(values.ptr(), values.ptr() + values.size());
		r_point = Point(std::move(point));
		return Status::ok();
	}

	const DictSpace *dict = p_space.get_if<DictSpace>();
	if (p_value.get_type() != godot::Variant::DICTIONARY) {
		return wrong_type(p_path, "Dictionary", "Dict", p_value);
	}
	const godot::Dictionary values = p_value;
	if (values.size() != static_cast<int64_t>(dict->entries.size())) {
		return Status::error(StatusCode::INVALID_ARGUMENT, at_path(p_path, "Dict space has " + std::to_string(dict->entries.size()) + " keys but the value has " + std::to_string(values.size())));
	}
	DictPoint point;
	for (const auto &entry : dict->entries) {
		const godot::String key(entry.first.c_str());
		const std::string entry_path = p_path.empty() ? entry.first : p_path + "." + entry.first;
		if (!values.has(key)) {
			return Status::error(StatusCode::INVALID_ARGUMENT, at_path(p_path, "Dict value is missing key '" + entry.first + "'"));
		}
		Point entry_point;
		Status status = convert(values[key], entry.second, entry_path, entry_point);
		if (!status.is_ok()) {
			return status;
		}
		point.entries.emplace_back(entry.first, std::move(entry_point));
	}
	r_point = Point(std::move(point));
	return Status::ok();
}

} // namespace

Status variant_to_point(const godot::Variant &p_value, const Space &p_space, Point &r_point) {
	return convert(p_value, p_space, {}, r_point);
}

godot::Variant point_to_variant(const Point &p_point) {
	if (const BoxPoint *box = p_point.get_if<BoxPoint>()) {
		godot::PackedFloat32Array values;
		values.resize(static_cast<int64_t>(box->values.size()));
		for (size_t i = 0; i < box->values.size(); ++i) {
			values.set(static_cast<int64_t>(i), box->values[i]);
		}
		return values;
	}
	if (const DiscretePoint *discrete = p_point.get_if<DiscretePoint>()) {
		return static_cast<int64_t>(discrete->value);
	}
	if (const MultiDiscretePoint *multi_discrete = p_point.get_if<MultiDiscretePoint>()) {
		godot::PackedInt32Array values;
		for (int32_t value : multi_discrete->values) {
			values.push_back(value);
		}
		return values;
	}
	if (const MultiBinaryPoint *multi_binary = p_point.get_if<MultiBinaryPoint>()) {
		godot::PackedByteArray values;
		for (uint8_t value : multi_binary->values) {
			values.push_back(value);
		}
		return values;
	}
	godot::Dictionary values;
	for (const auto &entry : p_point.get_if<DictPoint>()->entries) {
		values[godot::String(entry.first.c_str())] = point_to_variant(entry.second);
	}
	return values;
}

} // namespace schola
