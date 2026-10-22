// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "bindings/spaces/schola_space.h"

#include "core/common/status.h"
#include "core/spaces/point.h"
#include "core/spaces/space.h"
#include "interop/space_dictionary.h"
#include "interop/variant_conversion.h"

#include <godot_cpp/core/class_db.hpp>

namespace {

godot::String to_godot_string(const std::string &p_value) {
	return godot::String::utf8(p_value.c_str());
}

} // namespace

godot::String ScholaSpace::get_validation_error(const godot::Variant &p_value) const {
	const schola::Space space = to_space();
	schola::Status status = space.check_definition();
	if (!status.is_ok()) {
		return "invalid space definition: " + to_godot_string(status.get_message());
	}
	schola::Point point;
	status = schola::variant_to_point(p_value, space, point);
	if (status.is_ok()) {
		status = space.validate(point);
	}
	return status.is_ok() ? godot::String() : to_godot_string(status.get_message());
}

godot::String ScholaSpace::get_definition_error() const {
	const schola::Status status = to_space().check_definition();
	return status.is_ok() ? godot::String() : to_godot_string(status.get_message());
}

godot::Dictionary ScholaSpace::to_dictionary() const {
	return schola::space_to_dictionary(to_space());
}

int64_t ScholaSpace::get_flattened_size() const {
	const schola::Space space = to_space();
	return space.check_definition().is_ok() ? space.get_flattened_size() : 0;
}

godot::Variant ScholaSpace::make_default_value() const {
	const schola::Space space = to_space();
	if (!space.check_definition().is_ok()) {
		return godot::Variant();
	}
	return schola::point_to_variant(space.make_point());
}

godot::String ScholaSpace::describe() const {
	return to_godot_string(to_space().to_string());
}

void ScholaSpace::_bind_methods() {
	godot::ClassDB::bind_method(godot::D_METHOD("get_validation_error", "value"), &ScholaSpace::get_validation_error);
	godot::ClassDB::bind_method(godot::D_METHOD("get_definition_error"), &ScholaSpace::get_definition_error);
	godot::ClassDB::bind_method(godot::D_METHOD("to_dictionary"), &ScholaSpace::to_dictionary);
	godot::ClassDB::bind_method(godot::D_METHOD("get_flattened_size"), &ScholaSpace::get_flattened_size);
	godot::ClassDB::bind_method(godot::D_METHOD("make_default_value"), &ScholaSpace::make_default_value);
	godot::ClassDB::bind_method(godot::D_METHOD("describe"), &ScholaSpace::describe);
}
