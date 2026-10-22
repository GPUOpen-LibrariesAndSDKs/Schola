// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "bindings/spaces/schola_discrete_space.h"

#include "core/spaces/space.h"

#include <godot_cpp/core/class_db.hpp>

void ScholaDiscreteSpace::set_n(int32_t p_n) {
	n = p_n;
	emit_changed();
}

int32_t ScholaDiscreteSpace::get_n() const {
	return n;
}

schola::Space ScholaDiscreteSpace::to_space() const {
	return schola::Space(schola::DiscreteSpace{ n });
}

void ScholaDiscreteSpace::_bind_methods() {
	godot::ClassDB::bind_method(godot::D_METHOD("set_n", "n"), &ScholaDiscreteSpace::set_n);
	godot::ClassDB::bind_method(godot::D_METHOD("get_n"), &ScholaDiscreteSpace::get_n);

	ADD_PROPERTY(godot::PropertyInfo(godot::Variant::INT, "n"), "set_n", "get_n");
}
