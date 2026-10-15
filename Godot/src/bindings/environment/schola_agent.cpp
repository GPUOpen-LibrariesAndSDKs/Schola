// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "bindings/environment/schola_agent.h"

#include <godot_cpp/core/class_db.hpp>

void ScholaAgent::set_agent_type(const godot::String &p_agent_type) {
	agent_type = p_agent_type;
}

godot::String ScholaAgent::get_agent_type() const {
	return agent_type;
}

void ScholaAgent::_bind_methods() {
	// expose the optional policy-sharing label in the Inspector and GDScript
	godot::ClassDB::bind_method(godot::D_METHOD("set_agent_type", "agent_type"), &ScholaAgent::set_agent_type);
	godot::ClassDB::bind_method(godot::D_METHOD("get_agent_type"), &ScholaAgent::get_agent_type);
	ADD_PROPERTY(godot::PropertyInfo(godot::Variant::STRING, "agent_type"), "set_agent_type", "get_agent_type");
}
