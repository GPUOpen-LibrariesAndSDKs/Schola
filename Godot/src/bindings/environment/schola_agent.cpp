// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "bindings/environment/schola_agent.h"

#include "bindings/spaces/schola_space.h"

#include <godot_cpp/core/class_db.hpp>

void ScholaAgent::set_agent_id(const godot::String &p_agent_id) {
	agent_id = p_agent_id;
	update_configuration_warnings();
}

godot::String ScholaAgent::get_agent_id() const {
	return agent_id.is_empty() ? godot::String(get_name()) : agent_id;
}

void ScholaAgent::set_agent_type(const godot::String &p_agent_type) {
	agent_type = p_agent_type;
}

godot::String ScholaAgent::get_agent_type() const {
	return agent_type.is_empty() ? get_agent_id() : agent_type;
}

void ScholaAgent::set_observation_space(const godot::Ref<ScholaSpace> &p_observation_space) {
	observation_space = p_observation_space;
	update_configuration_warnings();
}

godot::Ref<ScholaSpace> ScholaAgent::get_observation_space() const {
	return observation_space;
}

void ScholaAgent::set_action_space(const godot::Ref<ScholaSpace> &p_action_space) {
	action_space = p_action_space;
	update_configuration_warnings();
}

godot::Ref<ScholaSpace> ScholaAgent::get_action_space() const {
	return action_space;
}

godot::PackedStringArray ScholaAgent::get_configuration_errors() const {
	godot::PackedStringArray errors;
	if (observation_space.is_null()) {
		errors.append("observation_space is not set");
	} else {
		const godot::String error = observation_space->get_definition_error();
		if (!error.is_empty()) {
			errors.append("observation_space: " + error);
		}
	}
	if (action_space.is_null()) {
		errors.append("action_space is not set");
	} else {
		const godot::String error = action_space->get_definition_error();
		if (!error.is_empty()) {
			errors.append("action_space: " + error);
		}
	}
	return errors;
}

void ScholaAgent::_bind_methods() {
	godot::ClassDB::bind_method(godot::D_METHOD("set_agent_id", "agent_id"), &ScholaAgent::set_agent_id);
	godot::ClassDB::bind_method(godot::D_METHOD("get_agent_id"), &ScholaAgent::get_agent_id);
	godot::ClassDB::bind_method(godot::D_METHOD("set_agent_type", "agent_type"), &ScholaAgent::set_agent_type);
	godot::ClassDB::bind_method(godot::D_METHOD("get_agent_type"), &ScholaAgent::get_agent_type);
	godot::ClassDB::bind_method(godot::D_METHOD("set_observation_space", "observation_space"), &ScholaAgent::set_observation_space);
	godot::ClassDB::bind_method(godot::D_METHOD("get_observation_space"), &ScholaAgent::get_observation_space);
	godot::ClassDB::bind_method(godot::D_METHOD("set_action_space", "action_space"), &ScholaAgent::set_action_space);
	godot::ClassDB::bind_method(godot::D_METHOD("get_action_space"), &ScholaAgent::get_action_space);
	godot::ClassDB::bind_method(godot::D_METHOD("get_configuration_errors"), &ScholaAgent::get_configuration_errors);
	GDVIRTUAL_BIND(_observe);
	GDVIRTUAL_BIND(_act, "action");
	ADD_PROPERTY(godot::PropertyInfo(godot::Variant::STRING, "agent_id"), "set_agent_id", "get_agent_id");
	ADD_PROPERTY(godot::PropertyInfo(godot::Variant::STRING, "agent_type"), "set_agent_type", "get_agent_type");
	ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "observation_space", godot::PROPERTY_HINT_RESOURCE_TYPE, "ScholaSpace"), "set_observation_space", "get_observation_space");
	ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "action_space", godot::PROPERTY_HINT_RESOURCE_TYPE, "ScholaSpace"), "set_action_space", "get_action_space");
}

godot::PackedStringArray ScholaAgent::_get_configuration_warnings() const {
	return get_configuration_errors();
}
