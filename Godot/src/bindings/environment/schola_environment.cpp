// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "bindings/environment/schola_environment.h"

#include <godot_cpp/core/class_db.hpp>

namespace {

void collect_agents(const godot::Node *p_node, godot::TypedArray<ScholaAgent> &r_agents) {
	const godot::TypedArray<godot::Node> children = p_node->get_children();
	for (int32_t index = 0; index < children.size(); ++index) {
		godot::Object *child_object = children[index];
		godot::Node *child = godot::Object::cast_to<godot::Node>(child_object);
		if (child == nullptr) {
			continue;
		}
		if (godot::Object::cast_to<ScholaEnvironment>(child) != nullptr) {
			continue;
		}
		if (ScholaAgent *agent = godot::Object::cast_to<ScholaAgent>(child)) {
			r_agents.push_back(agent);
		}
		collect_agents(child, r_agents);
	}
}

} // namespace

void ScholaEnvironment::set_max_episode_steps(int32_t p_max_episode_steps) {
	max_episode_steps = p_max_episode_steps;
	update_configuration_warnings();
}

int32_t ScholaEnvironment::get_max_episode_steps() const {
	return max_episode_steps;
}

godot::TypedArray<ScholaAgent> ScholaEnvironment::get_agents() const {
	godot::TypedArray<ScholaAgent> agents;
	collect_agents(this, agents);
	return agents;
}

godot::PackedStringArray ScholaEnvironment::get_configuration_errors() const {
	godot::PackedStringArray errors;
	if (max_episode_steps < 0) {
		errors.append("max_episode_steps must be zero or greater");
	}
	const godot::TypedArray<ScholaAgent> agents = get_agents();
	if (agents.is_empty()) {
		errors.append("Add a ScholaAgent node below this environment.");
	}
	for (int32_t index = 0; index < agents.size(); ++index) {
		godot::Object *agent_object = agents[index];
		ScholaAgent *agent = godot::Object::cast_to<ScholaAgent>(agent_object);
		if (agent == nullptr) {
			continue;
		}
		const godot::PackedStringArray agent_errors = agent->get_configuration_errors();
		for (int32_t error_index = 0; error_index < agent_errors.size(); ++error_index) {
			const godot::String error = agent_errors[error_index];
			errors.append("Agent '" + agent->get_agent_id() + "': " + error);
		}
	}
	return errors;
}

godot::PackedStringArray ScholaEnvironment::_get_configuration_warnings() const {
	return get_configuration_errors();
}

void ScholaEnvironment::_bind_methods() {
	godot::ClassDB::bind_method(godot::D_METHOD("set_max_episode_steps", "max_episode_steps"), &ScholaEnvironment::set_max_episode_steps);
	godot::ClassDB::bind_method(godot::D_METHOD("get_max_episode_steps"), &ScholaEnvironment::get_max_episode_steps);
	godot::ClassDB::bind_method(godot::D_METHOD("get_agents"), &ScholaEnvironment::get_agents);
	godot::ClassDB::bind_method(godot::D_METHOD("get_configuration_errors"), &ScholaEnvironment::get_configuration_errors);
	GDVIRTUAL_BIND(_reset_episode, "seed", "options");
	GDVIRTUAL_BIND(_get_reward, "agent");
	GDVIRTUAL_BIND(_is_terminated, "agent");
	GDVIRTUAL_BIND(_is_truncated, "agent");
	GDVIRTUAL_BIND(_get_info, "agent");
	ADD_PROPERTY(godot::PropertyInfo(godot::Variant::INT, "max_episode_steps"), "set_max_episode_steps", "get_max_episode_steps");
}
