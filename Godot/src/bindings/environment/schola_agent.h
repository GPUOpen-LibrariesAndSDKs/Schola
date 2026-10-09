// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>

namespace schola {
	class Agent;
} // namespace schola

class ScholaSpace;

// Godot-facing config node for one agent
// belongs to the nearest ancestor ScholaEnvironment, else the node has a config error
class ScholaAgent : public godot::Node {
	GDCLASS(ScholaAgent, godot::Node)

public:
	// IDs must be unique within one environment
	// empty agent_id : use the godot node name
	// nonempty id : use configured value
	void set_agent_id(const godot::String &p_agent_id);
	godot::String get_agent_id() const;

	// observation and action spaces; these will be inspector resource fields
	// note: we use godot::Ref<T> to allow multiple agents to point to the same resource
	void set_observation_space(const godot::Ref<ScholaSpace> &p_observation_space);
	godot::Ref<ScholaSpace> get_observation_space() const;
	void set_action_space(const godot::Ref<ScholaSpace> &p_action_space);
	godot::Ref<ScholaSpace> get_action_space() const;
	// note: US5 does not duplicate Box, Discrete, bounds, points, validation logic from US2

	// returns the US1 core adapter while this node is owned by an environment
	// (ScholaAgent is the godot wrapper, schola::Agent is abstraction owned by US1)
	schola::Agent *get_agent();
	const schola::Agent *get_agent() const;

	// warnings/errors; empty when the inspector and scene-tree configuration is valid
	godot::PackedStringArray get_configuration_errors() const;
	// 		-> might return: no parent environemnt, missing action space, etc.
	godot::PackedStringArray _get_configuration_warnings() const override; // godot hook
	//		-> lets godot display config warning icon in editor inspector

protected:
	static void _bind_methods();

private:
	godot::String agent_id;
	godot::Ref<ScholaSpace> observation_space;
	godot::Ref<ScholaSpace> action_space;
};
