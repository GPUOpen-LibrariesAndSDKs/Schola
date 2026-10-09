// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <cstdint>

namespace schola {
	class Environment;
} // namespace schola

class ScholaAgent;

// Godot-facing config node for one trainable environment.
// Gameplay behaviour and lifecycle execution are supplied by US1/US4
class ScholaEnvironment : public godot::Node {
	GDCLASS(ScholaEnvironment, godot::Node)

public:
	// 0: no step limit
	// positive: us4 truncates unfinished episode when it reaches this many steps
	// negative: config error
	void set_max_episode_steps(int32_t p_max_episode_steps);
	int32_t get_max_episode_steps() const;

	// returns the ScholaAgent descendants under this environment (scene-tree order)
	// descendant ScholaEnvironment nodes are config errors because they make ownership ambiguous
	// 		-> separate environment nodes in a scene are valid, nested are not
	godot::TypedArray<ScholaAgent> get_agents() const;

	// returns the US1 core adapter while this node is in the scene tree
	// (ScholaEnvironment is the godot wrapper, schola::Environment is abstraction owned by US1)
	schola::Environment *get_environment();
	const schola::Environment *get_environment() const;

	// warnings/errors; empty when the inspector and scene-tree configuration is valid
	godot::PackedStringArray get_configuration_errors() const;
	// 		-> might return: no agents, dupe agent IDs, nested environment, etc
	godot::PackedStringArray _get_configuration_warnings() const override; // godot hook
	//		-> lets godot display config warning icon in editor inspector

protected:
	static void _bind_methods();

private:
	int32_t max_episode_steps = 0;
};
