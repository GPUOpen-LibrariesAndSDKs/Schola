// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>

namespace schola {
class Agent;
} // namespace schola

class ScholaSpace;

// Godot-facing config node for one agent
// When it has a ScholaEnvironment ancestor, that environment owns its training lifecycle
// A standalone agent is valid for local inference in a shipped game, where the
// training-only environment is absent
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

	// === US1 Additions Below ===

	/**
	 * Override in GDScript: report what this agent currently sees.
	 *
	 * In:   nothing.
	 *
	 * Does: called every step, after _act(). Read the world and build a
	 *       value matching the agent's declared observation space. The
	 *       world may not yet reflect this step's action.
	 *
	 * Out:  a Variant, converted into a Point using that space.
	 */
	GDVIRTUAL0R(godot::Variant, _observe)

	/**
	 * Override in GDScript: carry out one action.
	 *
	 * In:   a Variant action, converted from a Point. Validated against the
	 *       action space only when debug validation is on.
	 *
	 * Does: called every step, before _observe(). Apply the action to this
	 *       agent.
	 *
	 * Out:  nothing.
	 */
	GDVIRTUAL1(_act, godot::Variant)

	// === End US1 Additions ===

	// returns the US1 core adapter while this node is in the scene tree
	// (ScholaAgent is the godot wrapper, schola::Agent is abstraction owned by US1)
	// the adapter is the same whether or not a ScholaEnvironment registers this agent for training
	schola::Agent *get_agent();
	const schola::Agent *get_agent() const;

	// warnings/errors; empty when the inspector and scene-tree configuration is valid
	godot::PackedStringArray get_configuration_errors() const;
	// 		-> might return: missing action space, etc. (no parent environment is not an error)
	godot::PackedStringArray _get_configuration_warnings() const override; // godot hook
	//		-> lets godot display config warning icon in editor inspector

protected:
	static void _bind_methods();

private:
	godot::String agent_id;
	godot::Ref<ScholaSpace> observation_space;
	godot::Ref<ScholaSpace> action_space;
};
