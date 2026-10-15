// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/random_number_generator.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/variant.hpp>

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

	// === US1 Additions Below ===

	// Step order, so the override order here makes sense:
	//   1. ScholaAgent::_act runs for every agent still in the episode
	//   2. ScholaAgent::_observe, then _get_reward, _is_terminated, _is_truncated
	//      and _get_info run for each of those agents, in the same call
	//   3. Godot runs physics afterwards, so an action's effect may first appear
	//      in the next step's observation

	/**
	 * Override in GDScript: return the scene to a start state.
	 *
	 * In:   p_seed, an int when a seed was supplied and null otherwise;
	 *       p_options, free-form reset settings as a Dictionary.
	 *
	 * Does: called once per reset. When p_seed is an int, seed get_rng()
	 *       with it and draw all episode randomness from there; that is
	 *       what makes a run reproducible.
	 *
	 * Out:  nothing.
	 */
	GDVIRTUAL2(_reset_episode, godot::Variant, godot::Dictionary)

	/**
	 * Override in GDScript: reward for the step that just finished.
	 *
	 * In:   the agent this reward is for.
	 *
	 * Does: called once per agent each step, following _observe().
	 *
	 * Out:  a reward value. No default; every environment must decide this.
	 */
	GDVIRTUAL1R(double, _get_reward, ScholaAgent *)

	/**
	 * Override in GDScript: did this agent reach a real end state?
	 *
	 * In:   the agent in question.
	 *
	 * Does: called once per agent, same timing as _get_reward.
	 *
	 * Out:  true if the episode ended on its own terms for this agent.
	 *       Defaults to false if left unoverridden.
	 */
	GDVIRTUAL1R(bool, _is_terminated, ScholaAgent *)

	/**
	 * Override in GDScript: was this agent cut off early?
	 *
	 * In:   the agent in question.
	 *
	 * Does: called once per agent, same timing as _get_reward. A configured
	 *       step limit applies on top of whatever this returns.
	 *
	 * Out:  true if the episode was cut off externally for this agent.
	 *       Defaults to false if left unoverridden.
	 */
	GDVIRTUAL1R(bool, _is_truncated, ScholaAgent *)

	/**
	 * Override in GDScript: optional extra data passed through to Python.
	 *
	 * In:   the agent in question.
	 *
	 * Does: called once per agent, same timing as _get_reward.
	 *
	 * Out:  a Dictionary of String keys and String values.
	 */
	GDVIRTUAL1R(godot::Dictionary, _get_info, ScholaAgent *)

	// Steps completed since the last reset.
	int64_t get_episode_step() const;

	// This environment's random source. Use it for every episode-level
	// random choice; anything drawn from elsewhere ignores the seed and
	// makes runs unreproducible.
	godot::Ref<godot::RandomNumberGenerator> get_rng() const;

	// === End US1 Additions ===

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

	// === US1 Addition Below ===

	// signals, registered in _bind_methods:
	// episode_started: after _reset_episode, once observations are readable.
	// episode_ended: once every agent has terminated or truncated.

	// === End US1 Addition ===
};
