// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "core/environment/agent.h"
#include "core/environment/interaction_definition.h"

// Dependency on US4
#if !__has_include("core/common/types.h")
#error "US1 needs the shared AgentId type from core/common/types.h"
#endif

#include "core/common/status.h"
#include "core/common/types.h"
#include "core/spaces/point.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace schola {

/**
 * What Python asks for when starting a new episode. Mirrors the shared
 * EnvironmentSettings message (Proto/StateUpdates.proto) every engine's
 * Gym connector receives, not an invented shape.
 */
struct ResetSettings {
	/**
	 * A seed forces the RNG's internal state to a known value, so draws from
	 * it are reproducible: same seed, same numbers, every run.
	 *
	 * Unset leaves that internal state untouched, the next draw just
	 * continues from wherever it already was. Still random, just not
	 * reproducible, since that state depends on everything drawn before it.
	 */
	std::optional<int32_t> seed;

	// One settings packet for this episode, e.g. {"track": "hard"}.
	std::map<std::string, std::string> options;
};

/**
 * An agent's first observation of a fresh episode. Mirrors the shared
 * InitialAgentState message (Proto/State.proto).
 */
struct InitialAgentState {
	// What the agent sees the instant reset() places it back at the start.
	Point observation;

	// Free-form string context for Python, e.g. {"spawn_tile": "A3"}. Not
	// part of what the agent learns from, it is just extra info riding along.
	std::map<std::string, std::string> info;
};

/**
 * An agent's observation resulting from some step. Mirrors the shared
 * AgentState message (Proto/State.proto).
 */
struct AgentState {
	// What the agent sees after this step's action was applied.
	Point observation;

	// How good this step was. The signal training actually learns from.
	float reward = 0.0f;

	// Episode ended on its own terms (goal reached, agent died).
	bool terminated = false;

	// Episode was cut off externally (step limit, stop request).
	// Not interchangeable with terminated, training treats them differently.
	bool truncated = false;

	// Free-form string context for Python, same role as InitialAgentState::info.
	std::map<std::string, std::string> info;
};

/**
 * A group of agents sharing one episode clock.
 *
 * One step() is one logical step, as in Unreal. Godot owns the physics
 * clock, so an action's physical effect may only show up in a later step's
 * observation; callers accept that lag rather than splitting the step
 * around a physics frame.
 *
 * Agent membership and spaces are fixed after initialize(). Every method
 * returns Status and leaves its out-parameter untouched on failure.
 */
class Environment {
public:
	virtual ~Environment() = default;

	/**
	 * Declares every agent and its spaces.
	 *
	 * In:   r_agents, an empty map.
	 *
	 * Does: declares every agent and its spaces. Called once, before the
	 *       first reset.
	 *
	 * Out:  r_agents filled with one InteractionDefinition per agent. This
	 *       is how Python first discovers what agents exist.
	 */
	virtual Status initialize(std::map<AgentId, InteractionDefinition> &r_agents) = 0;

	/**
	 * Starts a new episode.
	 *
	 * In:   p_settings, this episode's seed/options; r_states, an empty map.
	 *
	 * Does: starts a new episode. May be called again mid-episode to abandon
	 *       and restart.
	 *
	 * Out:  r_states filled with one InitialAgentState per agent.
	 */
	virtual Status reset(const ResetSettings &p_settings, std::map<AgentId, InitialAgentState> &r_states) = 0;

	/**
	 * Runs one logical step.
	 *
	 * In:   p_actions, one Point per still-running agent. A finished agent
	 *       is simply absent, not an error. r_states, an empty map.
	 *
	 * Does: applies each action, then reads back every agent that acted.
	 *       Does not wait for a physics frame in between, so the observation
	 *       may not yet reflect this step's action.
	 *
	 * Out:  r_states filled with one AgentState per agent in p_actions.
	 */
	virtual Status step(const std::map<AgentId, Point> &p_actions, std::map<AgentId, AgentState> &r_states) = 0;

	/**
	 * Hands out direct agent pointers, bypassing the episode machinery.
	 *
	 * In:   r_agents, an empty vector.
	 *
	 * Does: nothing to the world; just looks up the current agents.
	 *
	 * Out:  r_agents filled with non-owning Agent* pointers, for callers
	 *       that only need observe()/act() (e.g. local inference). Godot
	 *       owns the real objects; never delete these or keep them past
	 *       this Environment's lifetime.
	 */
	virtual Status get_agents(std::vector<Agent *> &r_agents) const = 0;
};

} // namespace schola
