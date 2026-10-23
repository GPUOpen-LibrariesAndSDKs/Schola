// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "core/environment/environment.h"

#include <map>
#include <vector>

namespace schola::testing {

class FakeEnvironment final : public Environment {
public:
	// Configured responses
	std::map<AgentId, InteractionDefinition> definitions;
	std::map<AgentId, InitialAgentState> reset_states;
	std::map<AgentId, AgentState> step_states;
	std::vector<Agent *> agents;

	// Injected results
	Status initialize_status = Status::ok();
	Status reset_status = Status::ok();
	Status step_status = Status::ok();
	Status get_agents_status = Status::ok();

	// Recorded calls
	int initialize_calls = 0;
	int reset_calls = 0;
	int step_calls = 0;
	mutable int get_agents_calls = 0; // get_agents must be const

	ResetSettings last_reset_settings;
	std::map<AgentId, Point> last_actions;
	// Optional non-owned trace shared by environments in coordinator tests.
	std::vector<std::string> *call_log = nullptr;
	std::string label;

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
	Status initialize(std::map<AgentId, InteractionDefinition> &r_agents) override;

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
	Status reset(const ResetSettings &p_settings, std::map<AgentId, InitialAgentState> &r_states) override;

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
	Status step(const std::map<AgentId, Point> &p_actions, std::map<AgentId, AgentState> &r_states) override;

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
	Status get_agents(std::vector<Agent *> &r_agents) const override;
};

} // namespace schola::testing
