// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "core/environment/environment.h"

#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <vector>

namespace schola::testing {

// Reference US1 agent: walks a square grid toward a fixed goal cell.
//
// Observation: Box(4) = [x, y, goal_x - x, goal_y - y].
// Action: Discrete(4) = up, down, left, right. Moves are clamped to the grid.
class GridAgent final : public Agent {
public:
	static constexpr int32_t SIZE = 5;
	static constexpr int32_t GOAL_X = SIZE - 1;
	static constexpr int32_t GOAL_Y = SIZE - 1;

	enum Move : int32_t {
		UP = 0,
		DOWN = 1,
		LEFT = 2,
		RIGHT = 3,
	};

	Status define(InteractionDefinition &r_definition) override;
	Status observe(Point &r_observation) override;
	Status act(const Point &p_action) override;

	bool is_at_goal() const;
	void place(int32_t p_x, int32_t p_y);
	int32_t get_x() const;
	int32_t get_y() const;

private:
	int32_t x = 0;
	int32_t y = 0;
};

// Reference US1 environment: one GridAgent, seeded random start, +1 at the goal.
//
// Follows every rule in core/environment/environment.h with no Godot types, so
// US4's lifecycle and US3's connector can be exercised against real behavior
// instead of canned answers. Truncation is left to the lifecycle's step limit.
class GridWorldEnvironment final : public Environment {
public:
	static constexpr const char *AGENT_ID = "runner";
	static constexpr float GOAL_REWARD = 1.0f;
	static constexpr float STEP_REWARD = -0.01f;

	GridWorldEnvironment();

	Status initialize(std::map<AgentId, InteractionDefinition> &r_agents) override;
	Status reset(const ResetSettings &p_settings, std::map<AgentId, InitialAgentState> &r_states) override;
	Status step(const std::map<AgentId, Point> &p_actions, std::map<AgentId, AgentState> &r_states) override;
	Status get_agents(std::vector<Agent *> &r_agents) const override;

private:
	// Held by pointer so the const get_agents() can hand out a non-const Agent*.
	std::unique_ptr<GridAgent> agent;
	InteractionDefinition definition;
	std::mt19937 rng;
	bool initialized = false;
	bool has_episode = false;
	bool agent_done = false;
};

} // namespace schola::testing
