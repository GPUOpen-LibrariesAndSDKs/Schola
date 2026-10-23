// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "grid_world_environment.h"

#include <algorithm>
#include <utility>

namespace schola::testing {

namespace {

Space make_observation_space() {
	const float max_cell = static_cast<float>(GridAgent::SIZE - 1);
	return Space(BoxSpace{
			{ 0.0f, 0.0f, -max_cell, -max_cell },
			{ max_cell, max_cell, max_cell, max_cell },
			{ 4 },
	});
}

Space make_action_space() {
	return Space(DiscreteSpace{ 4 });
}

} // namespace

Status GridAgent::define(InteractionDefinition &r_definition) {
	r_definition = InteractionDefinition{ make_observation_space(), make_action_space(), "" };
	return Status::ok();
}

Status GridAgent::observe(Point &r_observation) {
	r_observation = Point(BoxPoint{ {
			static_cast<float>(x),
			static_cast<float>(y),
			static_cast<float>(GOAL_X - x),
			static_cast<float>(GOAL_Y - y),
	} });
	return Status::ok();
}

Status GridAgent::act(const Point &p_action) {
	// The Agent contract: a mismatched Point is rejected, never trusted.
	const Status valid = make_action_space().validate(p_action);
	if (!valid.is_ok()) {
		return valid;
	}

	switch (p_action.get_if<DiscretePoint>()->value) {
		case UP:
			y = std::min(y + 1, SIZE - 1);
			break;
		case DOWN:
			y = std::max(y - 1, 0);
			break;
		case LEFT:
			x = std::max(x - 1, 0);
			break;
		case RIGHT:
			x = std::min(x + 1, SIZE - 1);
			break;
	}
	return Status::ok();
}

bool GridAgent::is_at_goal() const {
	return x == GOAL_X && y == GOAL_Y;
}

void GridAgent::place(int32_t p_x, int32_t p_y) {
	x = p_x;
	y = p_y;
}

int32_t GridAgent::get_x() const {
	return x;
}

int32_t GridAgent::get_y() const {
	return y;
}

GridWorldEnvironment::GridWorldEnvironment() :
		agent(std::make_unique<GridAgent>()) {
}

Status GridWorldEnvironment::initialize(std::map<AgentId, InteractionDefinition> &r_agents) {
	if (initialized) {
		return Status::error(StatusCode::FAILED_PRECONDITION, "grid world is already initialized");
	}

	InteractionDefinition agent_definition;
	Status result = agent->define(agent_definition);
	if (!result.is_ok()) {
		return result;
	}
	result = agent_definition.observation_space.check_definition();
	if (!result.is_ok()) {
		return result;
	}
	result = agent_definition.action_space.check_definition();
	if (!result.is_ok()) {
		return result;
	}

	definition = agent_definition;
	initialized = true;
	r_agents = { { AGENT_ID, std::move(agent_definition) } };
	return Status::ok();
}

Status GridWorldEnvironment::reset(const ResetSettings &p_settings, std::map<AgentId, InitialAgentState> &r_states) {
	if (!initialized) {
		return Status::error(StatusCode::FAILED_PRECONDITION, "reset called before initialize");
	}

	// An unset seed leaves the generator alone, so the stream continues.
	if (p_settings.seed.has_value()) {
		rng.seed(static_cast<uint32_t>(*p_settings.seed));
	}

	// Any cell except the goal; indices past the goal shift up by one to skip it.
	const int32_t goal_index = GridAgent::GOAL_Y * GridAgent::SIZE + GridAgent::GOAL_X;
	std::uniform_int_distribution<int32_t> cells(0, GridAgent::SIZE * GridAgent::SIZE - 2);
	int32_t index = cells(rng);
	if (index >= goal_index) {
		index++;
	}
	agent->place(index % GridAgent::SIZE, index / GridAgent::SIZE);

	InitialAgentState initial;
	const Status result = agent->observe(initial.observation);
	if (!result.is_ok()) {
		return result;
	}

	has_episode = true;
	agent_done = false;
	r_states = { { AGENT_ID, std::move(initial) } };
	return Status::ok();
}

Status GridWorldEnvironment::step(const std::map<AgentId, Point> &p_actions, std::map<AgentId, AgentState> &r_states) {
	if (!has_episode) {
		return Status::error(StatusCode::FAILED_PRECONDITION, "step called before reset");
	}

	// Check every action before applying any, so a rejected step changes nothing.
	for (const auto &[agent_id, action] : p_actions) {
		if (agent_id != AGENT_ID) {
			return Status::error(StatusCode::NOT_FOUND, "unknown agent '" + agent_id + "'");
		}
		if (agent_done) {
			return Status::error(StatusCode::FAILED_PRECONDITION,
					"agent '" + agent_id + "' already finished; reset first");
		}
		const Status valid = definition.action_space.validate(action);
		if (!valid.is_ok()) {
			return valid;
		}
	}

	// A finished agent is absent from p_actions and is not reported.
	std::map<AgentId, AgentState> states;
	for (const auto &[agent_id, action] : p_actions) {
		Status result = agent->act(action);
		if (!result.is_ok()) {
			return result;
		}

		AgentState state;
		result = agent->observe(state.observation);
		if (!result.is_ok()) {
			return result;
		}
		state.terminated = agent->is_at_goal();
		state.reward = state.terminated ? GOAL_REWARD : STEP_REWARD;
		states.emplace(agent_id, std::move(state));
	}

	if (!states.empty() && states.begin()->second.terminated) {
		agent_done = true;
	}
	r_states = std::move(states);
	return Status::ok();
}

Status GridWorldEnvironment::get_agents(std::vector<Agent *> &r_agents) const {
	if (!initialized) {
		return Status::error(StatusCode::FAILED_PRECONDITION, "get_agents called before initialize");
	}

	r_agents = { agent.get() };
	return Status::ok();
}

} // namespace schola::testing
