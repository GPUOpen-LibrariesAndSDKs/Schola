// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "core/lifecycle/environment_lifecycle.h"

#include <algorithm>
#include <utility>

namespace schola {

namespace {

template <typename Left, typename Right>
bool same_agent_ids(const Left &p_left, const Right &p_right) {
	return p_left.size() == p_right.size() &&
			std::equal(p_left.begin(), p_left.end(), p_right.begin(),
					[](const auto &p_a, const auto &p_b) { return p_a.first == p_b.first; });
}

} // namespace

EnvironmentLifecycle::EnvironmentLifecycle(
		EnvironmentId p_id,
		Environment &p_environment,
		uint64_t p_episode_step_limit) :
		id(p_id),
		environment(p_environment),
		episode_step_limit(p_episode_step_limit) {
}

EnvironmentId EnvironmentLifecycle::get_id() const {
	return id;
}

LifecycleState EnvironmentLifecycle::get_state() const {
	return state;
}

uint64_t EnvironmentLifecycle::get_episode_step_count() const {
	return episode_step_count;
}

uint64_t EnvironmentLifecycle::get_episode_count() const {
	return episode_num;
}

const EnvironmentDefinition &EnvironmentLifecycle::get_definition() const {
	return definition;
}

Status EnvironmentLifecycle::initialize() {
	if (state != LifecycleState::UNINITIALIZED) {
		return Status::error(StatusCode::FAILED_PRECONDITION,
				"environment lifecycle is already initialized");
	}

	std::map<AgentId, InteractionDefinition> agents;
	Status result = environment.initialize(agents);
	if (!result.is_ok()) {
		state = LifecycleState::FAULTED;
		return result;
	}
	if (agents.empty()) {
		state = LifecycleState::FAULTED;
		return Status::error(StatusCode::INVALID_DATA, "environment must declare at least one agent");
	}

	definition = EnvironmentDefinition{ agents };
	state = LifecycleState::RESET_PENDING;
	return Status::ok();
}

Status EnvironmentLifecycle::reset(const ResetSettings &p_settings, InitialEnvironmentState &r_initial_state) {
	if (state != LifecycleState::RESET_PENDING && state != LifecycleState::ACTIVE && state != LifecycleState::COMPLETE) {
		return Status::error(StatusCode::FAILED_PRECONDITION,
				"environment is not in a resettable state");
	}

	std::map<AgentId, InitialAgentState> agents;
	Status result = environment.reset(p_settings, agents);

	if (!result.is_ok()) {
		state = LifecycleState::FAULTED;
		return result;
	}
	if (!same_agent_ids(agents, definition.agents)) {
		state = LifecycleState::FAULTED;
		return Status::error(StatusCode::INVALID_DATA, "reset must return exactly the declared agents");
	}

	r_initial_state.agents = std::move(agents);
	last_state.agents.clear();
	episode_step_count = 0;
	episode_num++;
	state = LifecycleState::ACTIVE;
	return Status::ok();
}

Status EnvironmentLifecycle::step(const std::map<AgentId, Point> &p_actions, EnvironmentState &r_state) {
	if (state != LifecycleState::ACTIVE) {
		return Status::error(StatusCode::FAILED_PRECONDITION,
				"environment is not in an active state");
	}
	size_t unfinished_count = 0;
	for (const auto &[agent_id, agent_definition] : definition.agents) {
		const auto previous = last_state.agents.find(agent_id);
		if (previous != last_state.agents.end() && (previous->second.terminated || previous->second.truncated)) {
			continue;
		}
		++unfinished_count;
		if (p_actions.find(agent_id) == p_actions.end()) {
			return Status::error(StatusCode::INVALID_ARGUMENT, "missing action for unfinished agent " + agent_id);
		}
	}
	if (p_actions.size() != unfinished_count) {
		return Status::error(StatusCode::INVALID_ARGUMENT, "actions must contain only unfinished declared agents");
	}

	std::map<AgentId, AgentState> step_states;
	Status result = environment.step(p_actions, step_states);
	if (!result.is_ok()) {
		state = LifecycleState::FAULTED;
		return result;
	}
	if (!same_agent_ids(step_states, p_actions)) {
		state = LifecycleState::FAULTED;
		return Status::error(StatusCode::INVALID_DATA, "step must return exactly the agents that acted");
	}

	episode_step_count++;
	if (episode_step_limit != 0 && episode_step_count >= episode_step_limit) {
		for (auto &[agent_id, agent_state] : step_states) {
			if (!agent_state.terminated && !agent_state.truncated) {
				agent_state.truncated = true;
			}
		}
	}

	for (const auto &[agent_id, agent_state] : step_states) {
		last_state.agents[agent_id] = agent_state;
	}

	const bool all_agents_done = std::all_of(last_state.agents.begin(), last_state.agents.end(),
			[](const auto &p_entry) { return p_entry.second.terminated || p_entry.second.truncated; });

	state = all_agents_done ? LifecycleState::COMPLETE : LifecycleState::ACTIVE;
	r_state = last_state;
	return Status::ok();
}

Status EnvironmentLifecycle::close() {
	last_state.agents.clear();
	episode_step_count = 0;
	state = LifecycleState::CLOSED;
	return Status::ok();
}

} // namespace schola
