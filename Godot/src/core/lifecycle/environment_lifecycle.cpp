// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "core/lifecycle/environment_lifecycle.h"

#include <utility>

namespace schola {

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

	std::map<AgentId, AgentState> step_states;
	Status result = environment.step(p_actions, step_states);
	if (!result.is_ok()) {
		state = LifecycleState::FAULTED;
		return result;
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

	bool all_agents_done = true;
	for (auto &[agent_id, agent_state] : last_state.agents) {
		if (!agent_state.terminated && !agent_state.truncated) {
			all_agents_done = false;
			break;
		}
	}

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
