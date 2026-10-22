// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "core/stepper/simple_stepper.h"

#include "core/environment/agent.h"
#include "core/environment/interaction_definition.h"
#include "core/policy/policy.h"

#include <algorithm>
#include <string>

namespace schola {

namespace {

bool is_valid_index(int32_t p_index, size_t p_count) {
	return p_index >= 0 && static_cast<size_t>(p_index) < p_count;
}

Status index_error(const char *p_function, int32_t p_index) {
	return Status::error(StatusCode::NOT_FOUND, std::string(p_function) + ": no agent at index " + std::to_string(p_index));
}

} // namespace

Status SimpleStepper::init(const std::vector<Agent *> &p_agents, Policy *p_policy) {
	entries.clear();
	policy = p_policy;
	if (policy == nullptr) {
		return Status::error(StatusCode::INVALID_ARGUMENT, "SimpleStepper::init: policy is null");
	}
	for (Agent *agent : p_agents) {
		int32_t index = -1;
		Status status = add_agent(agent, index);
		if (!status.is_ok()) {
			entries.clear();
			return status;
		}
	}
	return Status::ok();
}

Status SimpleStepper::add_agent(Agent *p_agent, int32_t &r_index) {
	if (policy == nullptr) {
		return Status::error(StatusCode::FAILED_PRECONDITION, "SimpleStepper::add_agent: call init() first");
	}
	if (p_agent == nullptr) {
		return Status::error(StatusCode::INVALID_ARGUMENT, "SimpleStepper::add_agent: agent is null");
	}

	InteractionDefinition definition;
	Status status = p_agent->define(definition);
	if (!status.is_ok()) {
		return Status::error(status.get_code(), "SimpleStepper::add_agent: define: " + status.get_message());
	}
	if (entries.empty()) {
		observation_space = definition.observation_space;
		action_space = definition.action_space;
		// Points shaped for the previous spaces must not be reused.
		observations.clear();
		actions.clear();
	} else if (definition.observation_space != observation_space || definition.action_space != action_space) {
		return Status::error(StatusCode::INCOMPATIBLE, "SimpleStepper::add_agent: agent spaces differ from the other agents sharing this policy");
	}

	std::unique_ptr<PolicyState> state;
	status = policy->create_initial_state(state);
	if (!status.is_ok()) {
		return Status::error(status.get_code(), "SimpleStepper::add_agent: create_initial_state: " + status.get_message());
	}
	if (state == nullptr) {
		return Status::error(StatusCode::RUNTIME_FAILURE, "SimpleStepper::add_agent: policy created a null state");
	}

	entries.push_back({ p_agent, std::move(state) });
	r_index = static_cast<int32_t>(entries.size() - 1);
	return Status::ok();
}

Status SimpleStepper::remove_agent_at(int32_t p_index) {
	if (!is_valid_index(p_index, entries.size())) {
		return index_error("SimpleStepper::remove_agent_at", p_index);
	}
	entries.erase(entries.begin() + p_index);
	return Status::ok();
}

int32_t SimpleStepper::remove_agent(const Agent *p_agent) {
	const size_t before = entries.size();
	entries.erase(std::remove_if(entries.begin(), entries.end(), [p_agent](const Entry &p_entry) { return p_entry.agent == p_agent; }), entries.end());
	return static_cast<int32_t>(before - entries.size());
}

int32_t SimpleStepper::find_agent(const Agent *p_agent) const {
	for (size_t i = 0; i < entries.size(); ++i) {
		if (entries[i].agent == p_agent) {
			return static_cast<int32_t>(i);
		}
	}
	return -1;
}

int32_t SimpleStepper::get_agent_count() const {
	return static_cast<int32_t>(entries.size());
}

Agent *SimpleStepper::get_agent(int32_t p_index) const {
	return is_valid_index(p_index, entries.size()) ? entries[p_index].agent : nullptr;
}

PolicyState *SimpleStepper::get_state(int32_t p_index) const {
	return is_valid_index(p_index, entries.size()) ? entries[p_index].state.get() : nullptr;
}

Status SimpleStepper::reset_state(int32_t p_index) {
	if (!is_valid_index(p_index, entries.size())) {
		return index_error("SimpleStepper::reset_state", p_index);
	}
	entries[p_index].state->reset();
	return Status::ok();
}

void SimpleStepper::reset_states() {
	for (Entry &entry : entries) {
		entry.state->reset();
	}
}

Status SimpleStepper::step() {
	if (policy == nullptr) {
		return Status::error(StatusCode::FAILED_PRECONDITION, "SimpleStepper::step: call init() first");
	}
	const size_t count = entries.size();
	if (count == 0) {
		return Status::ok(); // Nothing to step, e.g. between enemy waves.
	}

	// Points are created only when the agent count changes; agents and the policy fill them in place.
	// The check matters: make_point() runs before resize() is called, even when the size is unchanged.
	if (observations.size() != count) {
		observations.resize(count, observation_space.make_point());
		actions.resize(count, action_space.make_point());
		observation_ptrs.resize(count);
		state_ptrs.resize(count);
		action_ptrs.resize(count);
	}

	for (size_t i = 0; i < count; ++i) {
		Status status = entries[i].agent->observe(observations[i]);
		if (!status.is_ok()) {
			return Status::error(status.get_code(), "SimpleStepper::step: agent " + std::to_string(i) + ": observe: " + status.get_message());
		}
		observation_ptrs[i] = &observations[i];
		state_ptrs[i] = entries[i].state.get();
		action_ptrs[i] = &actions[i];
	}

	Status status = policy->batched_think(observation_ptrs, state_ptrs, action_ptrs);
	if (!status.is_ok()) {
		return Status::error(status.get_code(), "SimpleStepper::step: " + status.get_message());
	}

	// Every agent acts even if one fails.
	Status first_error = Status::ok();
	for (size_t i = 0; i < count; ++i) {
		Status act_status = entries[i].agent->act(actions[i]);
		if (!act_status.is_ok() && first_error.is_ok()) {
			first_error = Status::error(act_status.get_code(), "SimpleStepper::step: agent " + std::to_string(i) + ": act: " + act_status.get_message());
		}
	}
	return first_error;
}

} // namespace schola
