// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "fake_environment.h"

namespace schola::testing {

Status FakeEnvironment::initialize(std::map<AgentId, InteractionDefinition> &r_agents) {
	initialize_calls++;

	if (initialize_status.is_ok()) {
		r_agents = definitions;
	}
	return initialize_status;
}

Status FakeEnvironment::reset(const ResetSettings &p_settings, std::map<AgentId, InitialAgentState> &r_states) {
	reset_calls++;

	last_reset_settings = p_settings;
	if (reset_status.is_ok()) {
		r_states = reset_states;
	}
	return reset_status;
}

Status FakeEnvironment::step(const std::map<AgentId, Point> &p_actions, std::map<AgentId, AgentState> &r_states) {
	step_calls++;
	last_actions = p_actions;

	if (step_status.is_ok()) {
		r_states = step_states;
	}
	return step_status;
}

Status FakeEnvironment::get_agents(std::vector<Agent *> &r_agents) const {
	get_agents_calls++;

	if (get_agents_status.is_ok()) {
		r_agents = agents;
	}
	return get_agents_status;
}

} // namespace schola::testing
