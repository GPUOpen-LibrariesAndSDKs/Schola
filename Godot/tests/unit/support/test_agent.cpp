// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "test_agent.h"

namespace schola::testing {

Status FakeAgent::define(InteractionDefinition &r_definition) {
	if (!define_result.is_ok()) {
		return define_result;
	}
	r_definition = definition;
	return Status::ok();
}

Status FakeAgent::observe(Point &r_observation) {
	received_observations.push_back(r_observation);
	if (!observe_result.is_ok()) {
		return observe_result;
	}
	r_observation = observation;
	return Status::ok();
}

Status FakeAgent::act(const Point &p_action) {
	actions.push_back(p_action);
	return act_result;
}

} // namespace schola::testing
