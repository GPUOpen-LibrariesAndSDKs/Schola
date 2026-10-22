// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "core/policy/policy.h"

#include <string>

namespace schola {

Status Policy::create_initial_state(std::unique_ptr<PolicyState> &r_state) const {
	r_state = std::make_unique<EmptyPolicyState>();
	return Status::ok();
}

Status Policy::batched_think(const std::vector<const Point *> &p_observations, const std::vector<PolicyState *> &p_states, const std::vector<Point *> &r_actions) {
	const size_t count = p_observations.size();
	if (p_states.size() != count || r_actions.size() != count) {
		return Status::error(StatusCode::INVALID_ARGUMENT,
				"batched_think: got " + std::to_string(count) + " observations, " + std::to_string(p_states.size()) +
						" states, and " + std::to_string(r_actions.size()) + " actions");
	}
	// Checks every entry first, so bad input never leaves some states advanced.
	for (size_t i = 0; i < count; ++i) {
		if (p_observations[i] == nullptr || p_states[i] == nullptr || r_actions[i] == nullptr) {
			return Status::error(StatusCode::INVALID_ARGUMENT, "batched_think: entry " + std::to_string(i) + " is null");
		}
	}
	for (size_t i = 0; i < count; ++i) {
		Status status = think(*p_observations[i], *p_states[i], *r_actions[i]);
		if (!status.is_ok()) {
			return Status::error(status.get_code(), "batched_think: entry " + std::to_string(i) + ": " + status.get_message());
		}
	}
	return Status::ok();
}

} // namespace schola
