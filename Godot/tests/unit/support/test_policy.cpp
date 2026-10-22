// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "test_policy.h"

namespace schola::testing {

Status CounterState::copy_from(const PolicyState &p_other) {
	const CounterState *other = dynamic_cast<const CounterState *>(&p_other);
	if (other == nullptr) {
		return Status::error(StatusCode::INVALID_ARGUMENT, "CounterState::copy_from: expected a CounterState, got " + p_other.to_string());
	}
	count = other->count;
	return Status::ok();
}

Status FakePolicy::init([[maybe_unused]] const InteractionDefinition &p_definition) {
	return Status::ok();
}

Status FakePolicy::create_initial_state(std::unique_ptr<PolicyState> &r_state) const {
	if (!stateful) {
		return Policy::create_initial_state(r_state);
	}
	r_state = std::make_unique<CounterState>();
	return Status::ok();
}

Status FakePolicy::think(const Point &p_observation, PolicyState &p_state, Point &r_action) {
	const size_t call = observations.size();
	observations.push_back(p_observation);
	states.push_back(&p_state);
	if (fail_at_call >= 0 && call == static_cast<size_t>(fail_at_call)) {
		return failure; // Leaves the state unchanged, like a real policy.
	}
	if (CounterState *counter = dynamic_cast<CounterState *>(&p_state)) {
		++counter->count;
	}
	r_action = action;
	return Status::ok();
}

} // namespace schola::testing
