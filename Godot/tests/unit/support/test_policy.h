// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "core/common/status.h"
#include "core/policy/policy.h"
#include "core/policy/policy_state.h"
#include "core/spaces/point.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace schola::testing {

// Counts the think() calls that advanced it, to check that each entry keeps its own state.
class CounterState final : public PolicyState {
public:
	int32_t count = 0;

	void reset() override { count = 0; }
	Status copy_from(const PolicyState &p_other) override;
	std::string to_string() const override { return "CounterState(" + std::to_string(count) + ")"; }
};

// Test double for Policy: public fields script what it returns and record what it receives.
class FakePolicy final : public Policy {
public:
	Point action; // Copied into r_action by every successful think().
	bool stateful = false; // create_initial_state() returns a CounterState instead of an EmptyPolicyState.
	std::vector<Point> observations; // Every observation received, in order.
	std::vector<const PolicyState *> states; // State of every think() call, in order.
	int32_t fail_at_call = -1; // Zero-based think() call that returns failure; -1 never fails.
	Status failure = Status::error(StatusCode::RUNTIME_FAILURE, "FakePolicy failure");

	Status init(const InteractionDefinition &p_definition) override;
	Status create_initial_state(std::unique_ptr<PolicyState> &r_state) const override;
	Status think(const Point &p_observation, PolicyState &p_state, Point &r_action) override;
};

} // namespace schola::testing
