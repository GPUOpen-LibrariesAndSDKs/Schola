// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "core/common/status.h"
#include "core/policy/policy.h"
#include "core/policy/policy_state.h"
#include "core/spaces/point.h"

#include <vector>

namespace schola::testing {

// Test double for Policy: public fields script what it returns and record what it receives.
class FakePolicy final : public Policy {
public:
	Point action; // Copied into r_action by every think().
	std::vector<Point> observations; // Every observation received, in order.
	std::vector<const PolicyState *> states; // State of every think() call, in order.
	Status next_error = Status::ok(); // Returned once by the next call, to test error paths.

	Status init(const InteractionDefinition &p_definition) override;
	Status think(const Point &p_observation, PolicyState &p_state, Point &r_action) override;
};

} // namespace schola::testing
