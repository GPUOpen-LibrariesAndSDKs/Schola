// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "core/common/status.h"
#include "core/policy/policy.h"
#include "core/spaces/point.h"

#include <cstdint>
#include <vector>

namespace schola::testing {

// Test double for Policy: public fields script what it returns and record what it receives.
class FakePolicy final : public Policy {
public:
	Point action; // Copied into r_action by every think().
	std::vector<int32_t> agents; // Agent index of every observation received, in order.
	std::vector<Point> observations; // Every observation received, in order.
	std::vector<int32_t> resets; // Agent index of every reset() call; reset_all() records -1.
	Status next_error = Status::ok(); // Returned once by the next call, to test error paths.

	Status init(const InteractionDefinition &p_definition, int32_t p_agent_count) override;
	Status think(int32_t p_agent, const Point &p_observation, Point &r_action) override;
	void reset(int32_t p_agent) override;
	void reset_all() override;
};

} // namespace schola::testing
