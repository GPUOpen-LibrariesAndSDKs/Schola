// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "core/common/status.h"
#include "core/environment/agent.h"
#include "core/environment/interaction_definition.h"
#include "core/spaces/point.h"

#include <cstdint>
#include <vector>

namespace schola::testing {

// Test double for Agent: public fields script what it returns and record what it receives.
class FakeAgent final : public Agent {
public:
	InteractionDefinition definition; // Returned by define().
	Point observation; // Copied into r_observation by observe().
	std::vector<Point> actions; // Every action received, in order.
	std::vector<Point> received_observations; // The point given to every observe(), before it is filled.
	// Returned by every call while set, to test error paths.
	Status define_result = Status::ok();
	Status observe_result = Status::ok();
	Status act_result = Status::ok();

	Status define(InteractionDefinition &r_definition) override;
	Status observe(Point &r_observation) override;
	Status act(const Point &p_action) override;
};

} // namespace schola::testing
