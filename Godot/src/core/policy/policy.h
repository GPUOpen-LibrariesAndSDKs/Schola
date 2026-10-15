// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "core/common/status.h"
#include "core/spaces/point.h"

#include <cstdint>
#include <vector>

namespace schola {

struct InteractionDefinition;

class Policy {
public:
	virtual ~Policy() = default;

	// Checks the model fits the spaces; allocates memory.
	virtual Status init(const InteractionDefinition &p_definition, int32_t p_agent_count) = 0;
	// One action for p_agent; r_action comes from action_space.make_point().
	virtual Status think(int32_t p_agent, const Point &p_observation, Point &r_action) = 0;
	// Entry i belongs to agent i. Default calls think() per agent.
	virtual Status batched_think(const std::vector<const Point *> &p_observations, const std::vector<Point *> &r_actions);
	virtual void reset([[maybe_unused]] int32_t p_agent) {} // Clears one agent's memory.
	virtual void reset_all() {} // Clears every agent's memory.
	virtual bool is_recurrent() const { return false; } // Keeps memory between calls.
	virtual bool is_busy() const { return false; } // An asynchronous inference is running.
};

} // namespace schola
