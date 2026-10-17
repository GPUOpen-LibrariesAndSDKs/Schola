// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "core/common/status.h"
#include "core/policy/policy_state.h"
#include "core/spaces/point.h"

#include <memory>
#include <vector>

namespace schola {

struct InteractionDefinition;

// Maps an observation and a state to an action. Holds no per-agent data.
class Policy {
public:
	virtual ~Policy() = default;

	// Checks the model fits the spaces; allocates shared buffers.
	virtual Status init(const InteractionDefinition &p_definition) = 0;
	// Default creates an EmptyPolicyState.
	virtual Status create_initial_state(std::unique_ptr<PolicyState> &r_state) const;
	// Advances p_state in place; leaves it unchanged on failure. r_action comes from action_space.make_point().
	virtual Status think(const Point &p_observation, PolicyState &p_state, Point &r_action) = 0;
	// Entry i of each vector belongs together. Default calls think() per entry.
	virtual Status batched_think(const std::vector<const Point *> &p_observations, const std::vector<PolicyState *> &p_states, const std::vector<Point *> &r_actions);
	virtual bool is_recurrent() const { return false; } // Model keeps memory between calls.
	virtual bool is_busy() const { return false; } // An asynchronous inference is running.
};

} // namespace schola
