// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "core/common/status.h"

#include <cstdint>
#include <vector>

namespace schola {

class Agent;
class Policy;

// Blocking observe -> think -> act loop on the main thread.
class SimpleStepper {
public:
	// Non-owning. Agents must share spaces; agent i keeps index i. Calls Policy::init.
	Status init(const std::vector<Agent *> &p_agents, Policy *p_policy);
	Status step(); // Observe all, one batched_think, act all.
	void reset(int32_t p_agent); // Forwards to Policy::reset.
	void reset_all(); // Forwards to Policy::reset_all.
};

} // namespace schola
