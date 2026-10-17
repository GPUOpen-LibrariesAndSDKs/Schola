// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "core/common/status.h"

#include <cstdint>
#include <vector>

namespace schola {

class Agent;
class Policy;
class PolicyState;

// Blocking observe -> think -> act loop on the main thread.
// Entries work like an array: kept in add order, and removal shifts later entries down.
// Owns one state per entry; does not own agents or the policy.
// An invalid index returns NOT_FOUND or nullptr.
class SimpleStepper {
public:
	// Replaces all entries, as if each agent were passed to add_agent(). p_policy must be initialized.
	Status init(const std::vector<Agent *> &p_agents, Policy *p_policy);
	// Appends an entry with a new state. The same agent may be added again; each entry has its own state.
	Status add_agent(Agent *p_agent, int32_t &r_index);
	Status remove_agent_at(int32_t p_index);
	int32_t remove_agent(const Agent *p_agent); // Removes every entry for p_agent; returns how many.
	int32_t find_agent(const Agent *p_agent) const; // First entry for p_agent, or -1.
	int32_t get_agent_count() const;
	Agent *get_agent(int32_t p_index) const;
	PolicyState *get_state(int32_t p_index) const;
	Status reset_state(int32_t p_index);
	void reset_states();

	Status step(); // Observe all, one batched_think, act all.
};

} // namespace schola
