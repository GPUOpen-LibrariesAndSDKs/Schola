// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "core/common/status.h"
#include "core/policy/policy_state.h"
#include "core/spaces/point.h"
#include "core/spaces/space.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace schola {

class Agent;
class Policy;

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

	// Observe all, one batched_think, act all. Every agent acts even if one fails; returns the first error.
	Status step();

private:
	struct Entry {
		Agent *agent = nullptr;
		std::unique_ptr<PolicyState> state;
	};

	Policy *policy = nullptr;
	std::vector<Entry> entries;
	// From the first agent's define(); entries share one policy, so they share one set of spaces.
	Space observation_space;
	Space action_space;
	// Reused every step, so stepping does not allocate once the agent count is stable.
	std::vector<Point> observations;
	std::vector<Point> actions;
	std::vector<const Point *> observation_ptrs;
	std::vector<PolicyState *> state_ptrs;
	std::vector<Point *> action_ptrs;
};

} // namespace schola
