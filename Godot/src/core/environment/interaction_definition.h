// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

// Dependency on US2
#if !__has_include("core/spaces/space.h")
#error "US1 needs the Space type from core/spaces/space.h"
#endif

#include "core/spaces/space.h"

#include <string>

namespace schola {

/**
 * What an agent publishes about itself, which is:
 *
 * - What it can see
 * - What it can be told to do
 * - Whether it shares a brain with other agents
 *
 * Space describes the shape allowed; Point (in agent.h) carries the real
 * values every step. This struct only ever holds the Space side.
 */
struct InteractionDefinition {
	// The shape observe()'s output (r_observation, a Point) must match.
	// Fixed once in define(), never changes mid-episode.
	Space observation_space;

	// The shape act()'s input (p_action, a Point) must match.
	Space action_space;

	/**
	 * The agent's grouping label, e.g.
	 *
	 * Agent "dumb_agent1" has agent_type "dumb"
	 * Agent "dumb_agent2" has agent_type "dumb"
	 * Agent "smart_agent1" has agent_type "smart"
	 *
	 * Empty means "stands alone" -- its agent ID is used instead.
	 * Essentially: agents sharing a type can share one trained policy.
	 */
	std::string agent_type;
};

} // namespace schola
