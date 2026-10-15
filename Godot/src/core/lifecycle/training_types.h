// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "core/common/types.h"

#if !__has_include("core/environment/environment.h")
#error "US4 requires the shared US1 environment contract"
#endif

#include "core/environment/environment.h"

#include <map>
#include <vector>

namespace schola {

enum class AutoResetMode {
	SAME_STEP,
	NEXT_STEP,
	DISABLED,
};

enum class LifecycleState {
	UNINITIALIZED,
	ACTIVE,
	COMPLETE,
	RESET_PENDING,
	FAULTED,
	CLOSED,
};

enum class CoordinatorState {
	CONFIGURING,
	READY,
	RUNNING,
	FAULTED,
	CLOSED,
};

struct EnvironmentDefinition {
	std::map<AgentId, InteractionDefinition> agents;
};

struct EnvironmentState {
	std::map<AgentId, AgentState> agents;
};

struct InitialEnvironmentState {
	std::map<AgentId, InitialAgentState> agents;
};

struct InitialState {
	// Contains only environments reset during this operation.
	std::map<EnvironmentId, InitialEnvironmentState> environments;
};

struct TrainingDefinition {
	// Position in this vector is the EnvironmentId.
	std::vector<EnvironmentDefinition> environments;
};

struct TrainingState {
	// One state per environment, ordered by EnvironmentId.
	std::vector<EnvironmentState> environments;
};

struct StepRequest {
	// One action map per environment, ordered by EnvironmentId.
	// A resetting NEXT_STEP environment may have an empty map.
	std::vector<std::map<AgentId, Point>> actions;
};

struct StepResult {
	TrainingState state;
	// Contains only environments reset by SAME_STEP auto-reset.
	InitialState initial_state;
};

struct ResetRequest {
	// Optional settings keyed by environment. Explicit reset resets every
	// environment; an omitted entry means that environment uses default settings.
	std::map<EnvironmentId, ResetSettings> environments;
};

} // namespace schola
