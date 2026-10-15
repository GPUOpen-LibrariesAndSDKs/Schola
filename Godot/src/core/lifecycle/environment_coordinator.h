// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "core/lifecycle/environment_lifecycle.h"

#include <memory>
#include <vector>

namespace schola {

// Coordinates a fixed, ordered set of lifecycle instances and owns them. Python
// owns space and point validation. The coordinator performs only the checks
// needed to route operations safely and enforce its lifecycle. If an operation
// fails after an environment mutates, the entire coordinator transitions to
// FAULTED and the failure is returned to the connector for reporting to Python.
// Output parameters remain unchanged whenever an operation returns a non-OK
// Status. A faulted coordinator rejects every operation except close(); the MVP
// does not attempt partial rollback or recovery.
class EnvironmentCoordinator {
public:
	EnvironmentCoordinator() = default;

	EnvironmentCoordinator(const EnvironmentCoordinator &) = delete;
	EnvironmentCoordinator &operator=(const EnvironmentCoordinator &) = delete;

	CoordinatorState get_state() const;

	// Each lifecycle ID must equal its position in the vector. Registration is
	// accepted only while CONFIGURING and fixed after define() succeeds. Ownership
	// transfers to the coordinator only when registration succeeds; validation
	// failure leaves both the coordinator and p_lifecycles unchanged.
	Status set_lifecycles(std::vector<std::unique_ptr<EnvironmentLifecycle>> &&p_lifecycles);

	// May be called exactly once while CONFIGURING, before define(), matching
	// StartGymConnector's position in the existing Python protocol sequence.
	// Invalid input leaves the coordinator CONFIGURING.
	Status set_autoreset_mode(AutoResetMode p_mode);

	// Requires CONFIGURING and an autoreset mode. Initializes every lifecycle and
	// transitions to READY, returning definitions in EnvironmentId order. The
	// output is changed only when every lifecycle initializes successfully. If any
	// lifecycle initialization fails after initialization begins, the coordinator
	// transitions to FAULTED because earlier lifecycles may already have changed.
	Status define(TrainingDefinition &r_definition);

	// Explicit reset may interrupt ACTIVE environments and resets every registered
	// environment. An omitted request entry supplies default ResetSettings,
	// matching a Python reset request with no seed or options. Requires READY or
	// RUNNING and transitions to RUNNING on success. Once any environment reset is
	// attempted, a failure transitions the coordinator to FAULTED.
	Status reset(const ResetRequest &p_request, InitialState &r_initial_state);

	// Routes actions without validating their contents. A completed environment
	// under DISABLED mode retains its final state and is not stepped. SAME_STEP
	// preserves terminal state and returns reset observations separately. Under
	// NEXT_STEP, a completed environment resets instead of consuming its action
	// map and returns its initial observation with zero reward and false completion
	// flags. Physical effects may appear in a later observation according to
	// Godot's normal frame timing. Requires RUNNING. Once any environment step or
	// autoreset is attempted, a failure transitions the coordinator to FAULTED.
	Status step(const StepRequest &p_request, StepResult &r_result);

	// Transitions to CLOSED from any state, closes every lifecycle, and rejects
	// future operations. Repeated calls are successful and have no additional
	// effect.
	Status close();

private:
	std::vector<std::unique_ptr<EnvironmentLifecycle>> environments;
	AutoResetMode autoreset_mode = AutoResetMode::DISABLED;
	CoordinatorState state = CoordinatorState::CONFIGURING;
	bool mode_set = false;
};

} // namespace schola
