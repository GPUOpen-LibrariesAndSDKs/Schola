// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "core/common/status.h"
#include "core/lifecycle/training_types.h"

namespace schola {

// Logical lifecycle boundary used by the connector. The connector calls every
// method serially on the backend's owning thread. Godot's ordinary frame timing
// may delay the physical effects of an action until a later observation.
// Synchronous output parameters remain unchanged whenever an operation returns
// a non-OK Status. The Godot-facing training host implements this contract around
// an EnvironmentCoordinator; US3 depends only on this interface and may provide
// a proxy or fake without depending on Godot lifecycle internals.
class TrainingBackend {
public:
	virtual ~TrainingBackend() = default;

	// Called exactly once before define(), reset(), or step(), matching
	// StartGymConnector's position in the existing Python protocol sequence.
	virtual Status set_autoreset_mode(AutoResetMode p_mode) = 0;

	// Initializes the fixed training session after the autoreset mode is set and
	// returns its definition.
	virtual Status define(TrainingDefinition &r_definition) = 0;

	// Performs an explicit reset synchronously on the owning thread.
	virtual Status reset(const ResetRequest &p_request, InitialState &r_initial_state) = 0;

	// Applies one logical training step. Point contents are passed through without
	// engine-side space validation; Python owns that validation.
	virtual Status step(const StepRequest &p_request, StepResult &r_result) = 0;

	// Rejects new operations and releases session resources.
	virtual Status close() = 0;
};

} // namespace schola
