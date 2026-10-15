// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

// Dependency on US4
#if !__has_include("core/lifecycle/training_backend.h")
#error "US3 needs TrainingBackend from core/lifecycle/training_backend.h"
#endif

#include "core/lifecycle/training_backend.h"

#include <godot_cpp/classes/object.hpp>

namespace schola {

/**
 * Implements TrainingBackend by calling the bound methods of the
 * ScholaTrainingHost node, using Variant arguments. That node is proposed for
 * US4 in Godot/docs/api_overview.md (section 4.5); no US4 pull request
 * defines it yet.
 *
 * The training extension never calls the runtime extension's C++ classes
 * directly, so a shipped game can drop the training extension entirely. Every
 * method must be called on the Godot main thread.
 */
class HostTrainingBackend final : public TrainingBackend {
public:
	// p_host is not owned; the host node outlives this backend.
	explicit HostTrainingBackend(godot::Object *p_host);

	Status set_autoreset_mode(AutoResetMode p_mode) override;
	Status define(TrainingDefinition &r_definition) override;
	Status reset(const ResetRequest &p_request, InitialState &r_initial_state) override;
	Status step(const StepRequest &p_request, StepResult &r_result) override;
	Status close() override;

private:
	godot::Object *host = nullptr; // Not owned.
};

} // namespace schola
