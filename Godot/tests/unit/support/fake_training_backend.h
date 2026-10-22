// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "core/common/status.h"
#include "core/lifecycle/training_backend.h"

#include <optional>
#include <string>
#include <vector>

namespace schola::testing {

// Test double for TrainingBackend: public fields script what each call returns and record what it
// received. Like the real backend, outputs are left unchanged when a call fails.
class FakeTrainingBackend final : public TrainingBackend {
public:
	// Every call, in order, appended to this shared log when set.
	std::vector<std::string> *log = nullptr;

	TrainingDefinition definition; // Returned by define().
	InitialState initial_state; // Returned by reset().
	StepResult step_result; // Returned by step().

	Status mode_status = Status::ok();
	Status define_status = Status::ok();
	Status reset_status = Status::ok();
	Status step_status = Status::ok();

	std::optional<AutoResetMode> mode; // Set by set_autoreset_mode().
	std::vector<ResetRequest> resets; // Every reset() request, in order.
	std::vector<StepRequest> steps; // Every step() request, in order.
	int close_count = 0;

	Status set_autoreset_mode(AutoResetMode p_mode) override {
		record("set_autoreset_mode");
		if (mode_status.is_ok()) {
			mode = p_mode;
		}
		return mode_status;
	}

	Status define(TrainingDefinition &r_definition) override {
		record("define");
		if (define_status.is_ok()) {
			r_definition = definition;
		}
		return define_status;
	}

	Status reset(const ResetRequest &p_request, InitialState &r_initial_state) override {
		record("reset");
		resets.push_back(p_request);
		if (reset_status.is_ok()) {
			r_initial_state = initial_state;
		}
		return reset_status;
	}

	Status step(const StepRequest &p_request, StepResult &r_result) override {
		record("step");
		steps.push_back(p_request);
		if (step_status.is_ok()) {
			r_result = step_result;
		}
		return step_status;
	}

	Status close() override {
		record("backend.close");
		++close_count;
		return Status::ok();
	}

private:
	void record(const char *p_call) {
		if (log != nullptr) {
			log->push_back(p_call);
		}
	}
};

} // namespace schola::testing
