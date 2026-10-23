// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "core/lifecycle/environment_coordinator.h"

#include <utility>

namespace schola {

namespace {

Status environment_error(EnvironmentId p_id, const Status &p_status) {
	return Status::error(p_status.get_code(),
			"environment " + std::to_string(p_id) + ": " + p_status.get_message());
}

} // namespace

CoordinatorState EnvironmentCoordinator::get_state() const {
	return state;
}

Status EnvironmentCoordinator::set_lifecycles(std::vector<std::unique_ptr<EnvironmentLifecycle>> &&p_lifecycles) {
	if (state != CoordinatorState::CONFIGURING) {
		return Status::error(StatusCode::FAILED_PRECONDITION, "lifecycles can only be registered while configuring");
	}
	if (p_lifecycles.empty()) {
		return Status::error(StatusCode::INVALID_ARGUMENT, "at least one lifecycle is required");
	}
	for (size_t index = 0; index < p_lifecycles.size(); ++index) {
		if (!p_lifecycles[index] || p_lifecycles[index]->get_id() != static_cast<EnvironmentId>(index)) {
			return Status::error(StatusCode::INVALID_ARGUMENT, "each lifecycle must be non-null and its ID must equal its position");
		}
		if (p_lifecycles[index]->get_state() != LifecycleState::UNINITIALIZED) {
			return Status::error(StatusCode::FAILED_PRECONDITION, "registered lifecycles must be uninitialized");
		}
	}
	environments = std::move(p_lifecycles);
	return Status::ok();
}

Status EnvironmentCoordinator::set_autoreset_mode(AutoResetMode p_mode) {
	if (state != CoordinatorState::CONFIGURING || mode_set) {
		return Status::error(StatusCode::FAILED_PRECONDITION, "autoreset mode must be configured exactly once before define");
	}
	if (p_mode != AutoResetMode::DISABLED && p_mode != AutoResetMode::SAME_STEP && p_mode != AutoResetMode::NEXT_STEP) {
		return Status::error(StatusCode::INVALID_ARGUMENT, "unknown autoreset mode");
	}
	autoreset_mode = p_mode;
	mode_set = true;
	return Status::ok();
}

Status EnvironmentCoordinator::define(TrainingDefinition &r_definition) {
	if (state != CoordinatorState::CONFIGURING || !mode_set || environments.empty()) {
		return Status::error(StatusCode::FAILED_PRECONDITION, "define requires registered lifecycles and an autoreset mode");
	}
	TrainingDefinition definition;
	for (const auto &environment : environments) {
		Status result = environment->initialize();
		if (!result.is_ok()) {
			state = CoordinatorState::FAULTED;
			return environment_error(environment->get_id(), result);
		}
		definition.environments.push_back(environment->get_definition());
	}
	r_definition = std::move(definition);
	last_state.environments.resize(environments.size());
	state = CoordinatorState::READY;
	return Status::ok();
}

Status EnvironmentCoordinator::reset(const ResetRequest &p_request, InitialState &r_initial_state) {
	if (state != CoordinatorState::READY && state != CoordinatorState::RUNNING) {
		return Status::error(StatusCode::FAILED_PRECONDITION, "reset requires a defined, open coordinator");
	}
	for (const auto &[id, settings] : p_request.environments) {
		if (id < 0 || static_cast<size_t>(id) >= environments.size()) {
			return Status::error(StatusCode::INVALID_ARGUMENT, "reset references an unknown environment ID");
		}
	}
	InitialState initial_state;
	for (const auto &environment : environments) {
		const EnvironmentId id = environment->get_id();
		const auto settings = p_request.environments.find(id);
		InitialEnvironmentState initial;
		Status result = environment->reset(settings == p_request.environments.end() ? ResetSettings{} : settings->second, initial);
		if (!result.is_ok()) {
			state = CoordinatorState::FAULTED;
			return environment_error(id, result);
		}
		initial_state.environments.emplace(id, std::move(initial));
	}
	r_initial_state = std::move(initial_state);
	last_state.environments.assign(environments.size(), EnvironmentState{});
	state = CoordinatorState::RUNNING;
	return Status::ok();
}

Status EnvironmentCoordinator::step(const StepRequest &p_request, StepResult &r_result) {
	if (state != CoordinatorState::RUNNING) {
		return Status::error(StatusCode::FAILED_PRECONDITION, "step requires a running coordinator");
	}
	if (p_request.actions.size() != environments.size()) {
		return Status::error(StatusCode::INVALID_ARGUMENT, "step requires one action map per environment");
	}
	StepResult batch;
	batch.state = last_state;
	for (size_t index = 0; index < environments.size(); ++index) {
		auto &environment = environments[index];
		const EnvironmentId id = environment->get_id();
		if (environment->get_state() == LifecycleState::COMPLETE) {
			if (autoreset_mode == AutoResetMode::DISABLED) {
				continue;
			}
			// NEXT_STEP consumes this call by resetting, regardless of its actions.
			InitialEnvironmentState initial;
			Status result = environment->reset({}, initial);
			if (!result.is_ok()) {
				state = CoordinatorState::FAULTED;
				return environment_error(id, result);
			}
			EnvironmentState reset_state;
			for (auto &[agent_id, agent] : initial.agents) {
				reset_state.agents.emplace(agent_id, AgentState{ std::move(agent.observation), 0.0f, false, false, std::move(agent.info) });
			}
			batch.state.environments[index] = std::move(reset_state);
			continue;
		}
		Status result = environment->step(p_request.actions[index], batch.state.environments[index]);
		if (!result.is_ok()) {
			state = CoordinatorState::FAULTED;
			return environment_error(id, result);
		}
		if (environment->get_state() == LifecycleState::COMPLETE && autoreset_mode == AutoResetMode::SAME_STEP) {
			InitialEnvironmentState initial;
			result = environment->reset({}, initial);
			if (!result.is_ok()) {
				state = CoordinatorState::FAULTED;
				return environment_error(id, result);
			}
			batch.initial_state.environments.emplace(id, std::move(initial));
		}
	}
	last_state = batch.state;
	r_result = std::move(batch);
	return Status::ok();
}

Status EnvironmentCoordinator::close() {
	if (state == CoordinatorState::CLOSED) {
		return Status::ok();
	}
	Status result = Status::ok();
	for (const auto &environment : environments) {
		Status closed = environment->close();
		if (!closed.is_ok() && result.is_ok()) {
			result = environment_error(environment->get_id(), closed);
		}
	}
	last_state.environments.clear();
	environments.clear();
	state = CoordinatorState::CLOSED;
	return result;
}

} // namespace schola
