// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "core/connector/connector.h"

#include <string>

namespace schola {

namespace {

Status not_running(const char *p_method) {
	return Status::error(StatusCode::FAILED_PRECONDITION,
			std::string(p_method) + " was called with no training session running; call ConnectorLoop::start() first.");
}

} // namespace

Status ConnectorLoop::start(Connector &p_connector, TrainingBackend &p_backend, std::chrono::milliseconds p_start_timeout) {
	if (is_running()) {
		return Status::error(StatusCode::FAILED_PRECONDITION,
				"ConnectorLoop::start() was called while a session is already running; call stop() first.");
	}
	connector = &p_connector;
	backend = &p_backend;

	Status status = connector->open();
	if (!status.is_ok()) {
		return end_session(status);
	}

	StartRequest request;
	status = connector->wait_for_start(request, p_start_timeout);
	if (!status.is_ok()) {
		return end_session(status);
	}

	status = backend->set_autoreset_mode(request.autoreset_mode);
	if (!status.is_ok()) {
		return end_session(status);
	}
	status = connector->reply_start(Status::ok());
	if (!status.is_ok()) {
		return end_session(status);
	}

	TrainingDefinition definition;
	status = backend->define(definition);
	if (!status.is_ok()) {
		return end_session(status);
	}
	status = connector->publish_definition(definition);
	if (!status.is_ok()) {
		return end_session(status);
	}

	last_update = std::chrono::steady_clock::now();
	return Status::ok();
}

Status ConnectorLoop::tick(std::chrono::milliseconds p_update_timeout) {
	if (!is_running()) {
		return not_running("ConnectorLoop::tick()");
	}
	ConnectorUpdate update;
	Status status = connector->wait_for_update(update, p_update_timeout);
	if (status.is_ok()) {
		return run_update(update);
	}
	if (status.get_code() == StatusCode::CLOSED) {
		// Python is already gone, so there is no pending call to answer.
		stop();
		return status;
	}
	return end_session(status);
}

Status ConnectorLoop::poll_tick(std::chrono::milliseconds p_update_timeout) {
	if (!is_running()) {
		return not_running("ConnectorLoop::poll_tick()");
	}
	ConnectorUpdate update;
	Status status = connector->poll_update(update);
	if (status.is_ok()) {
		return run_update(update);
	}
	if (status.get_code() == StatusCode::NOT_FOUND) {
		if (p_update_timeout.count() > 0 && std::chrono::steady_clock::now() - last_update >= p_update_timeout) {
			return end_session(Status::error(StatusCode::TIMEOUT,
					"No message from Python for " + std::to_string(p_update_timeout.count()) +
							" ms; the training process may have stopped. Increase the step timeout if steps are expected to take longer."));
		}
		return status;
	}
	if (status.get_code() == StatusCode::CLOSED) {
		stop();
		return status;
	}
	return end_session(status);
}

void ConnectorLoop::stop() {
	if (backend != nullptr) {
		// The session is ending either way; a close failure has nowhere left to go.
		Status ignored = backend->close();
		(void)ignored;
		backend = nullptr;
	}
	if (connector != nullptr) {
		connector->close();
		connector = nullptr;
	}
}

bool ConnectorLoop::is_running() const {
	return connector != nullptr;
}

Status ConnectorLoop::run_update(const ConnectorUpdate &p_update) {
	switch (p_update.kind) {
		case UpdateKind::RESET: {
			InitialState initial_state;
			Status status = backend->reset(p_update.reset, initial_state);
			if (!status.is_ok()) {
				return end_session(status);
			}
			status = connector->reply_reset(initial_state);
			if (!status.is_ok()) {
				return end_session(status);
			}
			break;
		}
		case UpdateKind::STEP: {
			StepResult result;
			Status status = backend->step(p_update.step, result);
			if (!status.is_ok()) {
				return end_session(status);
			}
			status = connector->reply_step(result);
			if (!status.is_ok()) {
				return end_session(status);
			}
			break;
		}
		case UpdateKind::CLOSE:
			stop();
			return Status::error(StatusCode::CLOSED, "Python closed the training session.");
	}
	last_update = std::chrono::steady_clock::now();
	return Status::ok();
}

Status ConnectorLoop::end_session(const Status &p_error) {
	connector->fail(p_error);
	stop();
	return p_error;
}

} // namespace schola
