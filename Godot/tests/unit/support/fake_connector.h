// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "core/common/status.h"
#include "core/connector/connector.h"

#include <deque>
#include <optional>
#include <string>
#include <vector>

namespace schola::testing {

// Test double for Connector: Python without a network. Public fields script what Python sends and
// record what the connector was asked to send back.
class FakeConnector final : public Connector {
public:
	// Every call, in order, appended to this shared log when set.
	std::vector<std::string> *log = nullptr;

	Status open_result = Status::ok();
	// Python's StartGymConnector. Unset means Python never connects: wait_for_start() times out.
	std::optional<StartRequest> start_request = StartRequest{};
	// Returned by wait_for_start() instead of a request, e.g. an unknown auto-reset mode.
	std::optional<Status> start_error;
	// Popped by wait_for_update() and poll_update(). When empty, wait_for_update() returns TIMEOUT and
	// poll_update() returns NOT_FOUND.
	std::deque<ConnectorUpdate> updates;
	// Returned once by the next wait_for_update() or poll_update() instead of an update.
	std::optional<Status> next_update_error;

	std::optional<Status> start_reply;
	std::optional<TrainingDefinition> published_definition;
	std::vector<InitialState> reset_replies;
	std::vector<StepResult> step_replies;
	std::optional<Status> failure; // Set by fail().
	int close_count = 0;

	Status open() override {
		record("open");
		return open_result;
	}

	Status wait_for_start(StartRequest &r_request, std::chrono::milliseconds p_timeout) override {
		record("wait_for_start");
		if (start_error) {
			return *start_error;
		}
		if (!start_request) {
			return Status::error(StatusCode::TIMEOUT,
					"Python did not connect within " + std::to_string(p_timeout.count()) + " ms.");
		}
		r_request = *start_request;
		return Status::ok();
	}

	Status reply_start(const Status &p_result) override {
		record("reply_start");
		start_reply = p_result;
		return Status::ok();
	}

	Status publish_definition(const TrainingDefinition &p_definition) override {
		record("publish_definition");
		published_definition = p_definition;
		return Status::ok();
	}

	Status wait_for_update(ConnectorUpdate &r_update, std::chrono::milliseconds p_timeout) override {
		record("wait_for_update");
		if (std::optional<Status> error = take_update_error()) {
			return *error;
		}
		if (updates.empty()) {
			return Status::error(StatusCode::TIMEOUT,
					"No message from Python within " + std::to_string(p_timeout.count()) + " ms.");
		}
		r_update = pop_update();
		return Status::ok();
	}

	Status poll_update(ConnectorUpdate &r_update) override {
		record("poll_update");
		if (std::optional<Status> error = take_update_error()) {
			return *error;
		}
		if (updates.empty()) {
			return Status::error(StatusCode::NOT_FOUND, "No message from Python yet.");
		}
		r_update = pop_update();
		return Status::ok();
	}

	Status reply_reset(const InitialState &p_initial_state) override {
		record("reply_reset");
		reset_replies.push_back(p_initial_state);
		return Status::ok();
	}

	Status reply_step(const StepResult &p_result) override {
		record("reply_step");
		step_replies.push_back(p_result);
		return Status::ok();
	}

	void fail(const Status &p_error) override {
		record("fail");
		failure = p_error;
	}

	void close() override {
		record("connector.close");
		++close_count;
	}

private:
	void record(const char *p_call) {
		if (log != nullptr) {
			log->push_back(p_call);
		}
	}

	std::optional<Status> take_update_error() {
		std::optional<Status> error = next_update_error;
		next_update_error.reset();
		return error;
	}

	ConnectorUpdate pop_update() {
		ConnectorUpdate update = updates.front();
		updates.pop_front();
		return update;
	}
};

} // namespace schola::testing
