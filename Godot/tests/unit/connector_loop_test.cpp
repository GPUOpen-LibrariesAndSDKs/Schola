// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "core/connector/connector.h"
#include "core/spaces/point.h"
#include "core/spaces/space.h"
#include "support/fake_connector.h"
#include "support/fake_training_backend.h"

#include <catch_amalgamated.hpp>

#include <chrono>
#include <string>
#include <thread>
#include <vector>

using namespace schola;
using namespace schola::testing;
using namespace std::chrono_literals;

namespace {

const AgentId AGENT = "agent_0";

TrainingDefinition one_agent_definition() {
	InteractionDefinition agent;
	agent.observation_space = Space(BoxSpace{ { 0, 0, 0, 0 }, { 1, 1, 1, 1 }, { 4 } });
	agent.action_space = Space(DiscreteSpace{ 4 });

	EnvironmentDefinition environment;
	environment.agents[AGENT] = agent;

	TrainingDefinition definition;
	definition.environments.push_back(environment);
	return definition;
}

ConnectorUpdate reset_update(int32_t p_seed) {
	ConnectorUpdate update;
	update.kind = UpdateKind::RESET;
	update.reset.environments[0].seed = p_seed;
	return update;
}

ConnectorUpdate step_update(int32_t p_action) {
	ConnectorUpdate update;
	update.kind = UpdateKind::STEP;
	update.step.actions.push_back({ { AGENT, Point(DiscretePoint{ p_action }) } });
	return update;
}

ConnectorUpdate close_update() {
	ConnectorUpdate update;
	update.kind = UpdateKind::CLOSE;
	return update;
}

// One loop wired to a fake Python and a fake backend that share a call log.
struct Session {
	std::vector<std::string> log;
	FakeConnector connector;
	FakeTrainingBackend backend;
	ConnectorLoop loop;

	Session() {
		connector.log = &log;
		backend.log = &log;
		backend.definition = one_agent_definition();
	}

	Status start() {
		return loop.start(connector, backend, 50ms);
	}
};

bool logged(const std::vector<std::string> &p_log, const std::string &p_call) {
	for (const std::string &call : p_log) {
		if (call == p_call) {
			return true;
		}
	}
	return false;
}

} // namespace

TEST_CASE("start follows the Python client's call order and publishes the definition", "[connector]") {
	Session session;

	REQUIRE(session.start().is_ok());

	CHECK(session.log == std::vector<std::string>{ "open", "wait_for_start", "set_autoreset_mode", "reply_start", "define", "publish_definition" });
	REQUIRE(session.connector.start_reply.has_value());
	CHECK(session.connector.start_reply->is_ok());
	REQUIRE(session.connector.published_definition.has_value());
	REQUIRE(session.connector.published_definition->environments.size() == 1);
	CHECK(session.connector.published_definition->environments[0].agents.count(AGENT) == 1);
	CHECK(session.loop.is_running());
}

TEST_CASE("start passes Python's auto-reset mode to the backend", "[connector]") {
	Session session;
	session.connector.start_request = StartRequest{ AutoResetMode::NEXT_STEP };

	REQUIRE(session.start().is_ok());

	REQUIRE(session.backend.mode.has_value());
	CHECK(*session.backend.mode == AutoResetMode::NEXT_STEP);
}

TEST_CASE("start times out and closes when Python never connects", "[connector]") {
	Session session;
	session.connector.start_request.reset();

	Status status = session.start();

	CHECK(status.get_code() == StatusCode::TIMEOUT);
	REQUIRE(session.connector.failure.has_value());
	CHECK(session.connector.failure->get_code() == StatusCode::TIMEOUT);
	CHECK_FALSE(session.backend.mode.has_value());
	CHECK(session.connector.close_count == 1);
	CHECK_FALSE(session.loop.is_running());
}

TEST_CASE("a failed open is reported and nothing else runs", "[connector]") {
	Session session;
	session.connector.open_result = Status::error(StatusCode::RUNTIME_FAILURE, "Port 8000 is already in use.");

	Status status = session.start();

	CHECK(status.get_code() == StatusCode::RUNTIME_FAILURE);
	CHECK_FALSE(logged(session.log, "wait_for_start"));
	CHECK(session.connector.failure.has_value());
	CHECK_FALSE(session.loop.is_running());
}

TEST_CASE("an auto-reset mode the backend rejects reaches Python as an error", "[connector]") {
	Session session;
	session.backend.mode_status = Status::error(StatusCode::INCOMPATIBLE, "Only DISABLED auto-reset is supported.");

	Status status = session.start();

	CHECK(status.get_code() == StatusCode::INCOMPATIBLE);
	REQUIRE(session.connector.failure.has_value());
	CHECK(session.connector.failure->get_message() == "Only DISABLED auto-reset is supported.");
	CHECK_FALSE(session.connector.start_reply.has_value());
	CHECK_FALSE(logged(session.log, "define"));
	CHECK(session.backend.close_count == 1);
	CHECK(session.connector.close_count == 1);
}

TEST_CASE("an invalid start request from Python never reaches the backend", "[connector]") {
	Session session;
	session.connector.start_error = Status::error(StatusCode::INVALID_ARGUMENT, "Unknown autoreset_type 7.");

	CHECK(session.start().get_code() == StatusCode::INVALID_ARGUMENT);
	CHECK_FALSE(session.backend.mode.has_value());
	CHECK_FALSE(session.loop.is_running());
}

TEST_CASE("start is rejected while a session is already running", "[connector]") {
	Session session;
	REQUIRE(session.start().is_ok());

	CHECK(session.loop.start(session.connector, session.backend, 50ms).get_code() == StatusCode::FAILED_PRECONDITION);
	CHECK(session.loop.is_running());
}

TEST_CASE("tick and poll_tick are rejected before start", "[connector]") {
	ConnectorLoop loop;

	CHECK(loop.tick(10ms).get_code() == StatusCode::FAILED_PRECONDITION);
	CHECK(loop.poll_tick(10ms).get_code() == StatusCode::FAILED_PRECONDITION);
}

TEST_CASE("tick routes a reset to the backend and replies with its initial state", "[connector]") {
	Session session;
	InitialAgentState agent_state;
	agent_state.info["spawn"] = "A1";
	session.backend.initial_state.environments[0].agents[AGENT] = agent_state;
	session.connector.updates.push_back(reset_update(42));
	REQUIRE(session.start().is_ok());

	REQUIRE(session.loop.tick(50ms).is_ok());

	REQUIRE(session.backend.resets.size() == 1);
	CHECK(session.backend.resets[0].environments.at(0).seed == 42);
	REQUIRE(session.connector.reset_replies.size() == 1);
	CHECK(session.connector.reset_replies[0].environments.at(0).agents.at(AGENT).info.at("spawn") == "A1");
	CHECK(session.loop.is_running());
}

TEST_CASE("tick routes a step to the backend and replies with its result", "[connector]") {
	Session session;
	AgentState agent_state;
	agent_state.reward = 1.5f;
	agent_state.terminated = true;
	EnvironmentState environment_state;
	environment_state.agents[AGENT] = agent_state;
	session.backend.step_result.state.environments.push_back(environment_state);
	session.connector.updates.push_back(step_update(2));
	REQUIRE(session.start().is_ok());

	REQUIRE(session.loop.tick(50ms).is_ok());

	REQUIRE(session.backend.steps.size() == 1);
	const DiscretePoint *action = session.backend.steps[0].actions.at(0).at(AGENT).get_if<DiscretePoint>();
	REQUIRE(action != nullptr);
	CHECK(action->value == 2);
	REQUIRE(session.connector.step_replies.size() == 1);
	const AgentState &reply = session.connector.step_replies[0].state.environments.at(0).agents.at(AGENT);
	CHECK(reply.reward == 1.5f);
	CHECK(reply.terminated);
}

TEST_CASE("a close from Python ends the session with CLOSED and no error", "[connector]") {
	Session session;
	session.connector.updates.push_back(close_update());
	REQUIRE(session.start().is_ok());

	CHECK(session.loop.tick(50ms).get_code() == StatusCode::CLOSED);

	CHECK_FALSE(session.connector.failure.has_value());
	CHECK(session.backend.close_count == 1);
	CHECK(session.connector.close_count == 1);
	CHECK_FALSE(session.loop.is_running());
}

TEST_CASE("a backend failure is reported to Python and ends the session", "[connector]") {
	Session session;
	session.backend.step_status = Status::error(StatusCode::RUNTIME_FAILURE, "agent_0 failed to apply its action.");
	session.connector.updates.push_back(step_update(1));
	session.connector.updates.push_back(step_update(1));
	REQUIRE(session.start().is_ok());

	CHECK(session.loop.tick(50ms).get_code() == StatusCode::RUNTIME_FAILURE);

	REQUIRE(session.connector.failure.has_value());
	CHECK(session.connector.failure->get_code() == StatusCode::RUNTIME_FAILURE);
	CHECK(session.connector.step_replies.empty());
	CHECK_FALSE(session.loop.is_running());
	CHECK(session.loop.tick(50ms).get_code() == StatusCode::FAILED_PRECONDITION);
	CHECK(session.backend.steps.size() == 1);
}

TEST_CASE("tick times out and reports it when Python stops sending", "[connector]") {
	Session session;
	REQUIRE(session.start().is_ok());

	CHECK(session.loop.tick(10ms).get_code() == StatusCode::TIMEOUT);

	REQUIRE(session.connector.failure.has_value());
	CHECK(session.connector.failure->get_code() == StatusCode::TIMEOUT);
	CHECK_FALSE(session.loop.is_running());
}

TEST_CASE("a disconnect ends the session without answering Python", "[connector]") {
	Session session;
	session.connector.next_update_error = Status::error(StatusCode::CLOSED, "Python disconnected.");
	REQUIRE(session.start().is_ok());

	CHECK(session.loop.tick(50ms).get_code() == StatusCode::CLOSED);

	CHECK_FALSE(session.connector.failure.has_value());
	CHECK(session.connector.close_count == 1);
	CHECK(session.backend.close_count == 1);
	CHECK_FALSE(session.loop.is_running());
}

TEST_CASE("poll_tick returns NOT_FOUND while waiting, then TIMEOUT after the limit", "[connector]") {
	Session session;
	REQUIRE(session.start().is_ok());

	CHECK(session.loop.poll_tick(10s).get_code() == StatusCode::NOT_FOUND);
	CHECK(session.loop.is_running());

	std::this_thread::sleep_for(30ms);
	CHECK(session.loop.poll_tick(10ms).get_code() == StatusCode::TIMEOUT);
	REQUIRE(session.connector.failure.has_value());
	CHECK(session.connector.failure->get_code() == StatusCode::TIMEOUT);
	CHECK_FALSE(session.loop.is_running());
}

TEST_CASE("poll_tick with a zero timeout never times out", "[connector]") {
	Session session;
	REQUIRE(session.start().is_ok());
	std::this_thread::sleep_for(5ms);

	CHECK(session.loop.poll_tick(0ms).get_code() == StatusCode::NOT_FOUND);
	CHECK(session.loop.is_running());
}

TEST_CASE("poll_tick runs an update as soon as it arrives", "[connector]") {
	Session session;
	session.connector.updates.push_back(step_update(3));
	REQUIRE(session.start().is_ok());

	REQUIRE(session.loop.poll_tick(10ms).is_ok());

	CHECK(session.connector.step_replies.size() == 1);
	CHECK(session.loop.is_running());
}

TEST_CASE("stop closes the backend and connector once and is safe to repeat", "[connector]") {
	Session session;
	REQUIRE(session.start().is_ok());

	session.loop.stop();
	session.loop.stop();

	CHECK(session.backend.close_count == 1);
	CHECK(session.connector.close_count == 1);
	CHECK_FALSE(session.loop.is_running());
}
