// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "core/lifecycle/environment_coordinator.h"
#include "support/fake_environment.h"

#include <catch_amalgamated.hpp>

#include <array>
#include <memory>
#include <string>
#include <vector>

using namespace schola;

namespace {

Point point(int32_t p_value) {
	return Point(DiscretePoint{ p_value });
}

AgentState agent_state(int32_t p_value, bool p_done = false) {
	return AgentState{ point(p_value), static_cast<float>(p_value), p_done, false, { { "step", std::to_string(p_value) } } };
}

struct Session {
	std::vector<std::string> calls;
	std::array<testing::FakeEnvironment, 2> environments;
	EnvironmentCoordinator coordinator;

	explicit Session(uint64_t p_limit = 0) {
		std::vector<std::unique_ptr<EnvironmentLifecycle>> lifecycles;
		for (size_t index = 0; index < environments.size(); ++index) {
			auto &environment = environments[index];
			environment.label = std::to_string(index);
			environment.call_log = &calls;
			environment.definitions.emplace("agent", InteractionDefinition{ Space(DiscreteSpace{ 10 }), Space(DiscreteSpace{ 2 }), environment.label });
			environment.reset_states.emplace("agent", InitialAgentState{ point(static_cast<int32_t>(index)), { { "episode", environment.label } } });
			environment.step_states.emplace("agent", agent_state(static_cast<int32_t>(index + 2)));
			lifecycles.push_back(std::make_unique<EnvironmentLifecycle>(static_cast<EnvironmentId>(index), environment, p_limit));
		}
		REQUIRE(coordinator.set_lifecycles(std::move(lifecycles)).is_ok());
	}

	void define(AutoResetMode p_mode = AutoResetMode::DISABLED) {
		REQUIRE(coordinator.set_autoreset_mode(p_mode).is_ok());
		TrainingDefinition definition;
		REQUIRE(coordinator.define(definition).is_ok());
	}

	void start(AutoResetMode p_mode = AutoResetMode::DISABLED) {
		define(p_mode);
		InitialState initial;
		REQUIRE(coordinator.reset({}, initial).is_ok());
	}

	StepRequest actions() const {
		return StepRequest{ { { { "agent", point(0) } }, { { "agent", point(1) } } } };
	}
};

StepResult sentinel_result() {
	StepResult result;
	result.state.environments.push_back(EnvironmentState{ { { "sentinel", agent_state(9) } } });
	result.initial_state.environments.emplace(99, InitialEnvironmentState{ { { "sentinel", InitialAgentState{ point(8), {} } } } });
	return result;
}

void check_sentinel(const StepResult &p_result) {
	REQUIRE(p_result.state.environments.size() == 1);
	REQUIRE(p_result.state.environments[0].agents.size() == 1);
	CHECK(p_result.state.environments[0].agents.at("sentinel").observation == point(9));
	CHECK(p_result.state.environments[0].agents.at("sentinel").reward == 9.0f);
	REQUIRE(p_result.initial_state.environments.size() == 1);
	CHECK(p_result.initial_state.environments.at(99).agents.at("sentinel").observation == point(8));
}

void check_faulted(Session &p_session) {
	CHECK(p_session.coordinator.get_state() == CoordinatorState::FAULTED);
	const auto calls = p_session.calls;
	TrainingDefinition definition;
	InitialState initial;
	StepResult result = sentinel_result();
	CHECK_FALSE(p_session.coordinator.define(definition).is_ok());
	CHECK_FALSE(p_session.coordinator.reset({}, initial).is_ok());
	CHECK_FALSE(p_session.coordinator.step(p_session.actions(), result).is_ok());
	CHECK_FALSE(p_session.coordinator.set_autoreset_mode(AutoResetMode::DISABLED).is_ok());
	std::vector<std::unique_ptr<EnvironmentLifecycle>> lifecycles;
	CHECK_FALSE(p_session.coordinator.set_lifecycles(std::move(lifecycles)).is_ok());
	CHECK(p_session.calls == calls);
	check_sentinel(result);
	CHECK(p_session.coordinator.close().is_ok());
	CHECK(p_session.coordinator.close().is_ok());
}

} // namespace

TEST_CASE("coordinator registration validates before taking ownership", "[coordinator]") {
	EnvironmentCoordinator coordinator;
	testing::FakeEnvironment environment;
	std::vector<std::unique_ptr<EnvironmentLifecycle>> lifecycles;
	CHECK(coordinator.get_state() == CoordinatorState::CONFIGURING);
	CHECK_FALSE(coordinator.set_lifecycles(std::move(lifecycles)).is_ok());

	SECTION("null lifecycle") {
		lifecycles.push_back(nullptr);
	}
	SECTION("ID differs from position") {
		lifecycles.push_back(std::make_unique<EnvironmentLifecycle>(1, environment));
	}
	SECTION("a later invalid entry preserves earlier ownership") {
		lifecycles.push_back(std::make_unique<EnvironmentLifecycle>(0, environment));
		lifecycles.push_back(nullptr);
	}
	SECTION("already closed lifecycle") {
		auto lifecycle = std::make_unique<EnvironmentLifecycle>(0, environment);
		REQUIRE(lifecycle->close().is_ok());
		lifecycles.push_back(std::move(lifecycle));
	}
	const auto count = lifecycles.size();
	const auto *first = lifecycles[0].get();
	CHECK_FALSE(coordinator.set_lifecycles(std::move(lifecycles)).is_ok());
	REQUIRE(lifecycles.size() == count);
	CHECK(lifecycles[0].get() == first);
	CHECK(coordinator.get_state() == CoordinatorState::CONFIGURING);
}

TEST_CASE("coordinator takes lifecycle ownership only on valid registration", "[coordinator]") {
	EnvironmentCoordinator coordinator;
	testing::FakeEnvironment environment;
	std::vector<std::unique_ptr<EnvironmentLifecycle>> lifecycles;
	lifecycles.push_back(std::make_unique<EnvironmentLifecycle>(0, environment));
	REQUIRE(coordinator.set_lifecycles(std::move(lifecycles)).is_ok());
	CHECK(lifecycles.empty());
	CHECK(coordinator.close().is_ok());
}

TEST_CASE("define requires registration and a valid one-time autoreset mode", "[coordinator]") {
	EnvironmentCoordinator empty;
	TrainingDefinition output;
	output.environments.resize(3);
	CHECK_FALSE(empty.define(output).is_ok());
	REQUIRE(empty.set_autoreset_mode(AutoResetMode::DISABLED).is_ok());
	CHECK_FALSE(empty.define(output).is_ok());
	CHECK(output.environments.size() == 3);

	Session session;
	CHECK_FALSE(session.coordinator.define(output).is_ok());
	CHECK(session.calls.empty());
	CHECK_FALSE(session.coordinator.set_autoreset_mode(static_cast<AutoResetMode>(99)).is_ok());
	REQUIRE(session.coordinator.set_autoreset_mode(AutoResetMode::NEXT_STEP).is_ok());
	CHECK_FALSE(session.coordinator.set_autoreset_mode(AutoResetMode::DISABLED).is_ok());
	REQUIRE(session.coordinator.define(output).is_ok());
	CHECK(session.coordinator.get_state() == CoordinatorState::READY);
	REQUIRE(output.environments.size() == 2);
	CHECK(output.environments[0].agents.at("agent").agent_type == "0");
	CHECK(output.environments[1].agents.at("agent").agent_type == "1");
	CHECK(session.calls == std::vector<std::string>{ "0:initialize", "1:initialize" });
	session.environments[0].definitions.at("agent").agent_type = "changed";
	CHECK(output.environments[0].agents.at("agent").agent_type == "0");
	CHECK_FALSE(session.coordinator.define(output).is_ok());
	std::vector<std::unique_ptr<EnvironmentLifecycle>> replacement;
	replacement.push_back(std::make_unique<EnvironmentLifecycle>(0, session.environments[0]));
	CHECK_FALSE(session.coordinator.set_lifecycles(std::move(replacement)).is_ok());
	CHECK(replacement.size() == 1);
}

TEST_CASE("reset and step reject premature calls without touching outputs", "[coordinator]") {
	Session session;
	InitialState initial;
	initial.environments.emplace(99, InitialEnvironmentState{});
	CHECK_FALSE(session.coordinator.reset({}, initial).is_ok());
	StepResult result = sentinel_result();
	CHECK_FALSE(session.coordinator.step(session.actions(), result).is_ok());
	CHECK(session.calls.empty());
	session.define();
	CHECK_FALSE(session.coordinator.step(session.actions(), result).is_ok());
	check_sentinel(result);
	CHECK(initial.environments.count(99) == 1);
}

TEST_CASE("reset routes settings by ID and steps route actions in stable order", "[coordinator]") {
	Session session;
	session.define();
	ResetRequest request;
	request.environments.emplace(1, ResetSettings{ 42, { { "difficulty", "hard" } } });
	InitialState initial;
	initial.environments.emplace(99, InitialEnvironmentState{});
	REQUIRE(session.coordinator.reset(request, initial).is_ok());
	CHECK(session.coordinator.get_state() == CoordinatorState::RUNNING);
	CHECK_FALSE(session.environments[0].last_reset_settings.seed.has_value());
	CHECK(session.environments[0].last_reset_settings.options.empty());
	CHECK(session.environments[1].last_reset_settings.seed == 42);
	CHECK(session.environments[1].last_reset_settings.options.at("difficulty") == "hard");
	REQUIRE(initial.environments.size() == 2);
	CHECK(initial.environments.at(0).agents.at("agent").observation == point(0));
	CHECK(initial.environments.at(1).agents.at("agent").observation == point(1));
	StepResult result = sentinel_result();
	const auto actions = session.actions();
	REQUIRE(session.coordinator.step(actions, result).is_ok());
	REQUIRE(result.state.environments.size() == 2);
	CHECK(result.initial_state.environments.empty());
	for (size_t index = 0; index < session.environments.size(); ++index) {
		CHECK(session.environments[index].last_actions == actions.actions[index]);
		CHECK(result.state.environments[index].agents.at("agent").observation == point(static_cast<int32_t>(index + 2)));
	}
	CHECK(session.calls == std::vector<std::string>{ "0:initialize", "1:initialize", "0:reset", "1:reset", "0:step", "1:step" });
	REQUIRE(session.coordinator.reset({}, initial).is_ok());
	CHECK_FALSE(session.environments[1].last_reset_settings.seed.has_value());
	CHECK(session.environments[1].last_reset_settings.options.empty());
	CHECK(result.state.environments[0].agents.at("agent").observation == point(2));
}

TEST_CASE("invalid batch structure is rejected before any environment runs", "[coordinator]") {
	Session session;
	session.start();
	const auto calls = session.calls;
	SECTION("unknown reset ID") {
		ResetRequest request;
		request.environments.emplace(2, ResetSettings{});
		InitialState output;
		output.environments.emplace(99, InitialEnvironmentState{});
		CHECK_FALSE(session.coordinator.reset(request, output).is_ok());
		CHECK(output.environments.count(99) == 1);
	}
	SECTION("negative reset ID") {
		ResetRequest request;
		request.environments.emplace(-1, ResetSettings{});
		InitialState output;
		CHECK_FALSE(session.coordinator.reset(request, output).is_ok());
	}
	SECTION("too few action maps") {
		StepRequest request;
		request.actions.resize(1);
		StepResult output = sentinel_result();
		CHECK_FALSE(session.coordinator.step(request, output).is_ok());
		check_sentinel(output);
	}
	SECTION("too many action maps") {
		StepRequest request;
		request.actions.resize(3);
		StepResult output = sentinel_result();
		CHECK_FALSE(session.coordinator.step(request, output).is_ok());
		check_sentinel(output);
	}
	CHECK(session.coordinator.get_state() == CoordinatorState::RUNNING);
	CHECK(session.calls == calls);
}

TEST_CASE("disabled autoreset retains terminal snapshots while other environments advance", "[coordinator][autoreset]") {
	Session session;
	session.start();
	session.environments[0].step_states.at("agent") = agent_state(4, true);
	StepResult terminal;
	REQUIRE(session.coordinator.step(session.actions(), terminal).is_ok());
	session.environments[0].step_states.at("agent") = agent_state(8);
	session.environments[1].step_states.at("agent") = agent_state(5);
	StepRequest request = session.actions();
	request.actions[0].clear();
	StepResult later;
	REQUIRE(session.coordinator.step(request, later).is_ok());
	const auto &retained = later.state.environments[0].agents.at("agent");
	CHECK(retained.observation == point(4));
	CHECK(retained.reward == 4.0f);
	CHECK(retained.terminated);
	CHECK(retained.info.at("step") == "4");
	CHECK(session.environments[0].step_calls == 1);
	CHECK(session.environments[0].reset_calls == 1);
	CHECK(session.environments[1].step_calls == 2);
	CHECK(later.state.environments[1].agents.at("agent").observation == point(5));
	CHECK(terminal.state.environments[1].agents.at("agent").observation == point(3));
	CHECK(later.initial_state.environments.empty());
	InitialState initial;
	REQUIRE(session.coordinator.reset({}, initial).is_ok());
	StepResult fresh;
	REQUIRE(session.coordinator.step(session.actions(), fresh).is_ok());
	CHECK_FALSE(fresh.state.environments[0].agents.at("agent").terminated);
	CHECK(fresh.state.environments[0].agents.at("agent").observation == point(8));
	CHECK(retained.observation == point(4));
}

TEST_CASE("same-step autoreset separates terminal snapshots from reset observations", "[coordinator][autoreset]") {
	Session session;
	session.start(AutoResetMode::SAME_STEP);
	session.environments[0].step_states.at("agent") = agent_state(4, true);
	StepResult result;
	REQUIRE(session.coordinator.step(session.actions(), result).is_ok());
	CHECK(result.state.environments[0].agents.at("agent").terminated);
	CHECK(result.state.environments[0].agents.at("agent").observation == point(4));
	REQUIRE(result.initial_state.environments.size() == 1);
	CHECK(result.initial_state.environments.at(0).agents.at("agent").observation == point(0));
	CHECK(session.environments[0].reset_calls == 2);
	CHECK(session.environments[1].reset_calls == 1);
	CHECK(session.calls == std::vector<std::string>{ "0:initialize", "1:initialize", "0:reset", "1:reset", "0:step", "0:reset", "1:step" });
	session.environments[0].step_states.at("agent") = agent_state(5);
	session.environments[0].reset_states.at("agent").observation = point(8);
	session.environments[1].step_states.at("agent") = agent_state(6, true);
	StepResult next;
	REQUIRE(session.coordinator.step(session.actions(), next).is_ok());
	CHECK(next.state.environments[0].agents.at("agent").observation == point(5));
	CHECK(next.initial_state.environments.count(0) == 0);
	CHECK(next.initial_state.environments.count(1) == 1);
	CHECK(result.initial_state.environments.at(0).agents.at("agent").observation == point(0));
}

TEST_CASE("agents finish independently within an environment and across the batch", "[coordinator]") {
	Session session;
	auto &first = session.environments[0];
	first.definitions.emplace("partner", first.definitions.at("agent"));
	first.reset_states.emplace("partner", InitialAgentState{ point(1), {} });
	first.step_states.at("agent") = agent_state(4, true);
	first.step_states.emplace("partner", agent_state(5));
	session.environments[1].step_states.at("agent") = agent_state(6, true);
	session.start();
	auto request = session.actions();
	request.actions[0].emplace("partner", point(1));
	StepResult initial;
	REQUIRE(session.coordinator.step(request, initial).is_ok());
	CHECK_FALSE(initial.state.environments[0].agents.at("partner").terminated);
	first.step_states.erase("agent");
	first.step_states.at("partner") = agent_state(7, true);
	request.actions[0].erase("agent");
	request.actions[1].clear();
	StepResult finished;
	REQUIRE(session.coordinator.step(request, finished).is_ok());
	REQUIRE(finished.state.environments[0].agents.size() == 2);
	CHECK(finished.state.environments[0].agents.at("agent").observation == point(4));
	CHECK(finished.state.environments[0].agents.at("partner").observation == point(7));
	CHECK(finished.state.environments[1].agents.at("agent").observation == point(6));
	CHECK(first.step_calls == 2);
	CHECK(session.environments[1].step_calls == 1);
	StepResult retained;
	REQUIRE(session.coordinator.step(StepRequest{ { {}, {} } }, retained).is_ok());
	CHECK(retained.state.environments[0].agents.at("partner").terminated);
	CHECK(first.step_calls == 2);
}

TEST_CASE("next-step autoreset consumes no action and returns a fresh observation", "[coordinator][autoreset]") {
	Session session;
	session.start(AutoResetMode::NEXT_STEP);
	session.environments[0].step_states.at("agent") = agent_state(4, true);
	StepResult terminal;
	REQUIRE(session.coordinator.step(session.actions(), terminal).is_ok());
	CHECK(terminal.state.environments[0].agents.at("agent").terminated);
	CHECK(session.environments[0].reset_calls == 1);
	StepRequest ignored = session.actions();
	SECTION("empty action map") {
		ignored.actions[0].clear();
	}
	SECTION("nonempty action map is ignored") {
		ignored.actions[0] = { { "ignored", point(99) } };
	}
	StepResult reset;
	REQUIRE(session.coordinator.step(ignored, reset).is_ok());
	const auto &agent = reset.state.environments[0].agents.at("agent");
	CHECK(agent.observation == point(0));
	CHECK(agent.reward == 0.0f);
	CHECK_FALSE(agent.terminated);
	CHECK_FALSE(agent.truncated);
	CHECK(agent.info.at("episode") == "0");
	CHECK(reset.initial_state.environments.empty());
	CHECK(session.environments[0].reset_calls == 2);
	CHECK(session.environments[0].step_calls == 1);
	CHECK(session.environments[1].step_calls == 2);
	session.environments[0].step_states.at("agent") = agent_state(5);
	StepResult next;
	REQUIRE(session.coordinator.step(session.actions(), next).is_ok());
	CHECK(session.environments[0].step_calls == 2);
	CHECK(next.state.environments[0].agents.at("agent").observation == point(5));
	CHECK(terminal.state.environments[0].agents.at("agent").observation == point(4));
}

TEST_CASE("autoreset uses default settings and supports repeated truncated episodes", "[coordinator][autoreset]") {
	Session session(1);
	auto mode = AutoResetMode::SAME_STEP;
	SECTION("same step") {}
	SECTION("next step") {
		mode = AutoResetMode::NEXT_STEP;
	}
	session.define(mode);
	ResetRequest request;
	request.environments.emplace(0, ResetSettings{ 42, { { "once", "only" } } });
	InitialState initial;
	REQUIRE(session.coordinator.reset(request, initial).is_ok());
	for (int episode = 0; episode < 3; ++episode) {
		StepResult terminal;
		REQUIRE(session.coordinator.step(session.actions(), terminal).is_ok());
		CHECK(terminal.state.environments[0].agents.at("agent").truncated);
		CHECK(terminal.state.environments[1].agents.at("agent").truncated);
		if (mode == AutoResetMode::NEXT_STEP) {
			StepResult reset;
			REQUIRE(session.coordinator.step(StepRequest{ { {}, {} } }, reset).is_ok());
			CHECK_FALSE(reset.state.environments[0].agents.at("agent").truncated);
		} else {
			CHECK(terminal.initial_state.environments.size() == 2);
		}
		CHECK_FALSE(session.environments[0].last_reset_settings.seed.has_value());
		CHECK(session.environments[0].last_reset_settings.options.empty());
	}
	CHECK(session.environments[0].reset_calls == 4);
	CHECK(session.environments[0].step_calls == 3);
}

TEST_CASE("failed define preserves output and faults the coordinator", "[coordinator][failure]") {
	Session session;
	size_t failed = 0;
	SECTION("first environment") {}
	SECTION("later environment") {
		failed = 1;
	}
	session.environments[failed].initialize_status = Status::error(StatusCode::RUNTIME_FAILURE, "initialize failed");
	REQUIRE(session.coordinator.set_autoreset_mode(AutoResetMode::DISABLED).is_ok());
	TrainingDefinition output;
	output.environments.resize(3);
	const Status result = session.coordinator.define(output);
	CHECK(result.get_code() == StatusCode::RUNTIME_FAILURE);
	CHECK(result.get_message().find("environment " + std::to_string(failed)) != std::string::npos);
	CHECK(output.environments.size() == 3);
	CHECK(session.environments[1].initialize_calls == (failed == 1 ? 1 : 0));
	check_faulted(session);
}

TEST_CASE("failed reset preserves output and faults the coordinator", "[coordinator][failure]") {
	Session session;
	session.define();
	size_t failed = 0;
	SECTION("first environment") {}
	SECTION("later environment") {
		failed = 1;
	}
	session.environments[failed].reset_status = Status::error(StatusCode::RUNTIME_FAILURE, "reset failed");
	InitialState output;
	output.environments.emplace(99, InitialEnvironmentState{});
	CHECK_FALSE(session.coordinator.reset({}, output).is_ok());
	REQUIRE(output.environments.size() == 1);
	CHECK(output.environments.count(99) == 1);
	CHECK(session.environments[1].reset_calls == (failed == 1 ? 1 : 0));
	check_faulted(session);
}

TEST_CASE("failed batch step preserves all output and faults the coordinator", "[coordinator][failure]") {
	Session session;
	session.start();
	size_t failed = 0;
	SECTION("first environment") {}
	SECTION("later environment") {
		failed = 1;
	}
	session.environments[failed].step_status = Status::error(StatusCode::RUNTIME_FAILURE, "step failed");
	StepResult output = sentinel_result();
	CHECK_FALSE(session.coordinator.step(session.actions(), output).is_ok());
	CHECK(session.environments[1].step_calls == (failed == 1 ? 1 : 0));
	check_sentinel(output);
	check_faulted(session);
}

TEST_CASE("failed autoreset preserves output and faults the coordinator", "[coordinator][autoreset][failure]") {
	Session session;
	auto mode = AutoResetMode::SAME_STEP;
	SECTION("same step") {}
	SECTION("next step") {
		mode = AutoResetMode::NEXT_STEP;
	}
	session.start(mode);
	session.environments[1].step_states.at("agent") = agent_state(4, true);
	if (mode == AutoResetMode::NEXT_STEP) {
		StepResult terminal;
		REQUIRE(session.coordinator.step(session.actions(), terminal).is_ok());
	}
	session.environments[1].reset_status = Status::error(StatusCode::RUNTIME_FAILURE, "autoreset failed");
	StepResult output = sentinel_result();
	CHECK_FALSE(session.coordinator.step(session.actions(), output).is_ok());
	check_sentinel(output);
	check_faulted(session);
}

TEST_CASE("lifecycle validation failure after earlier work faults the entire batch", "[coordinator][failure]") {
	Session session;
	session.start();
	auto request = session.actions();
	request.actions[1].clear();
	StepResult output = sentinel_result();
	CHECK_FALSE(session.coordinator.step(request, output).is_ok());
	CHECK(session.environments[0].step_calls == 1);
	CHECK(session.environments[1].step_calls == 0);
	check_sentinel(output);
	check_faulted(session);
}

TEST_CASE("close is legal in every coordinator phase and rejects subsequent work", "[coordinator]") {
	Session session;
	SECTION("configuring") {}
	SECTION("ready") {
		session.define();
	}
	SECTION("running") {
		session.start();
	}
	REQUIRE(session.coordinator.close().is_ok());
	CHECK(session.coordinator.get_state() == CoordinatorState::CLOSED);
	const auto calls = session.calls;
	REQUIRE(session.coordinator.close().is_ok());
	TrainingDefinition definition;
	InitialState initial;
	StepResult result = sentinel_result();
	CHECK_FALSE(session.coordinator.define(definition).is_ok());
	CHECK_FALSE(session.coordinator.reset({}, initial).is_ok());
	CHECK_FALSE(session.coordinator.step(session.actions(), result).is_ok());
	CHECK_FALSE(session.coordinator.set_autoreset_mode(AutoResetMode::DISABLED).is_ok());
	std::vector<std::unique_ptr<EnvironmentLifecycle>> lifecycles;
	CHECK_FALSE(session.coordinator.set_lifecycles(std::move(lifecycles)).is_ok());
	CHECK(session.calls == calls);
	check_sentinel(result);
}
