// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "core/lifecycle/environment_lifecycle.h"
#include "support/fake_environment.h"

#include <catch_amalgamated.hpp>

#include <map>
#include <string>
#include <vector>

using namespace schola;

namespace {

// Build a discrete point without repeating its nested representation.
Point make_point(int32_t p_value) {
	return Point(DiscretePoint{ p_value });
}

// Build a minimal discrete interaction definition for a test agent.
InteractionDefinition make_definition(const std::string &p_agent_type = "test_agent") {
	return InteractionDefinition{
		Space(DiscreteSpace{ 4 }),
		Space(DiscreteSpace{ 2 }),
		p_agent_type,
	};
}

// Build an initial state with a discrete observation and episode label.
InitialAgentState make_initial_state(int32_t p_observation, const std::string &p_episode) {
	return InitialAgentState{
		make_point(p_observation),
		{ { "episode", p_episode } },
	};
}

// Build a step result with configurable reward and terminal flags.
AgentState make_agent_state(int32_t p_observation, float p_reward, bool p_terminated = false,
		bool p_truncated = false, const std::string &p_label = "step") {
	return AgentState{
		make_point(p_observation),
		p_reward,
		p_terminated,
		p_truncated,
		{ { "label", p_label } },
	};
}

// Compare initial states without requiring an aggregate equality operator.
void require_initial_state(const InitialAgentState &p_actual, const InitialAgentState &p_expected) {
	REQUIRE(p_actual.observation == p_expected.observation);
	REQUIRE(p_actual.info == p_expected.info);
}

// Compare agent states without requiring an aggregate equality operator.
void require_agent_state(const AgentState &p_actual, const AgentState &p_expected) {
	REQUIRE(p_actual.observation == p_expected.observation);
	REQUIRE(p_actual.reward == p_expected.reward);
	REQUIRE(p_actual.terminated == p_expected.terminated);
	REQUIRE(p_actual.truncated == p_expected.truncated);
	REQUIRE(p_actual.info == p_expected.info);
}

// Initialize a lifecycle with definitions for the requested agents.
void initialize_lifecycle(EnvironmentLifecycle &p_lifecycle, testing::FakeEnvironment &p_environment,
		const std::vector<AgentId> &p_agent_ids = { "agent_0" }) {
	for (const AgentId &agent_id : p_agent_ids) {
		p_environment.definitions.emplace(agent_id, make_definition(agent_id));
		p_environment.reset_states.emplace(agent_id, make_initial_state(0, "initial"));
	}

	const Status result = p_lifecycle.initialize();
	REQUIRE(result.is_ok());
	REQUIRE(p_lifecycle.get_state() == LifecycleState::RESET_PENDING);
}

// Configure, initialize, and reset a lifecycle with the requested agents.
void start_active(EnvironmentLifecycle &p_lifecycle, testing::FakeEnvironment &p_environment,
		const std::vector<AgentId> &p_agent_ids = { "agent_0" }) {
	for (const AgentId &agent_id : p_agent_ids) {
		p_environment.reset_states.emplace(agent_id, make_initial_state(0, "initial"));
	}

	initialize_lifecycle(p_lifecycle, p_environment, p_agent_ids);

	InitialEnvironmentState initial_state;
	const Status reset_result = p_lifecycle.reset({}, initial_state);
	REQUIRE(reset_result.is_ok());
	REQUIRE(p_lifecycle.get_state() == LifecycleState::ACTIVE);
}

} // namespace

// A new lifecycle is uninitialized and has not touched its environment.
TEST_CASE("environment lifecycle starts uninitialized", "[lifecycle]") {
	testing::FakeEnvironment environment;
	constexpr EnvironmentId environment_id = 67;
	EnvironmentLifecycle lifecycle(environment_id, environment);

	CHECK(lifecycle.get_id() == environment_id);
	CHECK(lifecycle.get_state() == LifecycleState::UNINITIALIZED);
	CHECK(lifecycle.get_episode_count() == 0);
	CHECK(lifecycle.get_episode_step_count() == 0);
	CHECK(environment.initialize_calls == 0);
	CHECK(environment.reset_calls == 0);
	CHECK(environment.step_calls == 0);
}

// Initialization stores the environment definition and awaits a reset.
TEST_CASE("initialize stores the definition and transitions to reset pending", "[lifecycle]") {
	testing::FakeEnvironment environment;
	constexpr EnvironmentId environment_id = 67;
	const InteractionDefinition expected_definition = make_definition();
	environment.definitions.emplace("agent_0", expected_definition);
	EnvironmentLifecycle lifecycle(environment_id, environment);

	const Status result = lifecycle.initialize();

	REQUIRE(result.is_ok());
	CHECK(lifecycle.get_state() == LifecycleState::RESET_PENDING);
	CHECK(environment.initialize_calls == 1);

	const auto &actual_agents = lifecycle.get_definition().agents;
	REQUIRE(actual_agents.size() == 1);

	const InteractionDefinition &actual = actual_agents.at("agent_0");
	CHECK(actual.observation_space == expected_definition.observation_space);
	CHECK(actual.action_space == expected_definition.action_space);
	CHECK(actual.agent_type == expected_definition.agent_type);
}

// The lifecycle owns its definition independently of the environment.
TEST_CASE("initialized definition remains stable", "[lifecycle]") {
	testing::FakeEnvironment environment;
	const InteractionDefinition expected_definition = make_definition();
	environment.definitions.emplace("agent_0", expected_definition);
	EnvironmentLifecycle lifecycle(0, environment);

	const Status result = lifecycle.initialize();
	REQUIRE(result.is_ok());

	environment.definitions.at("agent_0").agent_type = "mutated";
	const InteractionDefinition &actual = lifecycle.get_definition().agents.at("agent_0");
	CHECK(actual.agent_type == expected_definition.agent_type);
}

// Reinitialization fails without calling the environment again.
TEST_CASE("repeated initialize is rejected before calling the environment", "[lifecycle]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);
	initialize_lifecycle(lifecycle, environment);

	const Status result = lifecycle.initialize();

	CHECK_FALSE(result.is_ok());
	CHECK(lifecycle.get_state() == LifecycleState::RESET_PENDING);
	CHECK(environment.initialize_calls == 1);
}

// Reset forwards settings and activates the lifecycle.
TEST_CASE("reset forwards settings and activates the lifecycle", "[lifecycle]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);
	initialize_lifecycle(lifecycle, environment);

	ResetSettings settings;
	settings.seed = 42;
	settings.options.emplace("difficulty", "hard");
	InitialEnvironmentState result_state;
	const Status result = lifecycle.reset(settings, result_state);

	REQUIRE(result.is_ok());
	CHECK(lifecycle.get_state() == LifecycleState::ACTIVE);
	CHECK(environment.reset_calls == 1);
	CHECK(environment.last_reset_settings.seed == settings.seed);
	CHECK(environment.last_reset_settings.options == settings.options);
}

// Reset returns state that is independent of the environment's storage.
TEST_CASE("reset returns an owned initial state", "[lifecycle]") {
	testing::FakeEnvironment environment;
	const InitialAgentState expected_state = make_initial_state(2, "first");
	environment.reset_states.emplace("agent_0", expected_state);
	EnvironmentLifecycle lifecycle(0, environment);
	initialize_lifecycle(lifecycle, environment);

	InitialEnvironmentState result_state;
	const Status result = lifecycle.reset({}, result_state);

	REQUIRE(result.is_ok());
	REQUIRE(result_state.agents.size() == 1);
	require_initial_state(result_state.agents.at("agent_0"), expected_state);

	environment.reset_states.at("agent_0").observation = make_point(3);
	require_initial_state(result_state.agents.at("agent_0"), expected_state);
}

// Reset may interrupt an active episode.
TEST_CASE("reset is allowed while active", "[lifecycle]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);
	start_active(lifecycle, environment);

	InitialEnvironmentState state;
	const Status result = lifecycle.reset({}, state);

	REQUIRE(result.is_ok());
	CHECK(lifecycle.get_state() == LifecycleState::ACTIVE);
	CHECK(environment.reset_calls == 2);
}

// Reset starts a new active episode after completion.
TEST_CASE("reset is allowed after completion", "[lifecycle]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);
	start_active(lifecycle, environment);

	environment.step_states = {
		{ "agent_0", make_agent_state(1, 1.0f, true) },
	};
	EnvironmentState terminal_state;
	const Status step_result = lifecycle.step({ { "agent_0", make_point(1) } }, terminal_state);
	REQUIRE(step_result.is_ok());
	REQUIRE(lifecycle.get_state() == LifecycleState::COMPLETE);

	InitialEnvironmentState reset_state;
	const Status result = lifecycle.reset({}, reset_state);

	REQUIRE(result.is_ok());
	CHECK(lifecycle.get_state() == LifecycleState::ACTIVE);
	CHECK(environment.reset_calls == 2);
}

// A successful reset starts exactly one episode.
TEST_CASE("reset increments the episode count", "[lifecycle]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);
	initialize_lifecycle(lifecycle, environment);

	InitialEnvironmentState state;
	REQUIRE(lifecycle.reset({}, state).is_ok());

	CHECK(lifecycle.get_episode_count() == 1);
}

// A successful step advances the current episode by one step.
TEST_CASE("step increments the episode step count", "[lifecycle]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);
	start_active(lifecycle, environment);
	environment.step_states = {
		{ "agent_0", make_agent_state(1, 1.0f) },
	};

	EnvironmentState state;
	REQUIRE(lifecycle.step({ { "agent_0", make_point(1) } }, state).is_ok());

	CHECK(lifecycle.get_episode_step_count() == 1);
}

// A reset removes step progress from the preceding episode.
TEST_CASE("reset clears the episode step count", "[lifecycle]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);
	start_active(lifecycle, environment);
	environment.step_states = {
		{ "agent_0", make_agent_state(1, 1.0f) },
	};
	EnvironmentState step_state;
	REQUIRE(lifecycle.step({ { "agent_0", make_point(1) } }, step_state).is_ok());

	InitialEnvironmentState reset_state;
	REQUIRE(lifecycle.reset({}, reset_state).is_ok());

	CHECK(lifecycle.get_episode_step_count() == 0);
}

// An unsuccessful reset does not start an episode.
TEST_CASE("reset failure does not increment the episode count", "[lifecycle]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);
	initialize_lifecycle(lifecycle, environment);
	environment.reset_status = Status::error(StatusCode::RUNTIME_FAILURE, "reset failed");

	InitialEnvironmentState state;
	CHECK_FALSE(lifecycle.reset({}, state).is_ok());

	CHECK(lifecycle.get_episode_count() == 0);
}

// An unsuccessful step does not advance the current episode.
TEST_CASE("step failure does not increment the episode step count", "[lifecycle]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);
	start_active(lifecycle, environment);
	environment.step_status = Status::error(StatusCode::RUNTIME_FAILURE, "step failed");

	EnvironmentState state;
	CHECK_FALSE(lifecycle.step({ { "agent_0", make_point(1) } }, state).is_ok());

	CHECK(lifecycle.get_episode_step_count() == 0);
}

// Step forwards actions and leaves an unfinished lifecycle active.
TEST_CASE("step forwards actions", "[lifecycle]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);
	start_active(lifecycle, environment);

	const std::map<AgentId, Point> actions{
		{ "agent_0", make_point(1) },
	};
	const AgentState expected_state = make_agent_state(2, 1.5f);
	environment.step_states = {
		{ "agent_0", expected_state },
	};

	EnvironmentState result_state;
	const Status result = lifecycle.step(actions, result_state);

	REQUIRE(result.is_ok());
	CHECK(lifecycle.get_state() == LifecycleState::ACTIVE);
	CHECK(environment.step_calls == 1);
	CHECK(environment.last_actions == actions);
}

// Step returns state that is independent of the environment's storage.
TEST_CASE("step returns an owned state", "[lifecycle]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);
	start_active(lifecycle, environment);

	const AgentState expected_state = make_agent_state(2, 1.5f);
	environment.step_states = {
		{ "agent_0", expected_state },
	};

	EnvironmentState result_state;
	const Status result = lifecycle.step({ { "agent_0", make_point(1) } }, result_state);

	REQUIRE(result.is_ok());
	REQUIRE(result_state.agents.size() == 1);
	require_agent_state(result_state.agents.at("agent_0"), expected_state);

	environment.step_states.at("agent_0").observation = make_point(3);
	require_agent_state(result_state.agents.at("agent_0"), expected_state);
}

// Completion waits for every agent and retains earlier terminal states.
TEST_CASE("completion waits for every agent and preserves finished state", "[lifecycle]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);
	start_active(lifecycle, environment, { "agent_0", "agent_1" });

	const AgentState first_terminal = make_agent_state(1, 2.0f, true, false, "finished");
	environment.step_states = {
		{ "agent_0", first_terminal },
		{ "agent_1", make_agent_state(2, 0.5f) },
	};

	EnvironmentState first_result;
	const Status first_step_result = lifecycle.step(
			{
					{ "agent_0", make_point(1) },
					{ "agent_1", make_point(0) },
			},
			first_result);
	REQUIRE(first_step_result.is_ok());
	REQUIRE(lifecycle.get_state() == LifecycleState::ACTIVE);

	const AgentState second_terminal = make_agent_state(3, 3.0f, false, true, "finished");
	environment.step_states = {
		{ "agent_1", second_terminal },
	};

	EnvironmentState second_result;
	const Status second_step_result = lifecycle.step({ { "agent_1", make_point(1) } }, second_result);
	REQUIRE(second_step_result.is_ok());

	CHECK(lifecycle.get_state() == LifecycleState::COMPLETE);
	REQUIRE(second_result.agents.size() == 2);
	require_agent_state(second_result.agents.at("agent_0"), first_terminal);
	require_agent_state(second_result.agents.at("agent_1"), second_terminal);
}

// Reaching the step limit truncates unfinished agents and completes the episode.
TEST_CASE("configured step limit truncates unfinished agents", "[lifecycle]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment, 2);
	start_active(lifecycle, environment);

	environment.step_states = {
		{ "agent_0", make_agent_state(1, 0.5f) },
	};
	EnvironmentState first_result;
	const Status first_step_result = lifecycle.step({ { "agent_0", make_point(0) } }, first_result);
	REQUIRE(first_step_result.is_ok());
	CHECK_FALSE(first_result.agents.at("agent_0").truncated);
	CHECK(lifecycle.get_state() == LifecycleState::ACTIVE);

	environment.step_states = {
		{ "agent_0", make_agent_state(2, 0.5f) },
	};
	EnvironmentState second_result;
	const Status second_step_result = lifecycle.step({ { "agent_0", make_point(1) } }, second_result);
	REQUIRE(second_step_result.is_ok());

	CHECK(second_result.agents.at("agent_0").truncated);
	CHECK_FALSE(second_result.agents.at("agent_0").terminated);
	CHECK(lifecycle.get_state() == LifecycleState::COMPLETE);
}

// Step-limit truncation does not overwrite agents that are already finished.
TEST_CASE("step limit preserves existing completion flags", "[lifecycle]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment, 1);
	start_active(lifecycle, environment, { "terminated", "truncated", "unfinished" });

	const AgentState terminated = make_agent_state(1, 1.0f, true);
	const AgentState truncated = make_agent_state(2, 2.0f, false, true);
	environment.step_states = {
		{ "terminated", terminated },
		{ "truncated", truncated },
		{ "unfinished", make_agent_state(3, 3.0f) },
	};

	EnvironmentState state;
	const Status result = lifecycle.step(
			{
					{ "terminated", make_point(0) },
					{ "truncated", make_point(0) },
					{ "unfinished", make_point(0) },
			},
			state);

	REQUIRE(result.is_ok());
	require_agent_state(state.agents.at("terminated"), terminated);
	require_agent_state(state.agents.at("truncated"), truncated);
	CHECK(state.agents.at("unfinished").truncated);
}

// A zero step limit disables automatic truncation.
TEST_CASE("zero step limit never truncates an unfinished agent", "[lifecycle]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);
	start_active(lifecycle, environment);
	environment.step_states = {
		{ "agent_0", make_agent_state(1, 0.0f) },
	};

	for (int step = 0; step < 3; ++step) {
		EnvironmentState result;
		const Status step_result = lifecycle.step({ { "agent_0", make_point(0) } }, result);
		REQUIRE(step_result.is_ok());
		CHECK_FALSE(result.agents.at("agent_0").truncated);
		CHECK(lifecycle.get_state() == LifecycleState::ACTIVE);
	}
}

// An invalid reset preserves lifecycle state, output, and call count.
TEST_CASE("reset precondition failure has no side effects", "[lifecycle]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);

	InitialEnvironmentState initial_output;
	const InitialAgentState initial_sentinel = make_initial_state(3, "sentinel");
	initial_output.agents.emplace("sentinel", initial_sentinel);
	const Status reset_result = lifecycle.reset({}, initial_output);

	CHECK_FALSE(reset_result.is_ok());
	CHECK(lifecycle.get_state() == LifecycleState::UNINITIALIZED);
	CHECK(environment.reset_calls == 0);
	REQUIRE(initial_output.agents.size() == 1);
	require_initial_state(initial_output.agents.at("sentinel"), initial_sentinel);
}

// An invalid step preserves lifecycle state, output, and call count.
TEST_CASE("step precondition failure has no side effects", "[lifecycle]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);

	EnvironmentState step_output;
	const AgentState step_sentinel = make_agent_state(3, -1.0f, false, false, "sentinel");
	step_output.agents.emplace("sentinel", step_sentinel);
	const Status step_result = lifecycle.step({}, step_output);

	CHECK_FALSE(step_result.is_ok());
	CHECK(lifecycle.get_state() == LifecycleState::UNINITIALIZED);
	CHECK(environment.step_calls == 0);
	REQUIRE(step_output.agents.size() == 1);
	require_agent_state(step_output.agents.at("sentinel"), step_sentinel);
}

// An environment initialization failure faults the lifecycle.
TEST_CASE("initialize failure faults the lifecycle", "[lifecycle]") {
	testing::FakeEnvironment environment;
	environment.initialize_status = Status::error(StatusCode::RUNTIME_FAILURE, "initialize failed");
	EnvironmentLifecycle lifecycle(0, environment);

	const Status result = lifecycle.initialize();

	CHECK_FALSE(result.is_ok());
	CHECK(lifecycle.get_state() == LifecycleState::FAULTED);
	CHECK(environment.initialize_calls == 1);
}

// An environment reset failure preserves output and faults the lifecycle.
TEST_CASE("reset failure preserves output and faults the lifecycle", "[lifecycle]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);
	initialize_lifecycle(lifecycle, environment);
	environment.reset_status = Status::error(StatusCode::RUNTIME_FAILURE, "reset failed");

	InitialEnvironmentState output;
	const InitialAgentState sentinel = make_initial_state(3, "sentinel");
	output.agents.emplace("sentinel", sentinel);
	const Status result = lifecycle.reset({}, output);

	CHECK_FALSE(result.is_ok());
	CHECK(lifecycle.get_state() == LifecycleState::FAULTED);
	CHECK(environment.reset_calls == 1);
	REQUIRE(output.agents.size() == 1);
	require_initial_state(output.agents.at("sentinel"), sentinel);
}

// An environment step failure preserves output and faults the lifecycle.
TEST_CASE("step failure preserves output and faults the lifecycle", "[lifecycle]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);
	start_active(lifecycle, environment);
	environment.step_status = Status::error(StatusCode::RUNTIME_FAILURE, "step failed");

	EnvironmentState output;
	const AgentState sentinel = make_agent_state(3, -1.0f, false, false, "sentinel");
	output.agents.emplace("sentinel", sentinel);
	const Status result = lifecycle.step({ { "agent_0", make_point(0) } }, output);

	CHECK_FALSE(result.is_ok());
	CHECK(lifecycle.get_state() == LifecycleState::FAULTED);
	CHECK(environment.step_calls == 1);
	REQUIRE(output.agents.size() == 1);
	require_agent_state(output.agents.at("sentinel"), sentinel);
}

// A faulted lifecycle rejects work but can still be closed.
TEST_CASE("faulted lifecycle rejects operations until closed", "[lifecycle]") {
	testing::FakeEnvironment environment;
	environment.initialize_status = Status::error(StatusCode::RUNTIME_FAILURE, "initialize failed");
	EnvironmentLifecycle lifecycle(0, environment);
	const Status initial_initialize_result = lifecycle.initialize();
	REQUIRE_FALSE(initial_initialize_result.is_ok());
	REQUIRE(lifecycle.get_state() == LifecycleState::FAULTED);

	InitialEnvironmentState initial_state;
	EnvironmentState state;
	const Status initialize_result = lifecycle.initialize();
	const Status reset_result = lifecycle.reset({}, initial_state);
	const Status step_result = lifecycle.step({}, state);
	CHECK_FALSE(initialize_result.is_ok());
	CHECK_FALSE(reset_result.is_ok());
	CHECK_FALSE(step_result.is_ok());
	CHECK(environment.initialize_calls == 1);
	CHECK(environment.reset_calls == 0);
	CHECK(environment.step_calls == 0);

	const Status close_result = lifecycle.close();
	CHECK(close_result.is_ok());
	CHECK(lifecycle.get_state() == LifecycleState::CLOSED);
}

// Close succeeds from every lifecycle state.
TEST_CASE("close is legal from every state", "[lifecycle]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);

	SECTION("uninitialized") {
		const Status close_result = lifecycle.close();
		CHECK(close_result.is_ok());
	}

	SECTION("active") {
		start_active(lifecycle, environment);
		const Status close_result = lifecycle.close();
		CHECK(close_result.is_ok());
	}

	SECTION("reset pending") {
		initialize_lifecycle(lifecycle, environment);
		const Status close_result = lifecycle.close();
		CHECK(close_result.is_ok());
	}

	SECTION("complete") {
		start_active(lifecycle, environment);
		environment.step_states = {
			{ "agent_0", make_agent_state(1, 1.0f, true) },
		};
		EnvironmentState state;
		const Status step_result = lifecycle.step({ { "agent_0", make_point(1) } }, state);
		REQUIRE(step_result.is_ok());
		REQUIRE(lifecycle.get_state() == LifecycleState::COMPLETE);
		const Status close_result = lifecycle.close();
		CHECK(close_result.is_ok());
	}

	SECTION("faulted") {
		environment.initialize_status = Status::error(StatusCode::RUNTIME_FAILURE, "initialize failed");
		const Status initialize_result = lifecycle.initialize();
		REQUIRE_FALSE(initialize_result.is_ok());
		const Status close_result = lifecycle.close();
		CHECK(close_result.is_ok());
	}

	CHECK(lifecycle.get_state() == LifecycleState::CLOSED);
}

// Repeated close calls succeed without changing the closed state.
TEST_CASE("close is idempotent", "[lifecycle]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);
	const Status first_result = lifecycle.close();
	REQUIRE(first_result.is_ok());

	const Status second_result = lifecycle.close();

	CHECK(second_result.is_ok());
	CHECK(lifecycle.get_state() == LifecycleState::CLOSED);
}

// A closed lifecycle rejects work without calling the environment.
TEST_CASE("closed lifecycle rejects further operations", "[lifecycle]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);
	const Status close_result = lifecycle.close();
	REQUIRE(close_result.is_ok());

	InitialEnvironmentState initial_state;
	EnvironmentState state;
	const Status initialize_result = lifecycle.initialize();
	const Status reset_result = lifecycle.reset({}, initial_state);
	const Status step_result = lifecycle.step({}, state);
	CHECK_FALSE(initialize_result.is_ok());
	CHECK_FALSE(reset_result.is_ok());
	CHECK_FALSE(step_result.is_ok());
	CHECK(environment.initialize_calls == 0);
	CHECK(environment.reset_calls == 0);
	CHECK(environment.step_calls == 0);
	CHECK(lifecycle.get_state() == LifecycleState::CLOSED);
}

// Rejected operations after close leave their outputs unchanged.
TEST_CASE("closed lifecycle preserves output values", "[lifecycle]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);
	const Status close_result = lifecycle.close();
	REQUIRE(close_result.is_ok());

	InitialEnvironmentState initial_state;
	const InitialAgentState initial_sentinel = make_initial_state(3, "sentinel");
	initial_state.agents.emplace("sentinel", initial_sentinel);
	const Status reset_result = lifecycle.reset({}, initial_state);
	CHECK_FALSE(reset_result.is_ok());
	REQUIRE(initial_state.agents.size() == 1);
	require_initial_state(initial_state.agents.at("sentinel"), initial_sentinel);

	EnvironmentState state;
	const AgentState state_sentinel = make_agent_state(3, -1.0f, false, false, "sentinel");
	state.agents.emplace("sentinel", state_sentinel);
	const Status step_result = lifecycle.step({}, state);
	CHECK_FALSE(step_result.is_ok());
	REQUIRE(state.agents.size() == 1);
	require_agent_state(state.agents.at("sentinel"), state_sentinel);
}

// Reset clears cached state and the step count between episodes.
TEST_CASE("reset starts a second episode without leaking state or step count", "[lifecycle]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment, 2);
	start_active(lifecycle, environment);

	const AgentState first_episode_state = make_agent_state(1, 5.0f, false, false, "first");
	environment.step_states = {
		{ "agent_0", first_episode_state },
	};
	EnvironmentState first_result;
	const Status first_step_result = lifecycle.step({ { "agent_0", make_point(1) } }, first_result);
	REQUIRE(first_step_result.is_ok());

	const InitialAgentState second_initial = make_initial_state(3, "second");
	environment.reset_states = {
		{ "agent_0", second_initial },
	};
	InitialEnvironmentState second_reset;
	const Status reset_result = lifecycle.reset({}, second_reset);
	REQUIRE(reset_result.is_ok());
	require_initial_state(second_reset.agents.at("agent_0"), second_initial);

	const AgentState second_episode_state = make_agent_state(2, 1.0f, false, false, "second");
	environment.step_states = {
		{ "agent_0", second_episode_state },
	};
	EnvironmentState second_result;
	const Status second_step_result = lifecycle.step({ { "agent_0", make_point(0) } }, second_result);
	REQUIRE(second_step_result.is_ok());

	CHECK(lifecycle.get_state() == LifecycleState::ACTIVE);
	REQUIRE(second_result.agents.size() == 1);
	require_agent_state(second_result.agents.at("agent_0"), second_episode_state);
	CHECK_FALSE(second_result.agents.at("agent_0").truncated);
}

TEST_CASE("an empty environment definition faults initialization", "[lifecycle][regression]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);
	CHECK(lifecycle.initialize().get_code() == StatusCode::INVALID_DATA);
	CHECK(lifecycle.get_state() == LifecycleState::FAULTED);
}

TEST_CASE("invalid action membership preserves lifecycle and output before calling environment", "[lifecycle][regression]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment, 1);
	start_active(lifecycle, environment, { "agent_0", "agent_1" });
	std::map<AgentId, Point> actions{
		{ "agent_0", make_point(0) },
		{ "agent_1", make_point(0) },
	};
	SECTION("empty actions cannot complete the episode") {
		actions.clear();
	}
	SECTION("missing unfinished agent") {
		actions.erase("agent_1");
	}
	SECTION("unknown agent") {
		actions.emplace("unknown", make_point(0));
	}
	EnvironmentState output;
	output.agents.emplace("sentinel", make_agent_state(3, -1.0f));
	CHECK(lifecycle.step(actions, output).get_code() == StatusCode::INVALID_ARGUMENT);
	CHECK(lifecycle.get_state() == LifecycleState::ACTIVE);
	CHECK(environment.step_calls == 0);
	REQUIRE(output.agents.size() == 1);
	CHECK(output.agents.at("sentinel").reward == -1.0f);
	environment.step_states = {
		{ "agent_0", make_agent_state(0, 1.0f) },
		{ "agent_1", make_agent_state(0, 1.0f) },
	};
	REQUIRE(lifecycle.step({ { "agent_0", make_point(0) }, { "agent_1", make_point(0) } }, output).is_ok());
	CHECK(output.agents.at("agent_0").truncated);
}

TEST_CASE("malformed successful environment responses fault without publishing partial state", "[lifecycle][regression]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);
	start_active(lifecycle, environment, { "agent_0", "agent_1" });
	SECTION("reset misses an agent") {
		environment.reset_states.erase("agent_1");
		InitialEnvironmentState output;
		output.agents.emplace("sentinel", make_initial_state(3, "sentinel"));
		CHECK(lifecycle.reset({}, output).get_code() == StatusCode::INVALID_DATA);
		REQUIRE(output.agents.size() == 1);
		CHECK(output.agents.at("sentinel").observation == make_point(3));
	}
	SECTION("reset substitutes an unknown agent") {
		environment.reset_states.erase("agent_1");
		environment.reset_states.emplace("unknown", make_initial_state(0, "invalid"));
		InitialEnvironmentState output;
		CHECK(lifecycle.reset({}, output).get_code() == StatusCode::INVALID_DATA);
		CHECK(output.agents.empty());
	}
	SECTION("step returns empty or partial results") {
		SECTION("empty") {}
		SECTION("partial") {
			environment.step_states.emplace("agent_0", make_agent_state(1, 0.0f, true));
		}
		SECTION("unknown agent with matching count") {
			environment.step_states.emplace("agent_0", make_agent_state(1, 0.0f, true));
			environment.step_states.emplace("unknown", make_agent_state(1, 0.0f, true));
		}
		EnvironmentState output;
		output.agents.emplace("sentinel", make_agent_state(3, -1.0f));
		CHECK(lifecycle.step({ { "agent_0", make_point(0) }, { "agent_1", make_point(0) } }, output).get_code() == StatusCode::INVALID_DATA);
		REQUIRE(output.agents.size() == 1);
		CHECK(output.agents.at("sentinel").reward == -1.0f);
	}
	CHECK(lifecycle.get_state() == LifecycleState::FAULTED);
}

TEST_CASE("finished agents cannot be acted on or overwrite their final snapshot", "[lifecycle][regression]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);
	start_active(lifecycle, environment, { "agent_0", "agent_1" });
	const AgentState terminal = make_agent_state(1, 2.0f, true);
	environment.step_states = {
		{ "agent_0", terminal },
		{ "agent_1", make_agent_state(2, 0.0f) },
	};
	EnvironmentState output;
	REQUIRE(lifecycle.step({ { "agent_0", make_point(0) }, { "agent_1", make_point(0) } }, output).is_ok());
	CHECK(lifecycle.step({ { "agent_0", make_point(0) }, { "agent_1", make_point(0) } }, output).get_code() == StatusCode::INVALID_ARGUMENT);
	CHECK(environment.step_calls == 1);
	environment.step_states.erase("agent_0");
	environment.step_states.at("agent_1") = make_agent_state(3, 1.0f, true);
	REQUIRE(lifecycle.step({ { "agent_1", make_point(0) } }, output).is_ok());
	require_agent_state(output.agents.at("agent_0"), terminal);
	CHECK(lifecycle.get_state() == LifecycleState::COMPLETE);
}

TEST_CASE("lifecycle forwards point contents without space validation", "[lifecycle]") {
	testing::FakeEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);
	start_active(lifecycle, environment);
	environment.step_states.emplace("agent_0", make_agent_state(999, 1.0f));
	EnvironmentState output;
	REQUIRE(lifecycle.step({ { "agent_0", make_point(999) } }, output).is_ok());
	CHECK(environment.last_actions.at("agent_0") == make_point(999));
	CHECK(output.agents.at("agent_0").observation == make_point(999));
}
