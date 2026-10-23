// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "core/environment/environment.h"
#include "support/grid_world_environment.h"

#include <catch_amalgamated.hpp>

#include <map>
#include <vector>

using namespace schola;
using testing::GridAgent;
using testing::GridWorldEnvironment;

namespace {

const AgentId RUNNER = GridWorldEnvironment::AGENT_ID;

Point move_action(int32_t p_direction) {
	return Point(DiscretePoint{ p_direction });
}

// Pre-filled out-param, so a test can tell whether a failed call touched it.
template <typename T>
std::map<AgentId, T> sentinel_map() {
	return { { "sentinel", T{} } };
}

void initialize(GridWorldEnvironment &p_environment) {
	std::map<AgentId, InteractionDefinition> agents;
	REQUIRE(p_environment.initialize(agents).is_ok());
}

Point reset(GridWorldEnvironment &p_environment, ResetSettings p_settings = {}) {
	std::map<AgentId, InitialAgentState> states;
	REQUIRE(p_environment.reset(p_settings, states).is_ok());
	REQUIRE(states.size() == 1);
	return states.at(RUNNER).observation;
}

Point observe_directly(GridWorldEnvironment &p_environment) {
	std::vector<Agent *> agents;
	REQUIRE(p_environment.get_agents(agents).is_ok());
	Point observation;
	REQUIRE(agents.at(0)->observe(observation).is_ok());
	return observation;
}

// Reads the [x, y, dx, dy] observation and moves toward the goal.
Point greedy_action(const Point &p_observation) {
	const BoxPoint *box = p_observation.get_if<BoxPoint>();
	REQUIRE(box != nullptr);
	if (box->values[2] > 0.0f) {
		return move_action(GridAgent::RIGHT);
	}
	return move_action(GridAgent::UP);
}

} // namespace

TEST_CASE("initialize declares one agent with valid spaces", "[environment]") {
	GridWorldEnvironment environment;
	std::map<AgentId, InteractionDefinition> agents;

	REQUIRE(environment.initialize(agents).is_ok());

	REQUIRE(agents.size() == 1);
	const InteractionDefinition &definition = agents.at(RUNNER);
	CHECK(definition.observation_space.check_definition().is_ok());
	CHECK(definition.action_space.check_definition().is_ok());
	CHECK(definition.observation_space.get_flattened_size() == 4);
	REQUIRE(definition.action_space.get_if<DiscreteSpace>() != nullptr);
	CHECK(definition.action_space.get_if<DiscreteSpace>()->n == 4);
}

TEST_CASE("agent IDs and spaces are the same for every instance", "[environment]") {
	GridWorldEnvironment first;
	GridWorldEnvironment second;
	std::map<AgentId, InteractionDefinition> first_agents;
	std::map<AgentId, InteractionDefinition> second_agents;

	REQUIRE(first.initialize(first_agents).is_ok());
	REQUIRE(second.initialize(second_agents).is_ok());

	REQUIRE(first_agents.size() == second_agents.size());
	for (const auto &[agent_id, definition] : first_agents) {
		REQUIRE(second_agents.count(agent_id) == 1);
		CHECK(definition.observation_space == second_agents.at(agent_id).observation_space);
		CHECK(definition.action_space == second_agents.at(agent_id).action_space);
	}
}

TEST_CASE("initialize succeeds only once and a repeat leaves the output untouched", "[environment]") {
	GridWorldEnvironment environment;
	initialize(environment);
	auto agents = sentinel_map<InteractionDefinition>();

	const Status result = environment.initialize(agents);

	CHECK(result.get_code() == StatusCode::FAILED_PRECONDITION);
	REQUIRE(agents.size() == 1);
	CHECK(agents.count("sentinel") == 1);
}

TEST_CASE("get_agents returns one agent per declared definition", "[environment]") {
	GridWorldEnvironment environment;
	std::map<AgentId, InteractionDefinition> definitions;
	REQUIRE(environment.initialize(definitions).is_ok());
	std::vector<Agent *> agents;

	REQUIRE(environment.get_agents(agents).is_ok());

	REQUIRE(agents.size() == definitions.size());
	REQUIRE(agents[0] != nullptr);
	InteractionDefinition defined;
	REQUIRE(agents[0]->define(defined).is_ok());
	CHECK(defined.observation_space == definitions.at(RUNNER).observation_space);
	CHECK(defined.action_space == definitions.at(RUNNER).action_space);
}

TEST_CASE("calls before initialize fail and leave outputs untouched", "[environment]") {
	GridWorldEnvironment environment;

	auto reset_states = sentinel_map<InitialAgentState>();
	CHECK(environment.reset({}, reset_states).get_code() == StatusCode::FAILED_PRECONDITION);
	CHECK(reset_states.count("sentinel") == 1);

	std::vector<Agent *> agents;
	CHECK(environment.get_agents(agents).get_code() == StatusCode::FAILED_PRECONDITION);
	CHECK(agents.empty());
}

TEST_CASE("step before reset fails and leaves the output untouched", "[environment]") {
	GridWorldEnvironment environment;
	initialize(environment);
	auto states = sentinel_map<AgentState>();

	const Status result = environment.step({ { RUNNER, move_action(GridAgent::UP) } }, states);

	CHECK(result.get_code() == StatusCode::FAILED_PRECONDITION);
	REQUIRE(states.size() == 1);
	CHECK(states.count("sentinel") == 1);
}

TEST_CASE("the same seed gives the same start state", "[environment]") {
	GridWorldEnvironment first;
	GridWorldEnvironment second;
	initialize(first);
	initialize(second);

	ResetSettings settings;
	settings.seed = 42;

	CHECK(reset(first, settings) == reset(second, settings));
}

TEST_CASE("an unseeded reset continues the seeded random stream", "[environment]") {
	ResetSettings seeded;
	seeded.seed = 7;

	// Two environments that take the same path draw the same sequence.
	GridWorldEnvironment first;
	GridWorldEnvironment second;
	initialize(first);
	initialize(second);
	const Point seeded_start = reset(first, seeded);
	reset(second, seeded);

	std::vector<Point> first_starts;
	std::vector<Point> second_starts;
	for (int i = 0; i < 8; i++) {
		first_starts.push_back(reset(first));
		second_starts.push_back(reset(second));
	}
	CHECK(first_starts == second_starts);

	// Unseeded resets keep drawing instead of restarting from the seed.
	bool any_differs = false;
	for (const Point &start : first_starts) {
		any_differs = any_differs || start != seeded_start;
	}
	CHECK(any_differs);
}

TEST_CASE("step reports reward and terminates at the goal without truncating", "[environment]") {
	GridWorldEnvironment environment;
	initialize(environment);
	ResetSettings settings;
	settings.seed = 3;
	Point observation = reset(environment, settings);

	AgentState last;
	int steps = 0;
	do {
		std::map<AgentId, AgentState> states;
		REQUIRE(environment.step({ { RUNNER, greedy_action(observation) } }, states).is_ok());
		REQUIRE(states.size() == 1);
		last = states.at(RUNNER);
		observation = last.observation;
		CHECK_FALSE(last.truncated);
		if (!last.terminated) {
			CHECK(last.reward == GridWorldEnvironment::STEP_REWARD);
		}
		steps++;
	} while (!last.terminated && steps < 2 * GridAgent::SIZE);

	CHECK(last.terminated);
	CHECK_FALSE(last.truncated);
	CHECK(last.reward == GridWorldEnvironment::GOAL_REWARD);
	CHECK(observation == Point(BoxPoint{ { 4.0f, 4.0f, 0.0f, 0.0f } }));
}

TEST_CASE("step after the episode ended fails until reset", "[environment]") {
	GridWorldEnvironment environment;
	initialize(environment);
	Point observation = reset(environment);
	bool terminated = false;
	while (!terminated) {
		std::map<AgentId, AgentState> states;
		REQUIRE(environment.step({ { RUNNER, greedy_action(observation) } }, states).is_ok());
		observation = states.at(RUNNER).observation;
		terminated = states.at(RUNNER).terminated;
	}

	auto states = sentinel_map<AgentState>();
	CHECK(environment.step({ { RUNNER, move_action(GridAgent::DOWN) } }, states).get_code() == StatusCode::FAILED_PRECONDITION);
	CHECK(states.count("sentinel") == 1);

	reset(environment);
	std::map<AgentId, AgentState> after_reset;
	CHECK(environment.step({ { RUNNER, move_action(GridAgent::DOWN) } }, after_reset).is_ok());
}

TEST_CASE("an unknown agent ID is rejected and nothing moves", "[environment]") {
	GridWorldEnvironment environment;
	initialize(environment);
	const Point start = reset(environment);
	auto states = sentinel_map<AgentState>();

	const Status result = environment.step({ { RUNNER, move_action(GridAgent::RIGHT) }, { "ghost", move_action(GridAgent::RIGHT) } }, states);

	CHECK(result.get_code() == StatusCode::NOT_FOUND);
	CHECK(states.count("sentinel") == 1);
	CHECK(observe_directly(environment) == start);
}

TEST_CASE("an action outside the action space is rejected and nothing moves", "[environment]") {
	GridWorldEnvironment environment;
	initialize(environment);
	const Point start = reset(environment);

	SECTION("out-of-range discrete value") {
		auto states = sentinel_map<AgentState>();
		CHECK(environment.step({ { RUNNER, move_action(7) } }, states).get_code() == StatusCode::INVALID_ARGUMENT);
		CHECK(states.count("sentinel") == 1);
	}

	SECTION("wrong point kind") {
		auto states = sentinel_map<AgentState>();
		CHECK(environment.step({ { RUNNER, Point(BoxPoint{ { 1.0f } }) } }, states).get_code() == StatusCode::INVALID_ARGUMENT);
		CHECK(states.count("sentinel") == 1);
	}

	CHECK(observe_directly(environment) == start);
}

TEST_CASE("the agent itself rejects a mismatched action", "[environment]") {
	GridAgent agent;
	agent.place(2, 2);

	CHECK(agent.act(move_action(-1)).get_code() == StatusCode::INVALID_ARGUMENT);
	CHECK(agent.get_x() == 2);
	CHECK(agent.get_y() == 2);
}

TEST_CASE("moves are clamped to the grid", "[environment]") {
	GridAgent agent;
	agent.place(0, 0);

	REQUIRE(agent.act(move_action(GridAgent::LEFT)).is_ok());
	REQUIRE(agent.act(move_action(GridAgent::DOWN)).is_ok());

	CHECK(agent.get_x() == 0);
	CHECK(agent.get_y() == 0);
}

TEST_CASE("an empty action map is a no-op step", "[environment]") {
	GridWorldEnvironment environment;
	initialize(environment);
	const Point start = reset(environment);
	std::map<AgentId, AgentState> states;

	REQUIRE(environment.step({}, states).is_ok());

	CHECK(states.empty());
	CHECK(observe_directly(environment) == start);
}
