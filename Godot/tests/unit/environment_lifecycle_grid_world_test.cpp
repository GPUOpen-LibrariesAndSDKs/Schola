// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "core/lifecycle/environment_lifecycle.h"
#include "support/grid_world_environment.h"

#include <catch_amalgamated.hpp>

#include <map>

using namespace schola;
using testing::GridAgent;
using testing::GridWorldEnvironment;

namespace {

const AgentId RUNNER = GridWorldEnvironment::AGENT_ID;

Point move_action(int32_t p_direction) {
	return Point(DiscretePoint{ p_direction });
}

Point greedy_action(const Point &p_observation) {
	const BoxPoint *box = p_observation.get_if<BoxPoint>();
	REQUIRE(box != nullptr);
	if (box->values[2] > 0.0f) {
		return move_action(GridAgent::RIGHT);
	}
	return move_action(GridAgent::UP);
}

} // namespace

// US1's D2 completion check: the lifecycle identifies one agent and requests its
// declared spaces through the US1 contract alone, without referencing the D2 scene.
TEST_CASE("lifecycle identifies one agent and its declared spaces", "[lifecycle][environment]") {
	GridWorldEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);

	REQUIRE(lifecycle.initialize().is_ok());

	CHECK(lifecycle.get_state() == LifecycleState::RESET_PENDING);
	const EnvironmentDefinition &definition = lifecycle.get_definition();
	REQUIRE(definition.agents.size() == 1);
	REQUIRE(definition.agents.count(RUNNER) == 1);

	GridAgent reference;
	InteractionDefinition expected;
	REQUIRE(reference.define(expected).is_ok());
	CHECK(definition.agents.at(RUNNER).observation_space == expected.observation_space);
	CHECK(definition.agents.at(RUNNER).action_space == expected.action_space);
}

TEST_CASE("lifecycle runs a grid-world episode to the goal and resets", "[lifecycle][environment]") {
	GridWorldEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);
	REQUIRE(lifecycle.initialize().is_ok());

	ResetSettings settings;
	settings.seed = 11;
	InitialEnvironmentState initial;
	REQUIRE(lifecycle.reset(settings, initial).is_ok());
	REQUIRE(lifecycle.get_state() == LifecycleState::ACTIVE);
	Point observation = initial.agents.at(RUNNER).observation;

	EnvironmentState state;
	uint64_t steps = 0;
	while (lifecycle.get_state() == LifecycleState::ACTIVE && steps < static_cast<uint64_t>(2 * GridAgent::SIZE)) {
		REQUIRE(lifecycle.step({ { RUNNER, greedy_action(observation) } }, state).is_ok());
		observation = state.agents.at(RUNNER).observation;
		steps++;
	}

	CHECK(lifecycle.get_state() == LifecycleState::COMPLETE);
	CHECK(state.agents.at(RUNNER).terminated);
	CHECK_FALSE(state.agents.at(RUNNER).truncated);
	CHECK(state.agents.at(RUNNER).reward == GridWorldEnvironment::GOAL_REWARD);
	CHECK(lifecycle.get_episode_step_count() == steps);

	REQUIRE(lifecycle.reset({}, initial).is_ok());
	CHECK(lifecycle.get_state() == LifecycleState::ACTIVE);
	CHECK(lifecycle.get_episode_count() == 2);
	CHECK(lifecycle.get_episode_step_count() == 0);
}

TEST_CASE("lifecycle step limit truncates an episode the environment never ends", "[lifecycle][environment]") {
	GridWorldEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment, 3);
	REQUIRE(lifecycle.initialize().is_ok());
	InitialEnvironmentState initial;
	REQUIRE(lifecycle.reset({}, initial).is_ok());

	// Moving left only ever moves away from the goal, so only the limit can end it.
	EnvironmentState state;
	for (int i = 0; i < 3; i++) {
		REQUIRE(lifecycle.step({ { RUNNER, move_action(GridAgent::LEFT) } }, state).is_ok());
	}

	CHECK(lifecycle.get_state() == LifecycleState::COMPLETE);
	CHECK(state.agents.at(RUNNER).truncated);
	CHECK_FALSE(state.agents.at(RUNNER).terminated);
}

TEST_CASE("lifecycle surfaces a rejected action as a fault", "[lifecycle][environment]") {
	GridWorldEnvironment environment;
	EnvironmentLifecycle lifecycle(0, environment);
	REQUIRE(lifecycle.initialize().is_ok());
	InitialEnvironmentState initial;
	REQUIRE(lifecycle.reset({}, initial).is_ok());
	EnvironmentState state;

	const Status result = lifecycle.step({ { RUNNER, move_action(9) } }, state);

	CHECK(result.get_code() == StatusCode::INVALID_ARGUMENT);
	CHECK(lifecycle.get_state() == LifecycleState::FAULTED);
	CHECK(state.agents.empty());
}
