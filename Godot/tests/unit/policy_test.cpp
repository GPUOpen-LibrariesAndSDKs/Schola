// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "core/policy/policy.h"
#include "core/policy/policy_state.h"
#include "support/test_policy.h"

#include <catch_amalgamated.hpp>

#include <memory>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using schola::BoxPoint;
using schola::EmptyPolicyState;
using schola::Point;
using schola::PolicyState;
using schola::Status;
using schola::StatusCode;
using schola::testing::CounterState;
using schola::testing::FakePolicy;

namespace {

Point box(float p_value) {
	return Point(BoxPoint{ { p_value } });
}

float box_value(const Point &p_point) {
	return p_point.get_if<BoxPoint>()->values.at(0);
}

} // namespace

TEST_CASE("EmptyPolicyState copies only another EmptyPolicyState", "[policy]") {
	EmptyPolicyState state;
	EmptyPolicyState other;
	REQUIRE(state.copy_from(other).is_ok());

	CounterState counter;
	Status status = state.copy_from(counter);
	REQUIRE(status.get_code() == StatusCode::INVALID_ARGUMENT);
	REQUIRE_THAT(status.get_message(), ContainsSubstring("CounterState"));
}

TEST_CASE("the default initial state is an EmptyPolicyState", "[policy]") {
	FakePolicy policy;
	std::unique_ptr<PolicyState> state;
	REQUIRE(policy.create_initial_state(state).is_ok());
	REQUIRE(dynamic_cast<EmptyPolicyState *>(state.get()) != nullptr);
}

TEST_CASE("the default batched_think thinks once per entry, in order", "[policy]") {
	FakePolicy policy;
	policy.action = box(7.0f);
	std::vector<Point> observations = { box(1.0f), box(2.0f), box(3.0f) };
	std::vector<CounterState> states(3);
	std::vector<Point> actions(3, box(0.0f));

	Status status = policy.batched_think({ &observations[0], &observations[1], &observations[2] },
			{ &states[0], &states[1], &states[2] }, { &actions[0], &actions[1], &actions[2] });

	REQUIRE(status.is_ok());
	REQUIRE(policy.observations.size() == 3);
	for (size_t i = 0; i < 3; ++i) {
		REQUIRE(box_value(policy.observations[i]) == box_value(observations[i]));
		REQUIRE(policy.states[i] == &states[i]);
		REQUIRE(states[i].count == 1);
		REQUIRE(box_value(actions[i]) == 7.0f);
	}
}

TEST_CASE("batched_think rejects bad input before thinking", "[policy]") {
	FakePolicy policy;
	Point observation = box(1.0f);
	EmptyPolicyState state;
	Point action = box(0.0f);

	SECTION("mismatched sizes") {
		Status status = policy.batched_think({ &observation }, { &state, &state }, { &action });
		REQUIRE(status.get_code() == StatusCode::INVALID_ARGUMENT);
	}
	SECTION("a null entry") {
		Status status = policy.batched_think({ &observation, nullptr }, { &state, &state }, { &action, &action });
		REQUIRE(status.get_code() == StatusCode::INVALID_ARGUMENT);
		REQUIRE_THAT(status.get_message(), ContainsSubstring("entry 1"));
	}
	REQUIRE(policy.observations.empty());
}

TEST_CASE("batched_think stops at the first failing entry and names it", "[policy]") {
	FakePolicy policy;
	policy.fail_at_call = 1;
	policy.failure = Status::error(StatusCode::RUNTIME_FAILURE, "model crashed");
	std::vector<Point> observations = { box(1.0f), box(2.0f), box(3.0f) };
	std::vector<CounterState> states(3);
	std::vector<Point> actions(3, box(0.0f));

	Status status = policy.batched_think({ &observations[0], &observations[1], &observations[2] },
			{ &states[0], &states[1], &states[2] }, { &actions[0], &actions[1], &actions[2] });

	REQUIRE(status.get_code() == StatusCode::RUNTIME_FAILURE);
	REQUIRE_THAT(status.get_message(), ContainsSubstring("entry 1: model crashed"));
	REQUIRE(states[0].count == 1); // Entry 0 succeeded before the failure.
	REQUIRE(states[1].count == 0); // The failed entry keeps its state.
	REQUIRE(policy.observations.size() == 2); // Entry 2 was never reached.
}
