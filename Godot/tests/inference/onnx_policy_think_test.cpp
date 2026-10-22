// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "core/environment/interaction_definition.h"
#include "core/policy/policy_state.h"
#include "core/spaces/space.h"
#include "inference/onnx_policy.h"
#include "inference/onnx_policy_state.h"
#include "test_fixtures_path.h"

#include <catch_amalgamated.hpp>

#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using namespace schola;

namespace {

Status load(OnnxPolicy &r_policy, const std::string &p_name) {
	std::ifstream file(std::string(SCHOLA_TEST_FIXTURES_DIR) + "/" + p_name, std::ios::binary);
	REQUIRE(file.is_open());
	std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	return r_policy.load_model(bytes.data(), bytes.size(), p_name);
}

Space box(int32_t p_size) {
	return Space(BoxSpace{ std::vector<float>(p_size, -10.0f), std::vector<float>(p_size, 10.0f), { p_size } });
}

InteractionDefinition definition(Space p_observation, Space p_action) {
	InteractionDefinition result;
	result.observation_space = std::move(p_observation);
	result.action_space = std::move(p_action);
	return result;
}

struct Fixture {
	OnnxPolicy policy;
	std::unique_ptr<PolicyState> state;
	Point action;

	Fixture(const std::string &p_model, const InteractionDefinition &p_definition, int32_t p_sequence_length = 1) {
		REQUIRE(load(policy, p_model).is_ok());
		REQUIRE(policy.set_max_state_sequence_length(p_sequence_length).is_ok());
		Status status = policy.init(p_definition);
		INFO(status.get_message());
		REQUIRE(status.is_ok());
		REQUIRE(policy.create_initial_state(state).is_ok());
		action = p_definition.action_space.make_point();
	}

	Status think(const Point &p_observation) { return policy.think(p_observation, *state, action); }
};

std::vector<float> box_values(const Point &p_point) {
	return p_point.get_if<BoxPoint>()->values;
}

} // namespace

TEST_CASE("think runs a Box model", "[inference]") {
	Fixture fixture("box_scale.onnx", definition(box(3), box(3)));
	REQUIRE(fixture.policy.is_ready());
	REQUIRE(dynamic_cast<EmptyPolicyState *>(fixture.state.get()) != nullptr);

	REQUIRE(fixture.think(Point(BoxPoint{ { 1.0f, 2.0f, 3.0f } })).is_ok());
	REQUIRE(box_values(fixture.action) == std::vector<float>{ 2.0f, 4.0f, 6.0f });
}

TEST_CASE("think returns Discrete, MultiDiscrete, and MultiBinary actions", "[inference]") {
	SECTION("Discrete") {
		Fixture fixture("discrete.onnx", definition(box(4), Space(DiscreteSpace{ 4 })));
		REQUIRE(fixture.think(Point(BoxPoint{ { 0.0f, 5.0f, 1.0f, 2.0f } })).is_ok());
		REQUIRE(fixture.action.get_if<DiscretePoint>()->value == 1);
	}
	SECTION("MultiDiscrete") {
		Space space(MultiDiscreteSpace{ { 4, 4 } });
		Fixture fixture("multi_discrete.onnx", definition(space, space));
		REQUIRE(fixture.think(Point(MultiDiscretePoint{ { 3, 1 } })).is_ok());
		REQUIRE(fixture.action.get_if<MultiDiscretePoint>()->values == std::vector<int32_t>{ 3, 1 });
	}
	SECTION("MultiBinary") {
		Space space(MultiBinarySpace{ 3 });
		Fixture fixture("multi_binary.onnx", definition(space, space));
		REQUIRE(fixture.think(Point(MultiBinaryPoint{ { 1, 0, 1 } })).is_ok());
		REQUIRE(fixture.action.get_if<MultiBinaryPoint>()->values == std::vector<uint8_t>{ 1, 0, 1 });
	}
}

TEST_CASE("think matches Dict observations and actions by key", "[inference]") {
	SECTION("observation") {
		Space observation(DictSpace{ { { "b", Space(DiscreteSpace{ 5 }) }, { "a", box(2) } } });
		Fixture fixture("dict_obs.onnx", definition(observation, box(3)));
		DictPoint point{ { { "a", Point(BoxPoint{ { 1.0f, 2.0f } }) }, { "b", Point(DiscretePoint{ 4 }) } } };
		REQUIRE(fixture.think(Point(point)).is_ok());
		REQUIRE(box_values(fixture.action) == std::vector<float>{ 1.0f, 2.0f, 4.0f });
	}
	SECTION("action") {
		Space action(DictSpace{ { { "jump", Space(DiscreteSpace{ 2 }) }, { "move", box(2) } } });
		Fixture fixture("dict_action.onnx", definition(box(2), action));
		REQUIRE(fixture.think(Point(BoxPoint{ { 0.0f, 9.0f } })).is_ok());
		const DictPoint *dict = fixture.action.get_if<DictPoint>();
		REQUIRE(dict->entries[0].second.get_if<DiscretePoint>()->value == 1);
		REQUIRE(box_values(dict->entries[1].second) == std::vector<float>{ 0.0f, 9.0f });
	}
}

TEST_CASE("init rejects spaces that do not fit the model", "[inference]") {
	OnnxPolicy policy;
	REQUIRE(policy.init(definition(box(3), box(3))).get_code() == StatusCode::FAILED_PRECONDITION);
	REQUIRE(load(policy, "box_scale.onnx").is_ok());

	Status wrong_size = policy.init(definition(box(2), box(3)));
	REQUIRE(wrong_size.get_code() == StatusCode::INCOMPATIBLE);
	REQUIRE_THAT(wrong_size.get_message(), ContainsSubstring("input 'obs' has shape [?, 3], but the observation space needs [?, 2]"));

	Status wrong_kind = policy.init(definition(Space(DiscreteSpace{ 3 }), box(3)));
	REQUIRE(wrong_kind.get_code() == StatusCode::INCOMPATIBLE);
	REQUIRE_FALSE(policy.is_ready());

	REQUIRE(load(policy, "dict_obs.onnx").is_ok());
	Status missing_key = policy.init(definition(Space(DictSpace{ { { "a", box(2) } } }), box(3)));
	REQUIRE(missing_key.get_code() == StatusCode::INCOMPATIBLE);
	REQUIRE_THAT(missing_key.get_message(), ContainsSubstring("'b' has no matching observation"));
}

TEST_CASE("think keeps each state separate and leaves it unchanged on failure", "[inference]") {
	Fixture fixture("counter.onnx", definition(box(1), box(2)));
	const Point observation(BoxPoint{ { 0.0f } });

	for (float expected : { 0.0f, 1.0f, 2.0f }) {
		REQUIRE(fixture.think(observation).is_ok());
		REQUIRE(box_values(fixture.action) == std::vector<float>{ expected, expected });
	}

	std::unique_ptr<PolicyState> other;
	REQUIRE(fixture.policy.create_initial_state(other).is_ok());
	REQUIRE(fixture.policy.think(observation, *other, fixture.action).is_ok());
	REQUIRE(box_values(fixture.action) == std::vector<float>{ 0.0f, 0.0f });

	REQUIRE(fixture.think(Point(BoxPoint{ { 0.0f, 0.0f } })).get_code() == StatusCode::INVALID_ARGUMENT);
	REQUIRE(fixture.think(observation).is_ok());
	REQUIRE(box_values(fixture.action) == std::vector<float>{ 3.0f, 3.0f });

	fixture.state->reset();
	REQUIRE(fixture.think(observation).is_ok());
	REQUIRE(box_values(fixture.action) == std::vector<float>{ 0.0f, 0.0f });
}

TEST_CASE("think slides a state sequence of the configured length", "[inference]") {
	Fixture fixture("counter_seq.onnx", definition(box(1), box(2)), 3);
	const Point observation(BoxPoint{ { 0.0f } });

	for (float expected : { 0.0f, 1.0f, 2.0f, 3.0f }) {
		REQUIRE(fixture.think(observation).is_ok());
		REQUIRE(box_values(fixture.action) == std::vector<float>{ expected, expected });
	}
	const OnnxPolicyState &state = static_cast<const OnnxPolicyState &>(*fixture.state);
	REQUIRE(state.to_string() == "OnnxPolicyState(state_in_h: 3x2)");
	REQUIRE(std::vector<float>(state.get_buffers()[0].values.begin(), state.get_buffers()[0].values.begin() + 6) == std::vector<float>{ 2, 2, 3, 3, 4, 4 });
	REQUIRE(fixture.policy.set_max_state_sequence_length(5).get_code() == StatusCode::FAILED_PRECONDITION);
}

TEST_CASE("think rejects a state from another policy", "[inference]") {
	Fixture first("counter.onnx", definition(box(1), box(2)));
	Fixture second("counter.onnx", definition(box(1), box(2)));
	const Point observation(BoxPoint{ { 0.0f } });

	Status status = first.policy.think(observation, *second.state, first.action);
	REQUIRE(status.get_code() == StatusCode::INVALID_ARGUMENT);
	REQUIRE_THAT(status.get_message(), ContainsSubstring("not created by this policy"));

	EmptyPolicyState empty;
	REQUIRE(first.policy.think(observation, empty, first.action).get_code() == StatusCode::INVALID_ARGUMENT);
}

TEST_CASE("think rejects a state created before the model was reloaded", "[inference]") {
	Fixture fixture("counter.onnx", definition(box(1), box(2)));
	REQUIRE(load(fixture.policy, "counter_seq.onnx").is_ok());
	REQUIRE(fixture.policy.set_max_state_sequence_length(3).is_ok());
	REQUIRE(fixture.policy.init(definition(box(1), box(2))).is_ok());

	Status status = fixture.think(Point(BoxPoint{ { 0.0f } }));
	REQUIRE(status.get_code() == StatusCode::INVALID_ARGUMENT);
	REQUIRE_THAT(status.get_message(), ContainsSubstring("create a new state after reloading"));
}

TEST_CASE("think rejects a Discrete action outside its space", "[inference]") {
	Fixture fixture("discrete.onnx", definition(box(4), Space(DiscreteSpace{ 2 })));
	Status status = fixture.think(Point(BoxPoint{ { 0.0f, 0.0f, 0.0f, 9.0f } }));
	REQUIRE(status.get_code() == StatusCode::RUNTIME_FAILURE);
	REQUIRE_THAT(status.get_message(), ContainsSubstring("returned 3 for action, outside [0, 2)"));
}

TEST_CASE("batched_think advances every entry's own state", "[inference]") {
	Fixture fixture("counter.onnx", definition(box(1), box(2)));
	std::unique_ptr<PolicyState> second;
	REQUIRE(fixture.policy.create_initial_state(second).is_ok());
	const Point observation(BoxPoint{ { 0.0f } });
	Point first_action = fixture.action;
	Point second_action = fixture.action;

	REQUIRE(fixture.think(observation).is_ok());
	REQUIRE(fixture.policy.batched_think({ &observation, &observation }, { fixture.state.get(), second.get() }, { &first_action, &second_action }).is_ok());

	REQUIRE(box_values(first_action) == std::vector<float>{ 1.0f, 1.0f });
	REQUIRE(box_values(second_action) == std::vector<float>{ 0.0f, 0.0f });
}
