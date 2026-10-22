// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "core/environment/interaction_definition.h"
#include "core/policy/policy_state.h"
#include "core/spaces/space.h"
#include "inference/onnx_policy.h"
#include "test_fixtures_path.h"

#include <catch_amalgamated.hpp>

#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace schola;

namespace {

struct Expected {
	std::vector<float> observation;
	std::vector<float> action;
};

Expected read_expected(const std::string &p_name) {
	std::ifstream file(std::string(SCHOLA_TEST_FIXTURES_DIR) + "/" + p_name + ".expected.txt");
	REQUIRE(file.is_open());
	Expected expected;
	std::string line;
	while (std::getline(file, line)) {
		std::istringstream words(line);
		std::string label;
		words >> label;
		std::vector<float> &values = label == "observation" ? expected.observation : expected.action;
		for (float value; words >> value;) {
			values.push_back(value);
		}
	}
	return expected;
}

void load_and_init(OnnxPolicy &r_policy, const std::string &p_name, Space p_observation, Space p_action) {
	std::ifstream file(std::string(SCHOLA_TEST_FIXTURES_DIR) + "/" + p_name + ".onnx", std::ios::binary);
	REQUIRE(file.is_open());
	std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	REQUIRE(r_policy.load_model(bytes.data(), bytes.size(), p_name).is_ok());

	InteractionDefinition definition;
	definition.observation_space = std::move(p_observation);
	definition.action_space = std::move(p_action);
	Status status = r_policy.init(definition);
	INFO(status.get_message());
	REQUIRE(status.is_ok());
}

Space box(int32_t p_size) {
	return Space(BoxSpace{ std::vector<float>(p_size, -1.0f), std::vector<float>(p_size, 1.0f), { p_size } });
}

Point think(OnnxPolicy &r_policy, const Point &p_observation, const Space &p_action_space) {
	std::unique_ptr<PolicyState> state;
	REQUIRE(r_policy.create_initial_state(state).is_ok());
	Point action = p_action_space.make_point();
	Status status = r_policy.think(p_observation, *state, action);
	INFO(status.get_message());
	REQUIRE(status.is_ok());
	return action;
}

} // namespace

TEST_CASE("an exported PPO policy with a Discrete action matches Python", "[inference]") {
	const Expected expected = read_expected("schola_ppo_discrete");
	OnnxPolicy policy;
	const Space action_space(DiscreteSpace{ 3 });
	load_and_init(policy, "schola_ppo_discrete", box(4), action_space);

	Point action = think(policy, Point(BoxPoint{ expected.observation }), action_space);
	REQUIRE(action.get_if<DiscretePoint>()->value == static_cast<int32_t>(expected.action[0]));
}

TEST_CASE("an exported PPO policy with a Box action matches Python", "[inference]") {
	const Expected expected = read_expected("schola_ppo_box");
	OnnxPolicy policy;
	const Space action_space = box(2);
	load_and_init(policy, "schola_ppo_box", box(3), action_space);

	Point action = think(policy, Point(BoxPoint{ expected.observation }), action_space);
	const std::vector<float> &values = action.get_if<BoxPoint>()->values;
	REQUIRE(values.size() == expected.action.size());
	for (size_t i = 0; i < values.size(); ++i) {
		REQUIRE(values[i] == Catch::Approx(expected.action[i]).margin(1e-6));
	}
}

TEST_CASE("an exported PPO policy with a Dict observation matches Python", "[inference]") {
	const Expected expected = read_expected("schola_ppo_dict");
	OnnxPolicy policy;
	const Space action_space(MultiDiscreteSpace{ { 3, 2 } });
	load_and_init(policy, "schola_ppo_dict", Space(DictSpace{ { { "position", box(2) }, { "target", box(2) } } }), action_space);

	DictPoint observation{ {
			{ "position", Point(BoxPoint{ { expected.observation[0], expected.observation[1] } }) },
			{ "target", Point(BoxPoint{ { expected.observation[2], expected.observation[3] } }) },
	} };
	Point action = think(policy, Point(observation), action_space);
	REQUIRE(action.get_if<MultiDiscretePoint>()->values == std::vector<int32_t>{ static_cast<int32_t>(expected.action[0]), static_cast<int32_t>(expected.action[1]) });
}
