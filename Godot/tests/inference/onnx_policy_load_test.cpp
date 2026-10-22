// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "inference/onnx_policy.h"
#include "test_fixtures_path.h"

#include <catch_amalgamated.hpp>

#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using schola::OnnxPolicy;
using schola::Status;
using schola::StatusCode;

namespace {

std::vector<uint8_t> read_fixture(const std::string &p_name) {
	std::ifstream file(std::string(SCHOLA_TEST_FIXTURES_DIR) + "/" + p_name, std::ios::binary);
	REQUIRE(file.is_open());
	return std::vector<uint8_t>(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

Status load(OnnxPolicy &r_policy, const std::string &p_name) {
	std::vector<uint8_t> bytes = read_fixture(p_name);
	return r_policy.load_model(bytes.data(), bytes.size(), p_name);
}

} // namespace

TEST_CASE("load_model reads the model's input and output names", "[inference]") {
	OnnxPolicy policy;
	REQUIRE(load(policy, "box_scale.onnx").is_ok());
	REQUIRE(policy.get_input_names() == std::vector<std::string>{ "obs" });
	REQUIRE(policy.get_output_names() == std::vector<std::string>{ "action" });
	REQUIRE_FALSE(policy.is_recurrent());
	REQUIRE_FALSE(policy.is_ready());

	REQUIRE(load(policy, "dict_obs.onnx").is_ok());
	REQUIRE(policy.get_input_names() == std::vector<std::string>{ "a", "b" });
}

TEST_CASE("load_model detects recurrent state", "[inference]") {
	OnnxPolicy policy;
	REQUIRE(load(policy, "counter.onnx").is_ok());
	REQUIRE(policy.is_recurrent());

	REQUIRE(load(policy, "counter_seq.onnx").is_ok());
	REQUIRE(policy.is_recurrent());
}

TEST_CASE("load_model rejects data that is not a model", "[inference]") {
	OnnxPolicy policy;
	const std::vector<uint8_t> garbage = { 1, 2, 3, 4, 5 };
	Status status = policy.load_model(garbage.data(), garbage.size(), "garbage.onnx");
	REQUIRE(status.get_code() == StatusCode::INVALID_DATA);
	REQUIRE_THAT(status.get_message(), ContainsSubstring("garbage.onnx"));

	REQUIRE(policy.load_model(nullptr, 0, "empty.onnx").get_code() == StatusCode::INVALID_ARGUMENT);
}

TEST_CASE("load_model rejects invalid recurrent state", "[inference]") {
	OnnxPolicy policy;

	Status unpaired = load(policy, "bad_unpaired_state.onnx");
	REQUIRE(unpaired.get_code() == StatusCode::INCOMPATIBLE);
	REQUIRE_THAT(unpaired.get_message(), ContainsSubstring("1 state_in tensors but 0 state_out tensors"));

	Status not_float = load(policy, "bad_state_int.onnx");
	REQUIRE(not_float.get_code() == StatusCode::INCOMPATIBLE);
	REQUIRE_THAT(not_float.get_message(), ContainsSubstring("must be float"));
	REQUIRE(policy.get_input_names().empty());
}

TEST_CASE("loading another model replaces the previous one", "[inference]") {
	OnnxPolicy policy;
	REQUIRE(load(policy, "counter.onnx").is_ok());
	REQUIRE(load(policy, "box_scale.onnx").is_ok());
	REQUIRE_FALSE(policy.is_recurrent());
}

TEST_CASE("the state sequence length is set by hand", "[inference]") {
	OnnxPolicy policy;
	REQUIRE(policy.get_max_state_sequence_length() == 1);
	REQUIRE(policy.set_max_state_sequence_length(3).is_ok());
	REQUIRE(policy.get_max_state_sequence_length() == 3);
	REQUIRE(policy.set_max_state_sequence_length(0).get_code() == StatusCode::INVALID_ARGUMENT);
}
