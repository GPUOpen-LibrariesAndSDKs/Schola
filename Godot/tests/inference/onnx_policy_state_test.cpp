// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "core/policy/policy_state.h"
#include "inference/onnx_policy.h"
#include "inference/onnx_policy_state.h"

#include <catch_amalgamated.hpp>

#include <algorithm>
#include <memory>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using schola::EmptyPolicyState;
using schola::OnnxPolicy;
using schola::OnnxPolicyState;
using schola::PolicyState;
using schola::Status;
using schola::StatusCode;

namespace {

std::vector<float> sequence_of(const OnnxPolicyState::Buffer &p_buffer) {
	return std::vector<float>(p_buffer.values.begin(), p_buffer.values.begin() + p_buffer.sequence_length * p_buffer.state_size);
}

void write_pending(OnnxPolicyState &r_state, std::vector<float> p_values) {
	OnnxPolicyState::Buffer &buffer = r_state.get_buffers().at(0);
	std::copy(p_values.begin(), p_values.end(), buffer.pending());
}

} // namespace

TEST_CASE("a new state is zeroed and sized from its layouts", "[inference]") {
	OnnxPolicyState state(nullptr, { { "state_in_h", 3, 2 }, { "state_in_c", 1, 4 } });

	REQUIRE(state.get_buffers().size() == 2);
	REQUIRE(state.get_buffers()[0].values == std::vector<float>(8, 0.0f));
	REQUIRE(state.get_buffers()[1].values == std::vector<float>(8, 0.0f));
	REQUIRE(state.to_string() == "OnnxPolicyState(state_in_h: 3x2, state_in_c: 1x4)");
}

TEST_CASE("without a sequence, notify_state_update replaces the state", "[inference]") {
	OnnxPolicyState state(nullptr, { { "state_in_h", 1, 2 } });
	write_pending(state, { 1.0f, 2.0f });

	REQUIRE(sequence_of(state.get_buffers()[0]) == std::vector<float>{ 0.0f, 0.0f });
	REQUIRE(state.notify_state_update().is_ok());
	REQUIRE(sequence_of(state.get_buffers()[0]) == std::vector<float>{ 1.0f, 2.0f });
}

TEST_CASE("with a sequence, notify_state_update slides the window", "[inference]") {
	OnnxPolicyState state(nullptr, { { "state_in_h", 3, 2 } });

	write_pending(state, { 1.0f, 1.0f });
	REQUIRE(state.notify_state_update().is_ok());
	REQUIRE(sequence_of(state.get_buffers()[0]) == std::vector<float>{ 0, 0, 0, 0, 1, 1 });

	write_pending(state, { 2.0f, 2.0f });
	REQUIRE(state.notify_state_update().is_ok());
	write_pending(state, { 3.0f, 3.0f });
	REQUIRE(state.notify_state_update().is_ok());
	REQUIRE(sequence_of(state.get_buffers()[0]) == std::vector<float>{ 1, 1, 2, 2, 3, 3 });

	write_pending(state, { 4.0f, 4.0f });
	REQUIRE(state.notify_state_update().is_ok());
	REQUIRE(sequence_of(state.get_buffers()[0]) == std::vector<float>{ 2, 2, 3, 3, 4, 4 });
}

TEST_CASE("reset zeros the whole state", "[inference]") {
	OnnxPolicyState state(nullptr, { { "state_in_h", 2, 2 } });
	write_pending(state, { 5.0f, 5.0f });
	REQUIRE(state.notify_state_update().is_ok());

	state.reset();
	REQUIRE(state.get_buffers()[0].values == std::vector<float>(6, 0.0f));
}

TEST_CASE("copy_from copies matching states in place", "[inference]") {
	OnnxPolicyState source(nullptr, { { "state_in_h", 2, 2 } });
	write_pending(source, { 7.0f, 8.0f });
	REQUIRE(source.notify_state_update().is_ok());
	OnnxPolicyState target(nullptr, { { "state_in_h", 2, 2 } });
	const float *data_before = target.get_buffers()[0].values.data();

	REQUIRE(target.copy_from(source).is_ok());
	REQUIRE(target.get_buffers()[0].values == source.get_buffers()[0].values);
	REQUIRE(target.get_buffers()[0].values.data() == data_before);
}

TEST_CASE("copy_from rejects a different state", "[inference]") {
	OnnxPolicyState state(nullptr, { { "state_in_h", 2, 2 } });

	OnnxPolicyState other_size(nullptr, { { "state_in_h", 3, 2 } });
	REQUIRE(state.copy_from(other_size).get_code() == StatusCode::INVALID_ARGUMENT);

	EmptyPolicyState empty;
	Status status = state.copy_from(empty);
	REQUIRE(status.get_code() == StatusCode::INVALID_ARGUMENT);
	REQUIRE_THAT(status.get_message(), ContainsSubstring("EmptyPolicyState"));
}

TEST_CASE("create_initial_state needs an initialized policy", "[inference]") {
	OnnxPolicy policy;
	std::unique_ptr<PolicyState> state;
	REQUIRE(policy.create_initial_state(state).get_code() == StatusCode::FAILED_PRECONDITION);
	REQUIRE(state == nullptr);
}
