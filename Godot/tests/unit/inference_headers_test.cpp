// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Compiles the public US7 inference headers together. Behavioral tests are added with the implementation.

#include "core/policy/policy.h"
#include "core/policy/policy_state.h"
#include "core/stepper/simple_stepper.h"
#include "inference/onnx_policy.h"
#include "inference/onnx_policy_state.h"
#include "support/test_policy.h"

#include <cstdint>
#include <type_traits>
#include <utility>

namespace schola {

static_assert(std::is_abstract_v<Policy>);
static_assert(std::is_abstract_v<PolicyState>);
static_assert(std::is_base_of_v<PolicyState, EmptyPolicyState>);
static_assert(std::is_base_of_v<PolicyState, OnnxPolicyState>);
static_assert(std::is_base_of_v<Policy, OnnxPolicy>);
static_assert(std::is_base_of_v<Policy, testing::FakePolicy>);
static_assert(std::is_same_v<decltype(std::declval<Policy &>().think(std::declval<const Point &>(), std::declval<PolicyState &>(), std::declval<Point &>())), Status>);
static_assert(std::is_same_v<decltype(std::declval<SimpleStepper &>().add_agent(nullptr, std::declval<int32_t &>())), Status>);

} // namespace schola
