// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Compiles the public US7 inference headers together. Behavioral tests are added with the implementation.

#include "core/policy/policy.h"
#include "core/policy/simple_stepper.h"
#include "inference/onnx_policy.h"
#include "support/test_policy.h"

#include <type_traits>
#include <utility>

namespace schola {

static_assert(std::is_abstract_v<Policy>);
static_assert(std::is_base_of_v<Policy, OnnxPolicy>);
static_assert(std::is_base_of_v<Policy, testing::FakePolicy>);
static_assert(std::is_same_v<decltype(std::declval<Policy &>().think(0, std::declval<const Point &>(), std::declval<Point &>())), Status>);

} // namespace schola
