// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Compiles the public spaces headers together. Behavioral tests are added with the implementation.

#include "core/spaces/point.h"
#include "core/spaces/space.h"
#include "core/spaces/space_kind.h"

#include <type_traits>
#include <utility>

namespace schola {

static_assert(std::is_copy_constructible_v<Space> && std::is_copy_assignable_v<Space>);
static_assert(std::is_move_constructible_v<Space> && std::is_move_assignable_v<Space>);
static_assert(std::is_copy_constructible_v<Point> && std::is_copy_assignable_v<Point>);
static_assert(std::is_move_constructible_v<Point> && std::is_move_assignable_v<Point>);
static_assert(std::is_same_v<decltype(std::declval<const Space &>().validate(std::declval<const Point &>())), Status>);
static_assert(std::is_same_v<decltype(std::declval<const Space &>().check_definition()), Status>);

} // namespace schola
