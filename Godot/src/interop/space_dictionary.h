// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include <godot_cpp/variant/dictionary.hpp>

namespace schola {
class Space;

// Converts a space into a gymnasium-style Dictionary; infinite Box bounds become null for JSON
godot::Dictionary space_to_dictionary(const Space &p_space);

} // namespace schola
