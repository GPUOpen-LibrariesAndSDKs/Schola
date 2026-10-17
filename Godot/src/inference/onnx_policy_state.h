// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "core/policy/policy_state.h"

#include <string>

namespace schola {

// Unreal: UNNEPolicyState. One buffer per state_in/state_out tensor pair, allocated once.
class OnnxPolicyState final : public PolicyState {
public:
	void reset() override; // Zeros every buffer.
	// INVALID_ARGUMENT unless p_other is an OnnxPolicyState with the same buffer sizes.
	Status copy_from(const PolicyState &p_other) override;
	// Drops the oldest sequence entry, or replaces the state for models without a sequence dimension.
	Status notify_state_update() override;
	std::string to_string() const override;
};

} // namespace schola
