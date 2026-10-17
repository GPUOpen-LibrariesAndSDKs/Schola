// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "core/common/status.h"

#include <string>

namespace schola {

// What a policy remembers about one agent between think() calls, e.g. LSTM tensors.
// Created by Policy::create_initial_state() and owned by the caller, so one policy serves many agents.
class PolicyState {
public:
	virtual ~PolicyState() = default;

	virtual void reset() = 0; // Back to the start of an episode.
	// Overwrites this state; INVALID_ARGUMENT if p_other is a different state type.
	virtual Status copy_from(const PolicyState &p_other) = 0;
	// Called by think() after the model ran, before the new state is written. Default does nothing.
	virtual Status notify_state_update() { return Status::ok(); }
	virtual std::string to_string() const = 0;
};

// Unreal: UStatelessPolicyState. The state of policies with no memory, so a state is never null.
class EmptyPolicyState final : public PolicyState {
public:
	void reset() override {}
	Status copy_from(const PolicyState &p_other) override;
	std::string to_string() const override { return "EmptyPolicyState"; }
};

} // namespace schola
