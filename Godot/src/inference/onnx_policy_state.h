// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "core/policy/policy_state.h"

#include <cstdint>
#include <string>
#include <vector>

namespace schola {

class OnnxPolicy;

class OnnxPolicyState final : public PolicyState {
public:
	// One buffer per state_in_*/state_out_* pair: sequence_length entries of state_size values, then one
	// pending entry that state_out writes to, so a failed run never changes the state.
	struct Buffer {
		std::string name;
		int64_t sequence_length = 1;
		int64_t state_size = 0;
		std::vector<float> values;

		float *sequence() { return values.data(); }
		float *pending() { return values.data() + sequence_length * state_size; }
	};

	struct BufferLayout {
		std::string name;
		int64_t sequence_length = 1;
		int64_t state_size = 0;
	};

	OnnxPolicyState(const OnnxPolicy *p_owner, const std::vector<BufferLayout> &p_layouts);

	void reset() override;
	Status copy_from(const PolicyState &p_other) override;
	// Drops the oldest entry and appends the pending one.
	Status notify_state_update() override;
	std::string to_string() const override;

	const OnnxPolicy *get_owner() const;
	std::vector<Buffer> &get_buffers();
	const std::vector<Buffer> &get_buffers() const;

private:
	const OnnxPolicy *owner = nullptr;
	std::vector<Buffer> buffers;
};

} // namespace schola
