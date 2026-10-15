// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "core/policy/policy.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace schola {

// Runs a Schola-exported ONNX model with ONNX Runtime.
class OnnxPolicy final : public Policy {
public:
	// INVALID_DATA if the file is corrupt; INCOMPATIBLE if its recurrent state is invalid.
	Status load_model(const uint8_t *p_data, size_t p_size, const std::string &p_model_name);

	Status init(const InteractionDefinition &p_definition, int32_t p_agent_count) override;
	Status think(int32_t p_agent, const Point &p_observation, Point &r_action) override;
	// One model run for all agents, including their memory.
	Status batched_think(const std::vector<const Point *> &p_observations, const std::vector<Point *> &r_actions) override;
	void reset(int32_t p_agent) override; // Zeros the agent's memory.
	void reset_all() override;
	bool is_recurrent() const override; // Model has state_in/state_out tensors.

	bool is_ready() const; // load_model() and init() succeeded.
};

} // namespace schola
