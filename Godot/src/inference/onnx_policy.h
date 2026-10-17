// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "core/policy/policy.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace schola {

class OnnxPolicy final : public Policy {
public:
	// INVALID_DATA if the file is corrupt; INCOMPATIBLE if its recurrent state is invalid.
	Status load_model(const uint8_t *p_data, size_t p_size, const std::string &p_model_name);

	Status init(const InteractionDefinition &p_definition) override;
	// OnnxPolicyState for recurrent models, EmptyPolicyState otherwise.
	Status create_initial_state(std::unique_ptr<PolicyState> &r_state) const override;
	// INVALID_ARGUMENT if p_state was not created by this policy.
	Status think(const Point &p_observation, PolicyState &p_state, Point &r_action) override;
	Status batched_think(const std::vector<const Point *> &p_observations, const std::vector<PolicyState *> &p_states, const std::vector<Point *> &r_actions) override;
	bool is_recurrent() const override; // Model has state_in/state_out tensors.

	bool is_ready() const; // load_model() and init() succeeded.
};

} // namespace schola
