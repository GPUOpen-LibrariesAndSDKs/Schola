// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "bindings/inference/schola_policy_state.h"

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <cstdint>
#include <memory>

namespace schola {
class OnnxPolicy;
class PolicyState;
struct InteractionDefinition;
} // namespace schola

class ScholaSpace;

class ScholaOnnxPolicy : public godot::Resource {
	GDCLASS(ScholaOnnxPolicy, godot::Resource)

public:
	ScholaOnnxPolicy();
	~ScholaOnnxPolicy() override;

	void set_model_path(const godot::String &p_path);
	godot::String get_model_path() const;
	// Takes effect on the next load().
	void set_max_state_sequence_length(int32_t p_length);
	int32_t get_max_state_sequence_length() const;

	godot::Error load();
	bool is_loaded() const;
	bool is_recurrent() const;
	godot::String get_last_error() const;
	godot::PackedStringArray get_input_names() const;
	godot::PackedStringArray get_output_names() const;

	// Checks the model against these spaces; required before create_state() and infer().
	godot::Error initialize_spaces(const godot::Ref<ScholaSpace> &p_observation_space, const godot::Ref<ScholaSpace> &p_action_space);
	bool is_initialized() const;
	godot::Ref<ScholaPolicyState> create_state() const;
	// One inference on a GDScript value. p_state is advanced in place; required for recurrent models.
	godot::Variant infer(const godot::Variant &p_observation, const godot::Ref<ScholaPolicyState> &p_state = godot::Ref<ScholaPolicyState>());

	schola::OnnxPolicy *get_policy() const; // Shared by every stepper using this resource.

protected:
	static void _bind_methods();

private:
	void fail(const godot::String &p_message);

	godot::String model_path;
	int32_t max_state_sequence_length = 1;
	godot::String last_error;
	std::unique_ptr<schola::OnnxPolicy> policy;
	std::unique_ptr<schola::InteractionDefinition> definition;
	std::unique_ptr<schola::PolicyState> stateless_state;
};
