// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "bindings/inference/schola_policy_state.h"

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>

namespace schola {
class OnnxPolicy;
} // namespace schola

class ScholaOnnxPolicy : public godot::Resource {
	GDCLASS(ScholaOnnxPolicy, godot::Resource)

public:
	void set_model_path(const godot::String &p_path);
	godot::String get_model_path() const;

	godot::Error load();
	bool is_loaded() const;
	bool is_recurrent() const;
	godot::String get_last_error() const;
	godot::PackedStringArray get_input_names() const;
	godot::PackedStringArray get_output_names() const;

	godot::Ref<ScholaPolicyState> create_state() const;
	// One inference on a GDScript value. p_state is advanced in place; required for recurrent models.
	godot::Variant infer(const godot::Variant &p_observation, const godot::Ref<ScholaPolicyState> &p_state = godot::Ref<ScholaPolicyState>());

	schola::OnnxPolicy *get_policy() const; // Shared by every stepper using this resource.

protected:
	static void _bind_methods();
};
