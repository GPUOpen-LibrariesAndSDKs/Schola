// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <cstdint>
#include <memory>

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
	// One inference on a GDScript value; p_agent selects whose memory is used.
	godot::Variant infer(const godot::Variant &p_observation, int32_t p_agent = 0);
	void reset_memory(int32_t p_agent); // Clears one agent's memory used by infer().
	void reset_all_memory(); // Clears every agent's memory used by infer().

	// New policy with its own memory; shares the loaded model.
	std::unique_ptr<schola::OnnxPolicy> instantiate() const;

protected:
	static void _bind_methods();
};
