// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <cstdint>

namespace schola {
class Space;
} // namespace schola

// Godot-facing base resource for one observation or action space definition
// resources own Inspector properties and convert them into US2 core spaces
class ScholaSpace : public godot::Resource {
	GDCLASS(ScholaSpace, godot::Resource)

public:
	// Returns this resource's US2 space definition
	// Call Space::check_definition() to report invalid Inspector configuration
	virtual schola::Space to_space() const = 0;

	// Returns an empty string when p_value belongs to this space
	// Otherwise returns a reason why the value is invalid
	godot::String get_validation_error(const godot::Variant &p_value) const;

	// Returns an empty string when the definition is valid
	godot::String get_definition_error() const;
	godot::Dictionary to_dictionary() const;
	int64_t get_flattened_size() const;
	godot::Variant make_default_value() const;
	godot::String describe() const;

protected:
	static void _bind_methods();
};
