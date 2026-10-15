// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "bindings/spaces/schola_space.h"

#include <godot_cpp/variant/dictionary.hpp>

// Inspector resource for a named collection of US2 spaces
class ScholaDictSpace : public ScholaSpace {
	GDCLASS(ScholaDictSpace, ScholaSpace)

public:
	// Maps each String key to one ScholaSpace resource
	void set_spaces(const godot::Dictionary &p_spaces);
	godot::Dictionary get_spaces() const;

	schola::Space to_space() const override;

protected:
	static void _bind_methods();

private:
	godot::Dictionary spaces;
};
