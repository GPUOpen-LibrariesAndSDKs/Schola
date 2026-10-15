// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "bindings/spaces/schola_space.h"

#include <godot_cpp/variant/packed_int32_array.hpp>

// Inspector resource for a US2 MultiDiscrete space
class ScholaMultiDiscreteSpace : public ScholaSpace {
	GDCLASS(ScholaMultiDiscreteSpace, ScholaSpace)

public:
	// Number of choices for each independent dimension
	void set_nvec(const godot::PackedInt32Array &p_nvec);
	godot::PackedInt32Array get_nvec() const;

	schola::Space to_space() const override;

protected:
	static void _bind_methods();

private:
	godot::PackedInt32Array nvec;
};
