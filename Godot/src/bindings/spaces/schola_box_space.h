// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "bindings/spaces/schola_space.h"

#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>

#include <cstdint>

// Inspector resource for a continuous US2 Box space
class ScholaBoxSpace : public ScholaSpace {
	GDCLASS(ScholaBoxSpace, ScholaSpace)

public:
	// One lower and upper bound per Box element in row-major order
	void set_low(const godot::PackedFloat32Array &p_low);
	godot::PackedFloat32Array get_low() const;
	void set_high(const godot::PackedFloat32Array &p_high);
	godot::PackedFloat32Array get_high() const;

	// Positive dimensions in row-major order
	// An empty shape uses one dimension containing low.size()
	void set_shape(const godot::PackedInt32Array &p_shape);
	godot::PackedInt32Array get_shape() const;

	// US2 DType value for the Box elements
	void set_dtype(int32_t p_dtype);
	int32_t get_dtype() const;

	schola::Space to_space() const override;

protected:
	static void _bind_methods();

private:
	godot::PackedFloat32Array low;
	godot::PackedFloat32Array high;
	godot::PackedInt32Array shape;
	int32_t dtype = 0;
};
