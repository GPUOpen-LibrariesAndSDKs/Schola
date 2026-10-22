// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "bindings/spaces/schola_box_space.h"

#include "core/spaces/space.h"

#include <godot_cpp/core/class_db.hpp>

#include <limits>
#include <vector>

void ScholaBoxSpace::set_low(const godot::PackedFloat32Array &p_low) {
	low = p_low;
	emit_changed();
}

godot::PackedFloat32Array ScholaBoxSpace::get_low() const {
	return low;
}

void ScholaBoxSpace::set_high(const godot::PackedFloat32Array &p_high) {
	high = p_high;
	emit_changed();
}

godot::PackedFloat32Array ScholaBoxSpace::get_high() const {
	return high;
}

void ScholaBoxSpace::set_shape(const godot::PackedInt32Array &p_shape) {
	shape = p_shape;
	emit_changed();
}

godot::PackedInt32Array ScholaBoxSpace::get_shape() const {
	return shape;
}

void ScholaBoxSpace::set_dtype(int32_t p_dtype) {
	dtype = p_dtype;
	emit_changed();
}

int32_t ScholaBoxSpace::get_dtype() const {
	return dtype;
}

schola::Space ScholaBoxSpace::to_space() const {
	schola::BoxSpace box;
	box.dtype = static_cast<schola::DType>(dtype);
	box.shape.assign(shape.ptr(), shape.ptr() + shape.size());

	int64_t element_count = 0;
	if (box.shape.empty()) {
		element_count = !low.is_empty() ? low.size() : high.size();
		if (element_count > 0) {
			box.shape.push_back(static_cast<int32_t>(element_count));
		}
	} else {
		element_count = 1;
		for (int32_t dimension : box.shape) {
			element_count *= dimension > 0 ? dimension : 0;
		}
	}

	// An empty bound is unbounded on that side
	if (low.is_empty()) {
		box.low.assign(static_cast<size_t>(element_count), -std::numeric_limits<float>::infinity());
	} else {
		box.low.assign(low.ptr(), low.ptr() + low.size());
	}
	if (high.is_empty()) {
		box.high.assign(static_cast<size_t>(element_count), std::numeric_limits<float>::infinity());
	} else {
		box.high.assign(high.ptr(), high.ptr() + high.size());
	}
	return schola::Space(std::move(box));
}

void ScholaBoxSpace::_bind_methods() {
	godot::ClassDB::bind_method(godot::D_METHOD("set_low", "low"), &ScholaBoxSpace::set_low);
	godot::ClassDB::bind_method(godot::D_METHOD("get_low"), &ScholaBoxSpace::get_low);
	godot::ClassDB::bind_method(godot::D_METHOD("set_high", "high"), &ScholaBoxSpace::set_high);
	godot::ClassDB::bind_method(godot::D_METHOD("get_high"), &ScholaBoxSpace::get_high);
	godot::ClassDB::bind_method(godot::D_METHOD("set_shape", "shape"), &ScholaBoxSpace::set_shape);
	godot::ClassDB::bind_method(godot::D_METHOD("get_shape"), &ScholaBoxSpace::get_shape);
	godot::ClassDB::bind_method(godot::D_METHOD("set_dtype", "dtype"), &ScholaBoxSpace::set_dtype);
	godot::ClassDB::bind_method(godot::D_METHOD("get_dtype"), &ScholaBoxSpace::get_dtype);

	ADD_PROPERTY(godot::PropertyInfo(godot::Variant::PACKED_FLOAT32_ARRAY, "low"), "set_low", "get_low");
	ADD_PROPERTY(godot::PropertyInfo(godot::Variant::PACKED_FLOAT32_ARRAY, "high"), "set_high", "get_high");
	ADD_PROPERTY(godot::PropertyInfo(godot::Variant::PACKED_INT32_ARRAY, "shape"), "set_shape", "get_shape");
	ADD_PROPERTY(godot::PropertyInfo(godot::Variant::INT, "dtype", godot::PROPERTY_HINT_ENUM, "float32,uint8,uint16,uint32,uint64,int8,int16,int32,int64,float16,float64,bool"), "set_dtype", "get_dtype");
}
