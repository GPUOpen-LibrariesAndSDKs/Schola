// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "bindings/register_types.h"

#include "bindings/schola_runtime_probe.h"
#include "bindings/spaces/schola_box_space.h"
#include "bindings/spaces/schola_discrete_space.h"
#include "bindings/spaces/schola_space.h"

#include <godot_cpp/core/class_db.hpp>

void register_schola_binding_types() {
	GDREGISTER_CLASS(ScholaRuntimeProbe);
	GDREGISTER_ABSTRACT_CLASS(ScholaSpace);
	GDREGISTER_CLASS(ScholaBoxSpace);
	GDREGISTER_CLASS(ScholaDiscreteSpace);
}

void unregister_schola_binding_types() {
}
