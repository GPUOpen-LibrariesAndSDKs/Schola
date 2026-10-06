// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "bindings/register_types.h"

#include "bindings/schola_runtime_probe.h"

#include <godot_cpp/core/class_db.hpp>

void register_schola_binding_types() {
	GDREGISTER_CLASS(ScholaRuntimeProbe);
}

void unregister_schola_binding_types() {
}
