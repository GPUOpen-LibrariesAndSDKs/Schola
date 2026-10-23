// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "bindings/register_types.h"

#include "bindings/environment/schola_agent.h"
#include "bindings/environment/schola_environment.h"
#include "bindings/schola_runtime_probe.h"
#include "bindings/spaces/schola_box_space.h"
#include "bindings/spaces/schola_discrete_space.h"
#include "bindings/spaces/schola_space.h"

#include <godot_cpp/core/class_db.hpp>

void register_schola_binding_types() {
	GDREGISTER_CLASS(ScholaRuntimeProbe);
	GDREGISTER_CLASS(ScholaEnvironment);
	GDREGISTER_CLASS(ScholaAgent);
	GDREGISTER_ABSTRACT_CLASS(ScholaSpace);
	GDREGISTER_CLASS(ScholaBoxSpace);
	GDREGISTER_CLASS(ScholaDiscreteSpace);
}

void unregister_schola_binding_types() {
}
