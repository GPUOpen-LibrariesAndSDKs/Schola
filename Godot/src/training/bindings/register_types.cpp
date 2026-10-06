// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "training/bindings/register_types.h"

#include "training/schola_training_probe.h"

#include <godot_cpp/core/class_db.hpp>

void register_schola_training_binding_types() {
	GDREGISTER_CLASS(ScholaTrainingProbe);
}

void unregister_schola_training_binding_types() {
}
