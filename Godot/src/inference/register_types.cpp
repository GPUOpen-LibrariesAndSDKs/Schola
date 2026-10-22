// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "inference/register_types.h"

#include "bindings/inference/schola_onnx_policy.h"
#include "bindings/inference/schola_policy_state.h"

#include <godot_cpp/core/class_db.hpp>

void register_schola_inference_types() {
	GDREGISTER_CLASS(ScholaPolicyState);
	GDREGISTER_CLASS(ScholaOnnxPolicy);
}

void unregister_schola_inference_types() {
}
