// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "bindings/inference/schola_onnx_policy.h"

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/typed_array.hpp>

class ScholaAgent;

// Runs observe -> infer -> act for its agents every physics frame, without Python.
// Wraps schola::SimpleStepper. Signal inference_failed(message) is registered in _bind_methods().
class ScholaInferenceStepper : public godot::Node {
	GDCLASS(ScholaInferenceStepper, godot::Node)

public:
	void set_policy(const godot::Ref<ScholaOnnxPolicy> &p_policy);
	godot::Ref<ScholaOnnxPolicy> get_policy() const;
	void set_agents(const godot::TypedArray<godot::NodePath> &p_agents);
	godot::TypedArray<godot::NodePath> get_agents() const;
	void set_enabled(bool p_enabled);
	bool is_enabled() const;

	bool step();
	void reset(ScholaAgent *p_agent);
	void reset_all();

	godot::PackedStringArray get_configuration_errors() const;
	godot::PackedStringArray _get_configuration_warnings() const override;
	void _ready() override;
	void _physics_process(double p_delta) override;

protected:
	static void _bind_methods();
};
