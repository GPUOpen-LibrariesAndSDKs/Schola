// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>

namespace schola {
class PolicyState;
} // namespace schola

// GDScript handle to one agent's policy state. Created by ScholaOnnxPolicy.create_state().
class ScholaPolicyState : public godot::RefCounted {
	GDCLASS(ScholaPolicyState, godot::RefCounted)

public:
	void reset();
	godot::Error copy_from(const godot::Ref<ScholaPolicyState> &p_other);

	schola::PolicyState *get_state() const;

protected:
	static void _bind_methods();
};
