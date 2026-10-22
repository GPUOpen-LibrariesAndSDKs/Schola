// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/string.hpp>

#include <memory>

namespace schola {
class PolicyState;
} // namespace schola

// GDScript handle to one agent's policy state. Created by ScholaOnnxPolicy.create_state().
class ScholaPolicyState : public godot::RefCounted {
	GDCLASS(ScholaPolicyState, godot::RefCounted)

public:
	ScholaPolicyState();
	~ScholaPolicyState() override;

	void reset();
	godot::Error copy_from(const godot::Ref<ScholaPolicyState> &p_other);
	godot::String _to_string() const;

	schola::PolicyState *get_state() const;
	void set_state(std::unique_ptr<schola::PolicyState> p_state);

protected:
	static void _bind_methods();

private:
	std::unique_ptr<schola::PolicyState> state;
};
