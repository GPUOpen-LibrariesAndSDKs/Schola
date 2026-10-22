// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "bindings/inference/schola_policy_state.h"

#include "core/policy/policy_state.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/error_macros.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

ScholaPolicyState::ScholaPolicyState() = default;

ScholaPolicyState::~ScholaPolicyState() = default;

void ScholaPolicyState::reset() {
	if (state != nullptr) {
		state->reset();
	}
}

godot::Error ScholaPolicyState::copy_from(const godot::Ref<ScholaPolicyState> &p_other) {
	ERR_FAIL_COND_V_MSG(state == nullptr || p_other.is_null() || p_other->state == nullptr, godot::ERR_INVALID_PARAMETER, "ScholaPolicyState.copy_from: both states must come from ScholaOnnxPolicy.create_state().");
	schola::Status status = state->copy_from(*p_other->state);
	if (!status.is_ok()) {
		godot::UtilityFunctions::push_error(godot::String::utf8(status.get_message().c_str()));
		return godot::ERR_INVALID_PARAMETER;
	}
	return godot::OK;
}

godot::String ScholaPolicyState::_to_string() const {
	return state == nullptr ? godot::String("ScholaPolicyState(empty)") : godot::String::utf8(state->to_string().c_str());
}

schola::PolicyState *ScholaPolicyState::get_state() const {
	return state.get();
}

void ScholaPolicyState::set_state(std::unique_ptr<schola::PolicyState> p_state) {
	state = std::move(p_state);
}

void ScholaPolicyState::_bind_methods() {
	godot::ClassDB::bind_method(godot::D_METHOD("reset"), &ScholaPolicyState::reset);
	godot::ClassDB::bind_method(godot::D_METHOD("copy_from", "other"), &ScholaPolicyState::copy_from);
}
