// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "bindings/inference/schola_onnx_policy.h"

#include "bindings/spaces/schola_space.h"
#include "core/environment/interaction_definition.h"
#include "core/policy/policy_state.h"
#include "inference/onnx_policy.h"
#include "interop/variant_conversion.h"

#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <string>
#include <vector>

namespace {

godot::Error to_error(const schola::Status &p_status) {
	switch (p_status.get_code()) {
		case schola::StatusCode::OK:
			return godot::OK;
		case schola::StatusCode::INVALID_ARGUMENT:
			return godot::ERR_INVALID_PARAMETER;
		case schola::StatusCode::NOT_FOUND:
			return godot::ERR_DOES_NOT_EXIST;
		case schola::StatusCode::INVALID_DATA:
		case schola::StatusCode::INCOMPATIBLE:
			return godot::ERR_INVALID_DATA;
		case schola::StatusCode::FAILED_PRECONDITION:
			return godot::ERR_UNCONFIGURED;
		case schola::StatusCode::TIMEOUT:
			return godot::ERR_TIMEOUT;
		case schola::StatusCode::CLOSED:
			return godot::ERR_CONNECTION_ERROR;
		default:
			return godot::FAILED;
	}
}

godot::String to_godot(const std::string &p_text) {
	return godot::String::utf8(p_text.c_str());
}

godot::PackedStringArray to_godot(const std::vector<std::string> &p_names) {
	godot::PackedStringArray names;
	for (const std::string &name : p_names) {
		names.push_back(to_godot(name));
	}
	return names;
}

} // namespace

ScholaOnnxPolicy::ScholaOnnxPolicy() = default;

ScholaOnnxPolicy::~ScholaOnnxPolicy() = default;

void ScholaOnnxPolicy::set_model_path(const godot::String &p_path) {
	model_path = p_path;
}

godot::String ScholaOnnxPolicy::get_model_path() const {
	return model_path;
}

void ScholaOnnxPolicy::set_max_state_sequence_length(int32_t p_length) {
	max_state_sequence_length = p_length;
}

int32_t ScholaOnnxPolicy::get_max_state_sequence_length() const {
	return max_state_sequence_length;
}

godot::Error ScholaOnnxPolicy::load() {
	policy.reset();
	definition.reset();
	stateless_state.reset();
	last_error = godot::String();

	// FileAccess also reads res:// files packed into an exported game, which ONNX Runtime cannot open itself.
	const godot::PackedByteArray bytes = godot::FileAccess::get_file_as_bytes(model_path);
	if (bytes.is_empty()) {
		const godot::Error error = godot::FileAccess::get_open_error();
		fail("ScholaOnnxPolicy.load: cannot read '" + model_path + "'");
		return error == godot::OK ? godot::ERR_FILE_CORRUPT : error;
	}

	auto loaded = std::make_unique<schola::OnnxPolicy>();
	schola::Status status = loaded->set_max_state_sequence_length(max_state_sequence_length);
	if (status.is_ok()) {
		status = loaded->load_model(bytes.ptr(), static_cast<size_t>(bytes.size()), model_path.utf8().get_data());
	}
	if (!status.is_ok()) {
		fail(to_godot(status.get_message()));
		return to_error(status);
	}
	policy = std::move(loaded);
	return godot::OK;
}

bool ScholaOnnxPolicy::is_loaded() const {
	return policy != nullptr;
}

bool ScholaOnnxPolicy::is_recurrent() const {
	return policy != nullptr && policy->is_recurrent();
}

godot::String ScholaOnnxPolicy::get_last_error() const {
	return last_error;
}

godot::PackedStringArray ScholaOnnxPolicy::get_input_names() const {
	return policy == nullptr ? godot::PackedStringArray() : to_godot(policy->get_input_names());
}

godot::PackedStringArray ScholaOnnxPolicy::get_output_names() const {
	return policy == nullptr ? godot::PackedStringArray() : to_godot(policy->get_output_names());
}

godot::Error ScholaOnnxPolicy::initialize_spaces(const godot::Ref<ScholaSpace> &p_observation_space, const godot::Ref<ScholaSpace> &p_action_space) {
	if (policy == nullptr) {
		fail("ScholaOnnxPolicy.initialize_spaces: call load() first");
		return godot::ERR_UNCONFIGURED;
	}
	if (p_observation_space.is_null() || p_action_space.is_null()) {
		fail("ScholaOnnxPolicy.initialize_spaces: both spaces are required");
		return godot::ERR_INVALID_PARAMETER;
	}

	auto spaces = std::make_unique<schola::InteractionDefinition>();
	spaces->observation_space = p_observation_space->to_space();
	spaces->action_space = p_action_space->to_space();
	schola::Status status = policy->init(*spaces);
	std::unique_ptr<schola::PolicyState> state;
	if (status.is_ok() && !policy->is_recurrent()) {
		status = policy->create_initial_state(state);
	}
	if (!status.is_ok()) {
		definition.reset();
		fail(to_godot(status.get_message()));
		return to_error(status);
	}
	definition = std::move(spaces);
	stateless_state = std::move(state);
	return godot::OK;
}

bool ScholaOnnxPolicy::is_initialized() const {
	return definition != nullptr;
}

godot::Ref<ScholaPolicyState> ScholaOnnxPolicy::create_state() const {
	std::unique_ptr<schola::PolicyState> state;
	schola::Status status = !is_initialized()
			? schola::Status::error(schola::StatusCode::FAILED_PRECONDITION, "ScholaOnnxPolicy.create_state: call load() and initialize_spaces() first")
			: policy->create_initial_state(state);
	if (!status.is_ok()) {
		godot::UtilityFunctions::push_error(to_godot(status.get_message()));
		return godot::Ref<ScholaPolicyState>();
	}
	godot::Ref<ScholaPolicyState> result;
	result.instantiate();
	result->set_state(std::move(state));
	return result;
}

godot::Variant ScholaOnnxPolicy::infer(const godot::Variant &p_observation, const godot::Ref<ScholaPolicyState> &p_state) {
	if (!is_initialized()) {
		fail("ScholaOnnxPolicy.infer: call load() and initialize_spaces() first");
		return godot::Variant();
	}
	schola::PolicyState *state = p_state.is_valid() ? p_state->get_state() : stateless_state.get();
	if (state == nullptr) {
		fail("ScholaOnnxPolicy.infer: this model has memory; pass a state from create_state()");
		return godot::Variant();
	}

	schola::Point observation;
	schola::Status status = schola::variant_to_point(p_observation, definition->observation_space, observation);
	schola::Point action = definition->action_space.make_point();
	if (status.is_ok()) {
		status = policy->think(observation, *state, action);
	}
	if (!status.is_ok()) {
		fail(to_godot(status.get_message()));
		return godot::Variant();
	}
	return schola::point_to_variant(action);
}

schola::OnnxPolicy *ScholaOnnxPolicy::get_policy() const {
	return policy.get();
}

void ScholaOnnxPolicy::fail(const godot::String &p_message) {
	last_error = p_message;
	godot::UtilityFunctions::push_error(p_message);
}

void ScholaOnnxPolicy::_bind_methods() {
	godot::ClassDB::bind_method(godot::D_METHOD("set_model_path", "path"), &ScholaOnnxPolicy::set_model_path);
	godot::ClassDB::bind_method(godot::D_METHOD("get_model_path"), &ScholaOnnxPolicy::get_model_path);
	godot::ClassDB::bind_method(godot::D_METHOD("set_max_state_sequence_length", "length"), &ScholaOnnxPolicy::set_max_state_sequence_length);
	godot::ClassDB::bind_method(godot::D_METHOD("get_max_state_sequence_length"), &ScholaOnnxPolicy::get_max_state_sequence_length);
	godot::ClassDB::bind_method(godot::D_METHOD("load"), &ScholaOnnxPolicy::load);
	godot::ClassDB::bind_method(godot::D_METHOD("is_loaded"), &ScholaOnnxPolicy::is_loaded);
	godot::ClassDB::bind_method(godot::D_METHOD("is_recurrent"), &ScholaOnnxPolicy::is_recurrent);
	godot::ClassDB::bind_method(godot::D_METHOD("get_last_error"), &ScholaOnnxPolicy::get_last_error);
	godot::ClassDB::bind_method(godot::D_METHOD("get_input_names"), &ScholaOnnxPolicy::get_input_names);
	godot::ClassDB::bind_method(godot::D_METHOD("get_output_names"), &ScholaOnnxPolicy::get_output_names);
	godot::ClassDB::bind_method(godot::D_METHOD("initialize_spaces", "observation_space", "action_space"), &ScholaOnnxPolicy::initialize_spaces);
	godot::ClassDB::bind_method(godot::D_METHOD("is_initialized"), &ScholaOnnxPolicy::is_initialized);
	godot::ClassDB::bind_method(godot::D_METHOD("create_state"), &ScholaOnnxPolicy::create_state);
	godot::ClassDB::bind_method(godot::D_METHOD("infer", "observation", "state"), &ScholaOnnxPolicy::infer, DEFVAL(godot::Ref<ScholaPolicyState>()));

	ADD_PROPERTY(godot::PropertyInfo(godot::Variant::STRING, "model_path", godot::PROPERTY_HINT_FILE, "*.onnx"), "set_model_path", "get_model_path");
	ADD_PROPERTY(godot::PropertyInfo(godot::Variant::INT, "max_state_sequence_length", godot::PROPERTY_HINT_RANGE, "1,1024,1,or_greater"), "set_max_state_sequence_length", "get_max_state_sequence_length");
}
