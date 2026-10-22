// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "inference/onnx_policy.h"

#include "core/environment/interaction_definition.h"
#include "core/spaces/space.h"
#include "inference/onnx_policy_state.h"

#include <onnxruntime_c_api.h>

#include <cstring>
#include <string>
#include <utility>

namespace schola {

namespace {

const OrtApi &ort() {
	static const OrtApi *api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
	return *api;
}

Status from_ort(OrtStatus *p_status, StatusCode p_code, const std::string &p_context) {
	if (p_status == nullptr) {
		return Status::ok();
	}
	std::string message = p_context + ": " + ort().GetErrorMessage(p_status);
	ort().ReleaseStatus(p_status);
	return Status::error(p_code, message);
}

struct OrtDeleter {
	void operator()(OrtSession *p_session) const { ort().ReleaseSession(p_session); }
	void operator()(OrtSessionOptions *p_options) const { ort().ReleaseSessionOptions(p_options); }
	void operator()(OrtTypeInfo *p_type_info) const { ort().ReleaseTypeInfo(p_type_info); }
	void operator()(OrtValue *p_value) const { ort().ReleaseValue(p_value); }
	void operator()(OrtMemoryInfo *p_memory_info) const { ort().ReleaseMemoryInfo(p_memory_info); }
};

template <typename T>
using OrtPtr = std::unique_ptr<T, OrtDeleter>;

struct SharedEnvironment {
	OrtEnv *env = nullptr;
	Status status = Status::ok();

	SharedEnvironment() {
		status = from_ort(ort().CreateEnv(ORT_LOGGING_LEVEL_FATAL, "schola", &env), StatusCode::RUNTIME_FAILURE, "ONNX Runtime CreateEnv");
	}
	~SharedEnvironment() {
		if (env != nullptr) {
			ort().ReleaseEnv(env);
		}
	}
};

const SharedEnvironment &shared_environment() {
	static const SharedEnvironment environment;
	return environment;
}

bool starts_with(const std::string &p_text, const char *p_prefix) {
	return p_text.rfind(p_prefix, 0) == 0;
}

size_t element_size(ONNXTensorElementDataType p_type) {
	switch (p_type) {
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
			return 8;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
			return 4;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
			return 2;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
			return 1;
		default:
			return 0;
	}
}

template <typename T>
void put(std::vector<uint8_t> &r_buffer, size_t p_index, T p_value) {
	std::memcpy(r_buffer.data() + p_index * sizeof(T), &p_value, sizeof(T));
}

template <typename T>
T get(const std::vector<uint8_t> &p_buffer, size_t p_index) {
	T value;
	std::memcpy(&value, p_buffer.data() + p_index * sizeof(T), sizeof(T));
	return value;
}

void store(ONNXTensorElementDataType p_type, std::vector<uint8_t> &r_buffer, size_t p_index, double p_value) {
	switch (p_type) {
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
			put(r_buffer, p_index, static_cast<float>(p_value));
			break;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
			put(r_buffer, p_index, p_value);
			break;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
			put(r_buffer, p_index, static_cast<int64_t>(p_value));
			break;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
			put(r_buffer, p_index, static_cast<int32_t>(p_value));
			break;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
			put(r_buffer, p_index, static_cast<int16_t>(p_value));
			break;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
			put(r_buffer, p_index, static_cast<uint16_t>(p_value));
			break;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
			put(r_buffer, p_index, static_cast<int8_t>(p_value));
			break;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
			put(r_buffer, p_index, static_cast<uint8_t>(p_value));
			break;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
			put(r_buffer, p_index, static_cast<uint8_t>(p_value != 0.0));
			break;
		default:
			break;
	}
}

double load(ONNXTensorElementDataType p_type, const std::vector<uint8_t> &p_buffer, size_t p_index) {
	switch (p_type) {
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
			return get<float>(p_buffer, p_index);
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
			return get<double>(p_buffer, p_index);
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
			return static_cast<double>(get<int64_t>(p_buffer, p_index));
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
			return get<int32_t>(p_buffer, p_index);
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
			return get<int8_t>(p_buffer, p_index);
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
			return get<uint8_t>(p_buffer, p_index);
		default:
			return 0.0;
	}
}

bool accepts(SpaceKind p_kind, ONNXTensorElementDataType p_type, bool p_input) {
	switch (p_kind) {
		case SpaceKind::BOX:
			if (p_input) {
				return element_size(p_type) != 0 && p_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
			}
			return p_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || p_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
		case SpaceKind::DISCRETE:
		case SpaceKind::MULTI_DISCRETE:
			return p_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 || p_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
		case SpaceKind::MULTI_BINARY:
			return p_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 || p_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 || p_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
		default:
			return false;
	}
}

const char *kind_name(SpaceKind p_kind) {
	switch (p_kind) {
		case SpaceKind::BOX:
			return "Box";
		case SpaceKind::DISCRETE:
			return "Discrete";
		case SpaceKind::MULTI_DISCRETE:
			return "MultiDiscrete";
		case SpaceKind::MULTI_BINARY:
			return "MultiBinary";
		default:
			return "Dict";
	}
}

std::string shape_text(const std::vector<int64_t> &p_shape) {
	std::string text = "[";
	for (size_t i = 0; i < p_shape.size(); ++i) {
		text += (i == 0 ? "" : ", ") + (p_shape[i] < 0 ? std::string("?") : std::to_string(p_shape[i]));
	}
	return text + "]";
}

template <typename P>
P *find_leaf(P &p_point, const std::string &p_key) {
	if (p_key.empty()) {
		return &p_point;
	}
	auto *dict = p_point.template get_if<DictPoint>();
	if (dict == nullptr) {
		return nullptr;
	}
	for (auto &entry : dict->entries) {
		if (entry.first == p_key) {
			return &entry.second;
		}
	}
	return nullptr;
}

} // namespace

struct TensorDescription {
	std::string name;
	ONNXTensorElementDataType element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	std::vector<int64_t> shape; // -1 for a dynamic dimension.
};

struct StateLayout {
	size_t input_index = 0;
	size_t output_index = 0;
	bool has_sequence = false;
	int64_t state_size = 0;

	std::vector<int64_t> input_shape;
	std::vector<int64_t> output_shape;
};

struct Binding {
	std::string key;
	size_t tensor_index = 0;
	SpaceKind kind = SpaceKind::BOX;
	ONNXTensorElementDataType element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	std::vector<int64_t> shape;
	size_t element_count = 0;
	std::vector<int32_t> bounds;
	std::vector<uint8_t> buffer;
	OrtPtr<OrtValue> value;

	std::string label(const char *p_space) const {
		return key.empty() ? std::string(p_space) : std::string(p_space) + " '" + key + "'";
	}
};

namespace {

bool leaf_shape(const Space &p_space, Binding &r_binding, std::vector<int64_t> &r_dimensions) {
	if (const BoxSpace *box = p_space.get_if<BoxSpace>()) {
		r_binding.kind = SpaceKind::BOX;
		r_dimensions.assign(box->shape.begin(), box->shape.end());
	} else if (const DiscreteSpace *discrete = p_space.get_if<DiscreteSpace>()) {
		r_binding.kind = SpaceKind::DISCRETE;
		r_binding.bounds = { discrete->n };
	} else if (const MultiDiscreteSpace *multi_discrete = p_space.get_if<MultiDiscreteSpace>()) {
		r_binding.kind = SpaceKind::MULTI_DISCRETE;
		r_binding.bounds = multi_discrete->nvec;
		r_dimensions = { static_cast<int64_t>(multi_discrete->nvec.size()) };
	} else if (const MultiBinarySpace *multi_binary = p_space.get_if<MultiBinarySpace>()) {
		r_binding.kind = SpaceKind::MULTI_BINARY;
		r_dimensions = { multi_binary->n };
	} else {
		return false;
	}
	return true;
}

template <typename T>
bool write_values(Binding &r_binding, const std::vector<T> &p_values) {
	if (p_values.size() != r_binding.element_count) {
		return false;
	}
	for (size_t i = 0; i < p_values.size(); ++i) {
		store(r_binding.element_type, r_binding.buffer, i, static_cast<double>(p_values[i]));
	}
	return true;
}

Status write_observation(Binding &r_binding, const Point &p_point) {
	bool written = false;
	if (const BoxPoint *box = p_point.get_if<BoxPoint>(); box && r_binding.kind == SpaceKind::BOX) {
		written = write_values(r_binding, box->values);
	} else if (const DiscretePoint *discrete = p_point.get_if<DiscretePoint>(); discrete && r_binding.kind == SpaceKind::DISCRETE) {
		store(r_binding.element_type, r_binding.buffer, 0, discrete->value);
		written = true;
	} else if (const MultiDiscretePoint *multi_discrete = p_point.get_if<MultiDiscretePoint>(); multi_discrete && r_binding.kind == SpaceKind::MULTI_DISCRETE) {
		written = write_values(r_binding, multi_discrete->values);
	} else if (const MultiBinaryPoint *multi_binary = p_point.get_if<MultiBinaryPoint>(); multi_binary && r_binding.kind == SpaceKind::MULTI_BINARY) {
		written = write_values(r_binding, multi_binary->values);
	}
	if (!written) {
		return Status::error(StatusCode::INVALID_ARGUMENT, r_binding.label("observation") + " does not match its " + kind_name(r_binding.kind) + " space with " + std::to_string(r_binding.element_count) + " values");
	}
	return Status::ok();
}

Status read_integer(const Binding &p_binding, size_t p_index, int32_t &r_value) {
	const double value = load(p_binding.element_type, p_binding.buffer, p_index);
	if (value < 0.0 || value >= p_binding.bounds[p_index]) {
		return Status::error(StatusCode::RUNTIME_FAILURE,
				"the model returned " + std::to_string(static_cast<int64_t>(value)) + " for " + p_binding.label("action") + ", outside [0, " + std::to_string(p_binding.bounds[p_index]) + ")");
	}
	r_value = static_cast<int32_t>(value);
	return Status::ok();
}

Status read_action(const Binding &p_binding, Point &r_point) {
	if (BoxPoint *box = r_point.get_if<BoxPoint>(); box && p_binding.kind == SpaceKind::BOX && box->values.size() == p_binding.element_count) {
		for (size_t i = 0; i < box->values.size(); ++i) {
			box->values[i] = static_cast<float>(load(p_binding.element_type, p_binding.buffer, i));
		}
		return Status::ok();
	}
	if (DiscretePoint *discrete = r_point.get_if<DiscretePoint>(); discrete && p_binding.kind == SpaceKind::DISCRETE) {
		return read_integer(p_binding, 0, discrete->value);
	}
	if (MultiDiscretePoint *multi_discrete = r_point.get_if<MultiDiscretePoint>(); multi_discrete && p_binding.kind == SpaceKind::MULTI_DISCRETE && multi_discrete->values.size() == p_binding.element_count) {
		for (size_t i = 0; i < multi_discrete->values.size(); ++i) {
			Status status = read_integer(p_binding, i, multi_discrete->values[i]);
			if (!status.is_ok()) {
				return status;
			}
		}
		return Status::ok();
	}
	if (MultiBinaryPoint *multi_binary = r_point.get_if<MultiBinaryPoint>(); multi_binary && p_binding.kind == SpaceKind::MULTI_BINARY && multi_binary->values.size() == p_binding.element_count) {
		for (size_t i = 0; i < multi_binary->values.size(); ++i) {
			multi_binary->values[i] = load(p_binding.element_type, p_binding.buffer, i) != 0.0 ? 1 : 0;
		}
		return Status::ok();
	}
	return Status::error(StatusCode::INVALID_ARGUMENT, p_binding.label("action") + " point does not match its " + kind_name(p_binding.kind) + " space; create it with action_space.make_point()");
}

} // namespace

struct OnnxPolicy::Impl {
	OrtPtr<OrtSession> session;
	std::string model_name;
	std::vector<TensorDescription> inputs;
	std::vector<TensorDescription> outputs;
	std::vector<std::string> input_names;
	std::vector<std::string> output_names;
	std::vector<StateLayout> states;
	int32_t max_state_sequence_length = 1;
	OrtPtr<OrtMemoryInfo> memory_info;
	std::vector<Binding> observations;
	std::vector<Binding> actions;
	std::vector<const char *> run_input_names;
	std::vector<const char *> run_output_names;
	std::vector<const OrtValue *> run_inputs;
	std::vector<OrtValue *> run_outputs;
	std::vector<OrtPtr<OrtValue>> state_values;
	bool initialized = false;

	void clear_model() {
		session.reset();
		model_name.clear();
		inputs.clear();
		outputs.clear();
		input_names.clear();
		output_names.clear();
		states.clear();
		clear_bindings();
	}

	void clear_bindings() {
		initialized = false;
		observations.clear();
		actions.clear();
		run_input_names.clear();
		run_output_names.clear();
		run_inputs.clear();
		run_outputs.clear();
		state_values.clear();
	}

	Status error(StatusCode p_code, const std::string &p_message) const {
		return Status::error(p_code, "load_model(" + model_name + "): " + p_message);
	}

	Status check(OrtStatus *p_status) const {
		return from_ort(p_status, StatusCode::INVALID_DATA, "load_model(" + model_name + ")");
	}

	Status describe(bool p_inputs) {
		size_t count = 0;
		Status status = check(p_inputs ? ort().SessionGetInputCount(session.get(), &count) : ort().SessionGetOutputCount(session.get(), &count));
		OrtAllocator *allocator = nullptr;
		if (status.is_ok()) {
			status = check(ort().GetAllocatorWithDefaultOptions(&allocator));
		}

		std::vector<TensorDescription> &descriptions = p_inputs ? inputs : outputs;
		for (size_t i = 0; status.is_ok() && i < count; ++i) {
			TensorDescription description;
			char *name = nullptr;
			status = check(p_inputs ? ort().SessionGetInputName(session.get(), i, allocator, &name) : ort().SessionGetOutputName(session.get(), i, allocator, &name));
			if (!status.is_ok()) {
				break;
			}
			description.name = name;
			status = check(ort().AllocatorFree(allocator, name));

			OrtTypeInfo *raw_type_info = nullptr;
			if (status.is_ok()) {
				status = check(p_inputs ? ort().SessionGetInputTypeInfo(session.get(), i, &raw_type_info) : ort().SessionGetOutputTypeInfo(session.get(), i, &raw_type_info));
			}
			OrtPtr<OrtTypeInfo> type_info(raw_type_info);
			const OrtTensorTypeAndShapeInfo *tensor_info = nullptr;
			if (status.is_ok()) {
				status = check(ort().CastTypeInfoToTensorInfo(type_info.get(), &tensor_info));
			}
			if (status.is_ok() && tensor_info == nullptr) {
				status = error(StatusCode::INCOMPATIBLE, std::string(p_inputs ? "input '" : "output '") + description.name + "' is not a tensor");
			}

			size_t dimension_count = 0;
			if (status.is_ok()) {
				status = check(ort().GetTensorElementType(tensor_info, &description.element_type));
			}
			if (status.is_ok()) {
				status = check(ort().GetDimensionsCount(tensor_info, &dimension_count));
			}
			if (status.is_ok()) {
				description.shape.resize(dimension_count);
				status = check(ort().GetDimensions(tensor_info, description.shape.data(), dimension_count));
			}
			if (status.is_ok()) {
				descriptions.push_back(std::move(description));
			}
		}
		return status;
	}

	bool is_state_tensor(size_t p_index, bool p_input) const {
		for (const StateLayout &state : states) {
			if ((p_input ? state.input_index : state.output_index) == p_index) {
				return true;
			}
		}
		return false;
	}

	// Model tensors are named "obs" and "action", or after the Dict keys.
	Status bind(const Space &p_space, bool p_input) {
		const std::string context = "init(" + model_name + "): ";
		const char *tensor_kind = p_input ? "input" : "output";
		const char *space_kind = p_input ? "observation" : "action";
		std::vector<TensorDescription> &tensors = p_input ? inputs : outputs;
		std::vector<Binding> &bindings = p_input ? observations : actions;

		std::vector<std::pair<std::string, const Space *>> leaves;
		if (const DictSpace *dict = p_space.get_if<DictSpace>()) {
			for (const auto &entry : dict->entries) {
				leaves.emplace_back(entry.first, &entry.second);
			}
		} else {
			leaves.emplace_back("", &p_space);
		}

		for (const auto &[key, leaf] : leaves) {
			const std::string name = !key.empty() ? key : std::string(p_input ? "obs" : "action");
			size_t index = tensors.size();
			for (size_t i = 0; i < tensors.size(); ++i) {
				if (tensors[i].name == name && !is_state_tensor(i, p_input)) {
					index = i;
				}
			}
			if (index == tensors.size()) {
				return Status::error(StatusCode::INCOMPATIBLE, context + "the model has no " + tensor_kind + " '" + name + "' for the " + space_kind + " space");
			}

			Binding binding;
			binding.key = key;
			binding.tensor_index = index;
			std::vector<int64_t> dimensions;
			if (!leaf_shape(*leaf, binding, dimensions)) {
				return Status::error(StatusCode::INCOMPATIBLE, context + "nested Dict spaces are not supported, found one in the " + space_kind + " space");
			}
			const TensorDescription &tensor = tensors[index];
			if (!accepts(binding.kind, tensor.element_type, p_input)) {
				return Status::error(StatusCode::INCOMPATIBLE, context + tensor_kind + " '" + name + "' has ONNX element type " + std::to_string(tensor.element_type) + ", which does not fit a " + kind_name(binding.kind) + " space");
			}
			std::vector<int64_t> expected = { -1 };
			expected.insert(expected.end(), dimensions.begin(), dimensions.end());
			bool shape_matches = tensor.shape.size() == expected.size() && (tensor.shape[0] < 0 || tensor.shape[0] == 1);
			for (size_t i = 1; shape_matches && i < expected.size(); ++i) {
				shape_matches = tensor.shape[i] == expected[i];
			}
			if (!shape_matches) {
				return Status::error(StatusCode::INCOMPATIBLE, context + tensor_kind + " '" + name + "' has shape " + shape_text(tensor.shape) + ", but the " + space_kind + " space needs " + shape_text(expected));
			}

			binding.element_type = tensor.element_type;
			binding.shape = expected;
			binding.shape[0] = 1;
			binding.element_count = 1;
			for (int64_t dimension : dimensions) {
				binding.element_count *= static_cast<size_t>(dimension);
			}
			binding.buffer.assign(binding.element_count * element_size(binding.element_type), 0);
			bindings.push_back(std::move(binding));
		}

		for (size_t i = 0; i < tensors.size(); ++i) {
			bool bound = is_state_tensor(i, p_input);
			for (const Binding &binding : bindings) {
				bound = bound || binding.tensor_index == i;
			}
			if (!bound) {
				return Status::error(StatusCode::INCOMPATIBLE, context + "model " + tensor_kind + " '" + tensors[i].name + "' has no matching " + space_kind + " in the space");
			}
		}
		return Status::ok();
	}

	Status prepare_run() {
		const std::string context = "init(" + model_name + ")";
		if (memory_info == nullptr) {
			OrtMemoryInfo *raw_memory_info = nullptr;
			Status status = from_ort(ort().CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &raw_memory_info), StatusCode::RUNTIME_FAILURE, context);
			if (!status.is_ok()) {
				return status;
			}
			memory_info.reset(raw_memory_info);
		}

		for (std::vector<Binding> *bindings : { &observations, &actions }) {
			for (Binding &binding : *bindings) {
				OrtValue *raw_value = nullptr;
				Status status = from_ort(ort().CreateTensorWithDataAsOrtValue(memory_info.get(), binding.buffer.data(), binding.buffer.size(), binding.shape.data(), binding.shape.size(), binding.element_type, &raw_value),
						StatusCode::RUNTIME_FAILURE, context);
				if (!status.is_ok()) {
					return status;
				}
				binding.value.reset(raw_value);
			}
		}

		for (StateLayout &state : states) {
			state.input_shape = inputs[state.input_index].shape;
			state.input_shape[0] = 1;
			if (state.has_sequence) {
				state.input_shape[1] = max_state_sequence_length;
			}
			state.output_shape = outputs[state.output_index].shape;
			state.output_shape[0] = 1;
		}

		run_inputs.assign(inputs.size(), nullptr);
		for (const TensorDescription &input : inputs) {
			run_input_names.push_back(input.name.c_str());
		}
		for (const Binding &binding : observations) {
			run_inputs[binding.tensor_index] = binding.value.get();
		}
		for (const Binding &binding : actions) {
			run_output_names.push_back(outputs[binding.tensor_index].name.c_str());
			run_outputs.push_back(binding.value.get());
		}
		for (const StateLayout &state : states) {
			run_output_names.push_back(outputs[state.output_index].name.c_str());
			run_outputs.push_back(nullptr);
		}
		return Status::ok();
	}

	// Pairs state inputs with state outputs by order.
	Status find_states() {
		std::vector<size_t> state_inputs;
		std::vector<size_t> state_outputs;
		for (size_t i = 0; i < inputs.size(); ++i) {
			if (starts_with(inputs[i].name, "state_in")) {
				state_inputs.push_back(i);
			}
		}
		for (size_t i = 0; i < outputs.size(); ++i) {
			if (starts_with(outputs[i].name, "state_out")) {
				state_outputs.push_back(i);
			}
		}
		if (state_inputs.size() != state_outputs.size()) {
			return error(StatusCode::INCOMPATIBLE, std::to_string(state_inputs.size()) + " state_in tensors but " + std::to_string(state_outputs.size()) + " state_out tensors; expected one of each per state");
		}

		for (size_t pair = 0; pair < state_inputs.size(); ++pair) {
			const TensorDescription &input = inputs[state_inputs[pair]];
			const TensorDescription &output = outputs[state_outputs[pair]];
			if (input.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || output.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
				return error(StatusCode::INCOMPATIBLE, "state tensors '" + input.name + "' and '" + output.name + "' must be float");
			}
			if (input.shape.size() < 2 || output.shape.size() < 2) {
				return error(StatusCode::INCOMPATIBLE, "state tensors '" + input.name + "' and '" + output.name + "' need a batch and a state dimension");
			}

			StateLayout layout;
			layout.input_index = state_inputs[pair];
			layout.output_index = state_outputs[pair];
			layout.state_size = 1;
			for (size_t dimension = 1; dimension < input.shape.size(); ++dimension) {
				if (input.shape[dimension] > 0) {
					layout.state_size *= input.shape[dimension];
				} else if (dimension == 1) {
					layout.has_sequence = true; // A dynamic dimension right after the batch is the sequence.
				} else {
					return error(StatusCode::INCOMPATIBLE, "state input '" + input.name + "' has a dynamic dimension at index " + std::to_string(dimension) + "; only a sequence right after the batch is supported");
				}
			}

			int64_t output_size = 1;
			for (size_t dimension = 1; dimension < output.shape.size(); ++dimension) {
				if (output.shape[dimension] <= 0) {
					return error(StatusCode::INCOMPATIBLE, "state output '" + output.name + "' has a dynamic dimension besides the batch");
				}
				output_size *= output.shape[dimension];
			}
			if (output_size != layout.state_size) {
				return error(StatusCode::INCOMPATIBLE, "state output '" + output.name + "' holds " + std::to_string(output_size) + " values per entry, but '" + input.name + "' holds " + std::to_string(layout.state_size));
			}
			states.push_back(layout);
		}
		return Status::ok();
	}
};

OnnxPolicy::OnnxPolicy() :
		impl(std::make_unique<Impl>()) {}

OnnxPolicy::~OnnxPolicy() = default;

Status OnnxPolicy::load_model(const uint8_t *p_data, size_t p_size, const std::string &p_model_name) {
	impl->clear_model();
	impl->model_name = p_model_name;
	if (p_data == nullptr || p_size == 0) {
		return impl->error(StatusCode::INVALID_ARGUMENT, "no model data");
	}
	const SharedEnvironment &environment = shared_environment();
	if (!environment.status.is_ok()) {
		return environment.status;
	}

	OrtSessionOptions *raw_options = nullptr;
	Status status = from_ort(ort().CreateSessionOptions(&raw_options), StatusCode::RUNTIME_FAILURE, "load_model(" + p_model_name + ")");
	OrtPtr<OrtSessionOptions> options(raw_options);
	if (status.is_ok()) {
		status = from_ort(ort().SetIntraOpNumThreads(options.get(), 1), StatusCode::RUNTIME_FAILURE, "load_model(" + p_model_name + ")");
	}
	if (status.is_ok()) {
		status = from_ort(ort().SetInterOpNumThreads(options.get(), 1), StatusCode::RUNTIME_FAILURE, "load_model(" + p_model_name + ")");
	}
	if (status.is_ok()) {
		status = from_ort(ort().SetSessionGraphOptimizationLevel(options.get(), ORT_ENABLE_ALL), StatusCode::RUNTIME_FAILURE, "load_model(" + p_model_name + ")");
	}

	OrtSession *raw_session = nullptr;
	if (status.is_ok()) {
		status = impl->check(ort().CreateSessionFromArray(environment.env, p_data, p_size, options.get(), &raw_session));
	}
	impl->session.reset(raw_session);

	if (status.is_ok()) {
		status = impl->describe(true);
	}
	if (status.is_ok()) {
		status = impl->describe(false);
	}
	if (status.is_ok()) {
		status = impl->find_states();
	}
	if (!status.is_ok()) {
		impl->clear_model();
		return status;
	}
	for (const TensorDescription &input : impl->inputs) {
		impl->input_names.push_back(input.name);
	}
	for (const TensorDescription &output : impl->outputs) {
		impl->output_names.push_back(output.name);
	}
	return Status::ok();
}

Status OnnxPolicy::set_max_state_sequence_length(int32_t p_length) {
	if (p_length < 1) {
		return Status::error(StatusCode::INVALID_ARGUMENT, "set_max_state_sequence_length: length must be at least 1, got " + std::to_string(p_length));
	}
	if (impl->initialized) {
		return Status::error(StatusCode::FAILED_PRECONDITION, "set_max_state_sequence_length: call before init()");
	}
	impl->max_state_sequence_length = p_length;
	return Status::ok();
}

int32_t OnnxPolicy::get_max_state_sequence_length() const {
	return impl->max_state_sequence_length;
}

const std::vector<std::string> &OnnxPolicy::get_input_names() const {
	return impl->input_names;
}

const std::vector<std::string> &OnnxPolicy::get_output_names() const {
	return impl->output_names;
}

Status OnnxPolicy::init(const InteractionDefinition &p_definition) {
	impl->clear_bindings();
	if (impl->session == nullptr) {
		return Status::error(StatusCode::FAILED_PRECONDITION, "OnnxPolicy::init: call load_model() first");
	}
	Status status = p_definition.observation_space.check_definition();
	if (status.is_ok()) {
		status = p_definition.action_space.check_definition();
	}
	if (status.is_ok()) {
		status = impl->bind(p_definition.observation_space, true);
	}
	if (status.is_ok()) {
		status = impl->bind(p_definition.action_space, false);
	}
	if (status.is_ok()) {
		status = impl->prepare_run();
	}
	if (!status.is_ok()) {
		impl->clear_bindings();
		return status;
	}
	impl->initialized = true;
	return Status::ok();
}

Status OnnxPolicy::create_initial_state(std::unique_ptr<PolicyState> &r_state) const {
	if (!is_ready()) {
		return Status::error(StatusCode::FAILED_PRECONDITION, "OnnxPolicy::create_initial_state: call load_model() and init() first");
	}
	if (impl->states.empty()) {
		return Policy::create_initial_state(r_state);
	}
	std::vector<OnnxPolicyState::BufferLayout> layouts;
	for (const StateLayout &state : impl->states) {
		layouts.push_back({ impl->inputs[state.input_index].name, state.has_sequence ? impl->max_state_sequence_length : 1, state.state_size });
	}
	r_state = std::make_unique<OnnxPolicyState>(this, layouts);
	return Status::ok();
}

Status OnnxPolicy::think(const Point &p_observation, PolicyState &p_state, Point &r_action) {
	if (!is_ready()) {
		return Status::error(StatusCode::FAILED_PRECONDITION, "OnnxPolicy::think: call load_model() and init() first");
	}
	const std::string context = "think(" + impl->model_name + "): ";
	OnnxPolicyState *state = nullptr;
	if (is_recurrent()) {
		state = dynamic_cast<OnnxPolicyState *>(&p_state);
		if (state == nullptr || state->get_owner() != this) {
			return Status::error(StatusCode::INVALID_ARGUMENT, context + "the state was not created by this policy");
		}
		const std::vector<OnnxPolicyState::Buffer> &buffers = state->get_buffers();
		bool matches = buffers.size() == impl->states.size();
		for (size_t i = 0; matches && i < buffers.size(); ++i) {
			const StateLayout &layout = impl->states[i];
			matches = buffers[i].state_size == layout.state_size && buffers[i].sequence_length == (layout.has_sequence ? impl->max_state_sequence_length : 1);
		}
		if (!matches) {
			return Status::error(StatusCode::INVALID_ARGUMENT, context + "the state no longer matches the model; create a new state after reloading");
		}
	} else if (dynamic_cast<EmptyPolicyState *>(&p_state) == nullptr) {
		return Status::error(StatusCode::INVALID_ARGUMENT, context + "the state was not created by this policy");
	}

	for (Binding &binding : impl->observations) {
		const Point *leaf = find_leaf(p_observation, binding.key);
		Status status = leaf == nullptr ? Status::error(StatusCode::INVALID_ARGUMENT, binding.label("observation") + " is missing") : write_observation(binding, *leaf);
		if (!status.is_ok()) {
			return Status::error(status.get_code(), context + status.get_message());
		}
	}

	impl->state_values.clear();
	const size_t first_state_output = impl->actions.size();
	for (size_t i = 0; state != nullptr && i < impl->states.size(); ++i) {
		StateLayout &layout = impl->states[i];
		OnnxPolicyState::Buffer &buffer = state->get_buffers()[i];
		OrtValue *sequence = nullptr;
		OrtValue *pending = nullptr;
		Status status = from_ort(ort().CreateTensorWithDataAsOrtValue(impl->memory_info.get(), buffer.sequence(), static_cast<size_t>(buffer.sequence_length * buffer.state_size) * sizeof(float),
										 layout.input_shape.data(), layout.input_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &sequence),
				StatusCode::RUNTIME_FAILURE, context);
		impl->state_values.emplace_back(sequence);
		if (status.is_ok()) {
			status = from_ort(ort().CreateTensorWithDataAsOrtValue(impl->memory_info.get(), buffer.pending(), static_cast<size_t>(buffer.state_size) * sizeof(float),
									  layout.output_shape.data(), layout.output_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &pending),
					StatusCode::RUNTIME_FAILURE, context);
		}
		impl->state_values.emplace_back(pending);
		if (!status.is_ok()) {
			return status;
		}
		impl->run_inputs[layout.input_index] = sequence;
		impl->run_outputs[first_state_output + i] = pending;
	}

	Status status = from_ort(ort().Run(impl->session.get(), nullptr, impl->run_input_names.data(), impl->run_inputs.data(), impl->run_inputs.size(),
									 impl->run_output_names.data(), impl->run_output_names.size(), impl->run_outputs.data()),
			StatusCode::RUNTIME_FAILURE, context);
	if (!status.is_ok()) {
		return status;
	}

	for (const Binding &binding : impl->actions) {
		Point *leaf = find_leaf(r_action, binding.key);
		status = leaf == nullptr ? Status::error(StatusCode::INVALID_ARGUMENT, binding.label("action") + " is missing") : read_action(binding, *leaf);
		if (!status.is_ok()) {
			return Status::error(status.get_code(), context + status.get_message());
		}
	}
	return state == nullptr ? Status::ok() : state->notify_state_update();
}

Status OnnxPolicy::batched_think(const std::vector<const Point *> &p_observations, const std::vector<PolicyState *> &p_states, const std::vector<Point *> &r_actions) {
	// One run per entry for now; a single batched run is a later optimization.
	return Policy::batched_think(p_observations, p_states, r_actions);
}

bool OnnxPolicy::is_recurrent() const {
	return !impl->states.empty();
}

bool OnnxPolicy::is_ready() const {
	return impl->session != nullptr && impl->initialized;
}

} // namespace schola
