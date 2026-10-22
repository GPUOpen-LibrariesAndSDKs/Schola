// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "inference/onnx_policy_state.h"

#include <algorithm>
#include <string>

namespace schola {

OnnxPolicyState::OnnxPolicyState(const OnnxPolicy *p_owner, const std::vector<BufferLayout> &p_layouts) :
		owner(p_owner) {
	buffers.reserve(p_layouts.size());
	for (const BufferLayout &layout : p_layouts) {
		Buffer buffer;
		buffer.name = layout.name;
		buffer.sequence_length = layout.sequence_length;
		buffer.state_size = layout.state_size;
		buffer.values.assign(static_cast<size_t>((layout.sequence_length + 1) * layout.state_size), 0.0f);
		buffers.push_back(std::move(buffer));
	}
}

void OnnxPolicyState::reset() {
	for (Buffer &buffer : buffers) {
		std::fill(buffer.values.begin(), buffer.values.end(), 0.0f);
	}
}

Status OnnxPolicyState::copy_from(const PolicyState &p_other) {
	const OnnxPolicyState *other = dynamic_cast<const OnnxPolicyState *>(&p_other);
	if (other == nullptr) {
		return Status::error(StatusCode::INVALID_ARGUMENT, "OnnxPolicyState::copy_from: expected an OnnxPolicyState, got " + p_other.to_string());
	}
	if (other->buffers.size() != buffers.size()) {
		return Status::error(StatusCode::INVALID_ARGUMENT, "OnnxPolicyState::copy_from: " + other->to_string() + " does not match " + to_string());
	}
	for (size_t i = 0; i < buffers.size(); ++i) {
		if (other->buffers[i].sequence_length != buffers[i].sequence_length || other->buffers[i].state_size != buffers[i].state_size) {
			return Status::error(StatusCode::INVALID_ARGUMENT, "OnnxPolicyState::copy_from: " + other->to_string() + " does not match " + to_string());
		}
	}
	// Copies in place: model tensors point into these buffers, so they must never be reallocated.
	for (size_t i = 0; i < buffers.size(); ++i) {
		std::copy(other->buffers[i].values.begin(), other->buffers[i].values.end(), buffers[i].values.begin());
	}
	return Status::ok();
}

Status OnnxPolicyState::notify_state_update() {
	for (Buffer &buffer : buffers) {
		float *sequence = buffer.sequence();
		const int64_t size = buffer.state_size;
		std::copy(sequence + size, sequence + buffer.sequence_length * size, sequence);
		std::copy(buffer.pending(), buffer.pending() + size, sequence + (buffer.sequence_length - 1) * size);
	}
	return Status::ok();
}

std::string OnnxPolicyState::to_string() const {
	std::string text = "OnnxPolicyState(";
	for (size_t i = 0; i < buffers.size(); ++i) {
		text += (i == 0 ? "" : ", ") + buffers[i].name + ": " + std::to_string(buffers[i].sequence_length) + "x" + std::to_string(buffers[i].state_size);
	}
	return text + ")";
}

const OnnxPolicy *OnnxPolicyState::get_owner() const {
	return owner;
}

std::vector<OnnxPolicyState::Buffer> &OnnxPolicyState::get_buffers() {
	return buffers;
}

const std::vector<OnnxPolicyState::Buffer> &OnnxPolicyState::get_buffers() const {
	return buffers;
}

} // namespace schola
