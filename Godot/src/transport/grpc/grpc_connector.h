// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

// Dependency on the US3 gRPC/Protocol Buffers compatibility spike
#if !__has_include(<grpcpp/grpcpp.h>)
#error "US3 needs gRPC; its version is selected by the US3 compatibility spike"
#endif

#include "core/connector/connector.h"

#include <grpcpp/grpcpp.h>

#include <cstdint>
#include <memory>
#include <string>

namespace schola {

struct GrpcConnectorSettings {
	// Same default as the Unreal plugin.
	std::string address = "127.0.0.1";

	// Same default as the Unreal plugin; -ScholaPort=<n> overrides it.
	int32_t port = 8000;
};

/**
 * Serves Schola.GymService (Proto/GymConnector.proto) to the unmodified Python
 * client.
 *
 * gRPC handles each call on one of its worker threads. Those threads only
 * convert and queue messages, then wait for the main thread's reply; they
 * never touch Godot or a TrainingBackend.
 */
class GrpcConnector final : public Connector {
public:
	explicit GrpcConnector(GrpcConnectorSettings p_settings);

	// Calls close().
	~GrpcConnector() override;

	Status open() override;
	Status wait_for_start(StartRequest &r_request, std::chrono::milliseconds p_timeout) override;
	Status reply_start(const Status &p_result) override;
	Status publish_definition(const TrainingDefinition &p_definition) override;
	Status wait_for_update(ConnectorUpdate &r_update, std::chrono::milliseconds p_timeout) override;
	Status poll_update(ConnectorUpdate &r_update) override;
	Status reply_reset(const InitialState &p_initial_state) override;
	Status reply_step(const StepResult &p_result) override;
	void fail(const Status &p_error) override;
	void close() override;

private:
	// Implements the generated GymService. Defined in grpc_connector.cpp so
	// the generated headers stay out of this one.
	class Service;

	GrpcConnectorSettings settings;
	std::unique_ptr<Service> service;
	std::unique_ptr<grpc::Server> server;
};

} // namespace schola
