// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

// Dependency on the US3 gRPC/Protocol Buffers compatibility spike
#if !__has_include("GymConnector.pb.h")
#error "US3 needs Protocol Buffer bindings generated into Godot/generated/proto/"
#endif

#include "core/connector/connector.h"

#include "Definitions.pb.h"
#include "GymConnector.pb.h"
#include "State.pb.h"
#include "StateUpdates.pb.h"
#include <grpcpp/support/status.h>

#include <string_view>

namespace schola {

// Converts the GymService messages. Spaces and points inside them use US2's
// protobuf conversions.

/**
 * StartGymConnector request.
 *
 * Out: INVALID_ARGUMENT naming the value if autoreset_type is not one this
 *      build recognizes, for example one sent by a newer Python client.
 */
Status from_proto(const ::Schola::GymConnectorStartRequest &p_message, StartRequest &r_request);

/** RequestTrainingDefinition reply. */
void to_proto(const TrainingDefinition &p_definition, ::Schola::TrainingDefinition &r_message);

/**
 * UpdateState request.
 *
 * Out: Python's CLOSED status becomes UpdateKind::CLOSE. Its ERROR status
 *      returns CLOSED carrying Python's report. Points pass through without
 *      space validation; Python validates them.
 */
Status from_proto(const ::Schola::StateUpdate &p_message, ConnectorUpdate &r_update);

/** UpdateState reply after a reset. */
void to_proto(const InitialState &p_initial_state, ::Schola::State &r_message);

/** UpdateState reply after a step. */
void to_proto(const StepResult &p_result, ::Schola::State &r_message);

/**
 * Converts a Status into the gRPC status Python receives, prefixing the
 * message with p_rpc so Python's error names the call that failed.
 *
 * OK stays OK. Otherwise:
 *   INVALID_ARGUMENT, INVALID_DATA -> INVALID_ARGUMENT
 *   NOT_FOUND                      -> NOT_FOUND
 *   FAILED_PRECONDITION            -> FAILED_PRECONDITION
 *   INCOMPATIBLE                   -> FAILED_PRECONDITION
 *   TIMEOUT                        -> DEADLINE_EXCEEDED
 *   CLOSED                         -> UNAVAILABLE
 *   RUNTIME_FAILURE                -> INTERNAL
 */
grpc::Status to_grpc_status(const Status &p_status, std::string_view p_rpc);

} // namespace schola
