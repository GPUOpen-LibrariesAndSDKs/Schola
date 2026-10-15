// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

// Dependency on US4
#if !__has_include("core/lifecycle/training_backend.h")
#error "US3 needs TrainingBackend and the training types from core/lifecycle/"
#endif

#include "core/common/status.h"
#include "core/lifecycle/training_backend.h"
#include "core/lifecycle/training_types.h"

#include <chrono>

namespace schola {

/**
 * What Python asks for in StartGymConnector. Mirrors GymConnectorStartRequest
 * (Proto/GymConnector.proto).
 */
struct StartRequest {
	// SAME_STEP is the protocol's default value.
	AutoResetMode autoreset_mode = AutoResetMode::SAME_STEP;
};

/** The operation carried by one UpdateState message. */
enum class UpdateKind {
	STEP,
	RESET,
	CLOSE,
};

/** One UpdateState message, converted out of protobuf. */
struct ConnectorUpdate {
	UpdateKind kind = UpdateKind::STEP;

	// Valid only when kind is STEP.
	StepRequest step;

	// Valid only when kind is RESET.
	ResetRequest reset;
};

/**
 * The training-time source of actions: Python, seen without any protocol.
 *
 * Implementations receive Python's calls on their own threads and queue them.
 * Every method below is called on the Godot main thread, so an implementation
 * never calls into Godot or a TrainingBackend itself. GrpcConnector implements
 * it; unit tests use a fake.
 *
 * The existing Python client fixes the call order: StartGymConnector, then
 * RequestTrainingDefinition, then UpdateState until the session closes. Each
 * call stays pending until the matching reply method answers it.
 */
class Connector {
public:
	virtual ~Connector() = default;

	/**
	 * Starts listening for Python.
	 *
	 * Out: RUNTIME_FAILURE naming the address if it cannot listen, for
	 *      example because the port is already in use.
	 */
	virtual Status open() = 0;

	/**
	 * Blocks until Python calls StartGymConnector.
	 *
	 * Out: r_request filled with Python's settings. TIMEOUT if Python did not
	 *      connect within p_timeout; INVALID_ARGUMENT if it sent an auto-reset
	 *      mode this build does not recognize.
	 */
	virtual Status wait_for_start(StartRequest &r_request, std::chrono::milliseconds p_timeout) = 0;

	/**
	 * Answers the pending StartGymConnector with p_result, the backend's
	 * result from set_autoreset_mode. A non-OK result reaches Python as an
	 * error.
	 */
	virtual Status reply_start(const Status &p_result) = 0;

	/**
	 * Answers RequestTrainingDefinition with p_definition, now and for any
	 * later request in this session. Until it is called, Python's request
	 * waits.
	 */
	virtual Status publish_definition(const TrainingDefinition &p_definition) = 0;

	/**
	 * Blocks until Python's next UpdateState.
	 *
	 * Out: r_update filled. TIMEOUT after p_timeout (zero waits
	 *      indefinitely); CLOSED if Python disconnected or reported an error,
	 *      with Python's report in the message.
	 */
	virtual Status wait_for_update(ConnectorUpdate &r_update, std::chrono::milliseconds p_timeout) = 0;

	/**
	 * Never blocks.
	 *
	 * Out: OK with r_update filled if an update has arrived, NOT_FOUND if
	 *      not yet, CLOSED as for wait_for_update.
	 */
	virtual Status poll_update(ConnectorUpdate &r_update) = 0;

	/** Answers a pending RESET update. */
	virtual Status reply_reset(const InitialState &p_initial_state) = 0;

	/**
	 * Answers a pending STEP update. p_result.initial_state carries any
	 * environments reset by SAME_STEP auto-reset.
	 */
	virtual Status reply_step(const StepResult &p_result) = 0;

	/**
	 * Answers whichever call is pending with p_error, so Python raises it,
	 * then rejects every later call.
	 */
	virtual void fail(const Status &p_error) = 0;

	/**
	 * Stops listening and joins worker threads. Safe after a partial open()
	 * and safe to call twice.
	 */
	virtual void close() = 0;
};

/**
 * Runs one training session on the main thread: takes Python's messages from
 * a Connector and executes them on a TrainingBackend.
 */
class ConnectorLoop {
public:
	/**
	 * In:   p_connector and p_backend are not owned and must outlive the loop.
	 *
	 * Does: in Python's order: open, wait_for_start, set_autoreset_mode,
	 *       reply_start, define, publish_definition.
	 *
	 * Out:  the first failure, after it has been reported to Python through
	 *       Connector::fail.
	 */
	Status start(Connector &p_connector, TrainingBackend &p_backend, std::chrono::milliseconds p_start_timeout);

	/**
	 * Headless mode, once per physics frame: waits for an update, runs it on
	 * the backend, and replies.
	 *
	 * Out: CLOSED once Python ends the session; any other failure after it
	 *      has been reported to Python.
	 */
	Status tick(std::chrono::milliseconds p_update_timeout);

	/**
	 * Windowed mode: runs an update if one has arrived, otherwise returns
	 * NOT_FOUND immediately so the window stays responsive.
	 *
	 * Out: TIMEOUT once no update has arrived for p_update_timeout.
	 */
	Status poll_tick(std::chrono::milliseconds p_update_timeout);

	/** Closes the backend and the connector. Safe to call twice. */
	void stop();

private:
	Connector *connector = nullptr; // Not owned.
	TrainingBackend *backend = nullptr; // Not owned.
	std::chrono::steady_clock::time_point last_update;
};

} // namespace schola
