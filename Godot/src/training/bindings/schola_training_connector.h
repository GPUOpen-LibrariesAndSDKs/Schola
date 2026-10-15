// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "core/connector/connector.h"

#include <godot_cpp/classes/node.hpp>

#include <cstdint>
#include <memory>

/**
 * Runs a training session for the Godot scene.
 *
 * Created in code by ScholaTrainingHost (proposed for US4 in
 * Godot/docs/api_overview.md, section 4.5) when training is requested. It is
 * never placed in a scene, so its settings come from project settings rather
 * than the Inspector. _ready() starts the session; each physics frame runs one
 * Python update.
 *
 * Headless games block each frame until Python's next message. Windowed
 * games poll instead and pause the scene tree while waiting, so the window
 * stays responsive.
 */
class ScholaTrainingConnector : public godot::Node {
	GDCLASS(ScholaTrainingConnector, godot::Node)

public:
	// -ScholaPort=<n>, otherwise the schola/training/port project setting.
	void set_port(int32_t p_port);
	int32_t get_port() const;

	// schola/training/start_timeout_sec: how long to wait for Python to connect.
	// The default matches the Python client's environment_start_timeout.
	void set_start_timeout_sec(double p_seconds);
	double get_start_timeout_sec() const;

	// schola/training/step_timeout_sec: how long to wait between Python updates.
	void set_step_timeout_sec(double p_seconds);
	double get_step_timeout_sec() const;

	void _ready() override;
	void _physics_process(double p_delta) override;
	void _exit_tree() override;

protected:
	// Binds the properties and the training_finished(error: String) signal,
	// emitted once with an empty string when Python closes normally.
	static void _bind_methods();

private:
	int32_t port = 8000;
	double start_timeout_sec = 45.0;
	double step_timeout_sec = 60.0;
	std::unique_ptr<schola::Connector> connector;
	std::unique_ptr<schola::TrainingBackend> backend;
	schola::ConnectorLoop loop;
};
