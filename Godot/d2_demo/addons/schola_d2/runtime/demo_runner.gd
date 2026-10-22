class_name ScholaDemoRunner
extends Node
## Hands control to the keyboard, the Python trainer (JSON over TCP, a US3 stand-in) or a saved policy.

signal mode_changed(mode: Mode)
signal log_message(text: String, is_error: bool)
signal trainer_progress(stats: Dictionary)
signal action_chosen(action: Variant, source: String)

enum Mode { MANUAL, WAITING, TRAINING, STOPPING, POLICY }

const FRAME_WAIT_BUDGET_USEC := 10000
const GROUP := "schola_demo_runner"

@export var environment: ScholaDemoEnvironment
@export_group("Training client")
@export var port := 5757
@export var auto_launch_trainer := true
@export var python_executable := "python"
@export_file("*.py") var trainer_script := "res://python/trainer.py"
@export var trainer_episodes := 300
@export_group("Saved policy")
@export var policy_path := "res://policies/q_policy.json"
@export_group("Pacing")
@export_range(1, 2000) var steps_per_second := 20.0
@export var reset_delay := 0.8

var mode := Mode.MANUAL
var policy: ScholaSavedPolicy

var _server: TCPServer
var _peer: StreamPeerTCP
var _buffer := PackedByteArray()
var _pending: Array[Dictionary] = []
var _pid := -1
var _step_budget := 0.0
var _reset_timer := 0.0
var _stop_deadline := 0.0


func _ready() -> void:
	add_to_group(GROUP)
	environment.reset_episode()


func _exit_tree() -> void:
	_close_connection()
	_kill_trainer()


func start_training() -> void:
	if mode in [Mode.WAITING, Mode.TRAINING, Mode.STOPPING]:
		return
	policy = null
	_server = TCPServer.new()
	var err := _server.listen(port, "127.0.0.1")
	if err != OK:
		_log("could not listen on 127.0.0.1:%d (%s)" % [port, error_string(err)], true)
		_server = null
		return
	_set_mode(Mode.WAITING)
	_log("listening on 127.0.0.1:%d for the Python trainer" % port)
	if not auto_launch_trainer:
		_log("run: python python/trainer.py --port %d" % port)
		return
	var args := PackedStringArray([
		ProjectSettings.globalize_path(trainer_script),
		"--port", str(port),
		"--episodes", str(trainer_episodes),
		"--policy-out", ProjectSettings.globalize_path(policy_path),
	])
	_pid = OS.create_process(python_executable, args, true)
	if _pid == -1:
		_log("could not launch '%s'; run python/trainer.py yourself" % python_executable, true)
	else:
		_log("launched trainer (pid %d)" % _pid)


func stop_run() -> void:
	if mode == Mode.TRAINING and _peer != null:
		_send({"type": "stop"})
		_set_mode(Mode.STOPPING)
		_stop_deadline = Time.get_ticks_msec() / 1000.0 + 3.0
		_log("asked the trainer to save and stop")
		return
	_close_connection()
	_kill_trainer()
	policy = null
	_set_mode(Mode.MANUAL)
	environment.reset_episode()
	_log("run reset; keyboard controls the agent")


func load_saved_policy() -> void:
	if mode in [Mode.WAITING, Mode.TRAINING, Mode.STOPPING]:
		_log("stop training before loading a policy", true)
		return
	var loaded := ScholaSavedPolicy.new()
	var path := ProjectSettings.globalize_path(policy_path)
	var err := loaded.load_policy(path, environment.get_agent())
	if err != "":
		_log(err, true)
		return
	policy = loaded
	_set_mode(Mode.POLICY)
	environment.reset_episode()
	_log("loaded saved policy: %d states, %d training episodes" % [loaded.state_count(), loaded.episodes_trained])


func run_validation_check() -> void:
	var agent := environment.get_agent()
	var cases := [
		[agent.action_space, 7, "action 7"],
		[agent.action_space, 1.5, "action 1.5 (float)"],
		[agent.observation_space, PackedFloat32Array([1, 2, 3]), "3-value observation"],
		[agent.observation_space, PackedFloat32Array([9, 0, 0, 0]), "observation x = 9"],
		[agent.observation_space, PackedFloat32Array([NAN, 0, 0, 0]), "NaN observation"],
		[agent.observation_space, [0, 0, 0, 0], "Array instead of PackedFloat32Array"],
	]
	_log("US2 validation check:")
	for case in cases:
		var error: String = case[0].get_validation_error(case[1])
		_log("  %s -> %s" % [case[2], error if error != "" else "accepted"], error != "", false)
	var good := agent.observation_space.get_validation_error(agent._observe())
	_log("  current observation -> %s" % (good if good != "" else "accepted"), good != "")


func _unhandled_input(event: InputEvent) -> void:
	if mode != Mode.MANUAL or environment.is_episode_over():
		return
	var actions := {"ui_up": 0, "ui_down": 1, "ui_left": 2, "ui_right": 3}
	for input_name in actions:
		if event.is_action_pressed(input_name):
			_step(actions[input_name], "keyboard")
			get_viewport().set_input_as_handled()
			return


func _process(delta: float) -> void:
	_poll_network()
	if mode == Mode.STOPPING and Time.get_ticks_msec() / 1000.0 > _stop_deadline:
		_log("trainer did not stop in time; closing", true)
		stop_run()

	if mode in [Mode.MANUAL, Mode.POLICY] and environment.is_episode_over():
		_reset_timer += delta
		if _reset_timer >= reset_delay:
			_reset_timer = 0.0
			environment.reset_episode()
		return

	_step_budget += delta * steps_per_second
	var frame_deadline := Time.get_ticks_usec() + FRAME_WAIT_BUDGET_USEC
	while _step_budget >= 1.0:
		if not _run_one_step(frame_deadline):
			break
		_step_budget -= 1.0
	_step_budget = minf(_step_budget, 1.0)


func _run_one_step(frame_deadline: int) -> bool:
	match mode:
		Mode.TRAINING:
			while _pending.is_empty() and _peer != null and Time.get_ticks_usec() < frame_deadline:
				OS.delay_usec(50)
				_poll_network()
			if _pending.is_empty() or mode != Mode.TRAINING:
				return false
			_handle_request(_pending.pop_front())
			return true
		Mode.POLICY:
			if environment.is_episode_over():
				return false
			_step(policy.act(environment.last_observation), "saved policy")
			return true
	return false


func _step(action: Variant, source: String) -> Dictionary:
	action_chosen.emit(action, source)
	var result := environment.step(action)
	if result.get("error", "") != "":
		_log(result.error, true)
	return result


func _handle_request(request: Dictionary) -> void:
	match request.get("type"):
		"reset":
			var result := environment.reset_episode(request.get("seed"))
			if result.error != "":
				_send({"type": "error", "message": result.error})
				_log(result.error, true)
			else:
				_send({"type": "reset_result", "observation": Array(result.observation), "info": {}})
		"step":
			var action: Variant = request.get("action")
			# JSON numbers arrive as floats.
			if typeof(action) == TYPE_FLOAT and action == floor(action):
				action = int(action)
			var result := _step(action, "Python trainer")
			if result.get("error", "") != "":
				_send({"type": "error", "message": result.error})
			else:
				_send({
					"type": "step_result",
					"observation": Array(result.observation),
					"reward": result.reward,
					"terminated": result.terminated,
					"truncated": result.truncated,
					"info": {"outcome": result.outcome},
				})


func _poll_network() -> void:
	if _server != null and _peer == null and _server.is_connection_available():
		_peer = _server.take_connection()
		_peer.set_no_delay(true)
		_server.stop()
		_server = null
		_set_mode(Mode.TRAINING)
		var agent := environment.get_agent()
		_send({
			"type": "hello",
			"agent_id": agent.agent_id,
			"observation_space": agent.observation_space.to_dictionary(),
			"action_space": agent.action_space.to_dictionary(),
			"config": environment.get_config(),
		})
		_log("trainer connected; sent US2 space definitions")

	if _peer == null:
		return
	_peer.poll()
	if _peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		_log("trainer disconnected")
		_close_connection()
		_set_mode(Mode.MANUAL)
		environment.reset_episode()
		return
	var available := _peer.get_available_bytes()
	if available > 0:
		var received := _peer.get_data(available)
		if received[0] == OK:
			_buffer.append_array(received[1])
	var newline := _buffer.find(10)
	while newline != -1:
		var line := _buffer.slice(0, newline).get_string_from_utf8()
		_buffer = _buffer.slice(newline + 1)
		_handle_message(line)
		newline = _buffer.find(10)


func _handle_message(line: String) -> void:
	var message: Variant = JSON.parse_string(line)
	if typeof(message) != TYPE_DICTIONARY:
		_log("bad message from trainer: %s" % line, true)
		return
	match message.get("type"):
		"reset", "step":
			_pending.append(message)
		"progress":
			trainer_progress.emit(message)
		"saved":
			_log("trainer saved policy to %s" % message.get("path"))
		"log":
			_log("python: %s" % message.get("message"))
		_:
			_log("unknown message type from trainer: %s" % message.get("type"), true)


func _send(message: Dictionary) -> void:
	if _peer != null:
		_peer.put_data((JSON.stringify(message) + "\n").to_utf8_buffer())


func _close_connection() -> void:
	if _peer != null:
		_peer.disconnect_from_host()
		_peer = null
	if _server != null:
		_server.stop()
		_server = null
	_buffer.clear()
	_pending.clear()


func _kill_trainer() -> void:
	if _pid != -1 and OS.is_process_running(_pid):
		OS.kill(_pid)
	_pid = -1


func _set_mode(new_mode: Mode) -> void:
	mode = new_mode
	_reset_timer = 0.0
	mode_changed.emit(mode)


func _log(text: String, is_error := false, warn := true) -> void:
	log_message.emit(text, is_error)
	if is_error and warn:
		push_warning("Schola D2: " + text)
	else:
		print("Schola D2: " + text)
