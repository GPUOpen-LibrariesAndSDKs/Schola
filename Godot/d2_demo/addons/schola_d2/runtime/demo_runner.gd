class_name ScholaDemoRunner
extends Node
## Hands control to the keyboard, the Python trainer or a saved policy. Training goes through
## ScholaLocalConnector, the D2 stand-in for US3's connector; this script plays the part of US3's
## ConnectorLoop and runs each of Python's requests on the environment.

signal mode_changed(mode: Mode)
signal log_message(text: String, is_error: bool)
signal trainer_progress(stats: Dictionary)
signal action_chosen(action: Variant, source: String)
signal connection_state_changed(state: ScholaLocalConnector.State, message: String)

enum Mode { MANUAL, WAITING, TRAINING, STOPPING, POLICY }

const FRAME_WAIT_BUDGET_USEC := 10000
const GROUP := "schola_demo_runner"

@export var environment: ScholaDemoEnvironment
@export_group("Training client")
@export var port := 8000
@export var start_timeout_sec := 45.0
@export var step_timeout_sec := 60.0
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
var connector: ScholaLocalConnector
var connection_state := ScholaLocalConnector.State.IDLE

var _pid := -1
var _step_budget := 0.0
var _reset_timer := 0.0
var _stop_deadline := 0.0


func _ready() -> void:
	add_to_group(GROUP)
	environment.reset_episode()


func _exit_tree() -> void:
	if connector != null:
		connector.close("Godot is shutting down.", false)
	_kill_trainer()


func start_training() -> void:
	if mode in [Mode.WAITING, Mode.TRAINING, Mode.STOPPING]:
		return
	policy = null
	connector = ScholaLocalConnector.new()
	connector.start_timeout_sec = start_timeout_sec
	connector.step_timeout_sec = step_timeout_sec
	connector.state_changed.connect(_on_connection_state_changed)
	connector.notification_received.connect(_on_trainer_notification)
	if not connector.open(port):
		return
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
	if mode == Mode.TRAINING and connector != null and connector.state == ScholaLocalConnector.State.TRAINING:
		connector.request_stop("Stopped by the user in Godot; save the policy and close the session.")
		_set_mode(Mode.STOPPING)
		_stop_deadline = Time.get_ticks_msec() / 1000.0 + 3.0
		_log("asked the trainer to save and stop")
		return
	if connector != null and connector.is_active():
		connector.close("Run reset by the user.", false)
		_set_connection_state(ScholaLocalConnector.State.CLOSED, "Run reset by the user.")
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
	if connector != null:
		connector.poll()
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
			var request := connector.poll_request()
			while request.is_empty() and connector.is_active() and Time.get_ticks_usec() < frame_deadline:
				OS.delay_usec(50)
				connector.poll()
				request = connector.poll_request()
			if request.is_empty() or mode != Mode.TRAINING:
				return false
			_handle_request(request)
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
	var agent := environment.get_agent()
	match request.kind:
		"start":
			if request.autoreset_mode != "DISABLED":
				connector.fail("INCOMPATIBLE", "The D2 demo supports only DISABLED auto-reset; Python asked for %s." % request.autoreset_mode)
				return
			connector.reply_start()
			connector.publish_definition(_training_definition())
		"reset":
			var settings: Dictionary = request.environments.get(0, {})
			var result := environment.reset_episode(settings.get("seed"), settings.get("options", {}))
			if result.error != "":
				connector.fail("RUNTIME_FAILURE", "Reset failed: %s" % result.error)
				return
			connector.reply_reset({"0": {agent.agent_id: {
				"observation": ScholaLocalConnector.to_json_value(result.observation),
				"info": {},
			}}})
		"step":
			var actions: Dictionary = request.actions[0]
			if not actions.has(agent.agent_id):
				connector.fail("INVALID_ARGUMENT", "The step has no action for agent %s." % agent.agent_id)
				return
			var action: Variant = actions[agent.agent_id]
			# JSON numbers arrive as floats.
			if typeof(action) == TYPE_FLOAT and action == floor(action):
				action = int(action)
			var action_error: String = agent.action_space.get_validation_error(action)
			if action_error != "":
				connector.fail("INVALID_ARGUMENT", "Action %s rejected by %s: %s" % [JSON.stringify(action), agent.action_space.describe(), action_error])
				return
			if environment.is_episode_over():
				connector.fail("FAILED_PRECONDITION", "Python stepped an episode that already ended; with DISABLED auto-reset it must reset first.")
				return
			var result := _step(action, "Python trainer")
			if result.get("error", "") != "":
				connector.fail("RUNTIME_FAILURE", "Step failed: %s" % result.error)
				return
			var info: Dictionary = result.info.duplicate()
			info["outcome"] = result.outcome
			connector.reply_step([{agent.agent_id: {
				"observation": ScholaLocalConnector.to_json_value(result.observation),
				"reward": result.reward,
				"terminated": result.terminated,
				"truncated": result.truncated,
				"info": ScholaLocalConnector.info_to_strings(info),
			}}])


# Mirrors TrainingDefinition: one {agent_id: definition} per environment. The demo has one of each.
func _training_definition() -> Array:
	var agent := environment.get_agent()
	return [{agent.agent_id: {
		"observation_space": agent.observation_space.to_dictionary(),
		"action_space": agent.action_space.to_dictionary(),
		"agent_type": "",
	}}]


func _on_connection_state_changed(state: ScholaLocalConnector.State, message: String) -> void:
	_set_connection_state(state, message)
	match state:
		ScholaLocalConnector.State.WAITING:
			_set_mode(Mode.WAITING)
		ScholaLocalConnector.State.CONNECTED, ScholaLocalConnector.State.TRAINING:
			if mode != Mode.STOPPING:
				_set_mode(Mode.TRAINING)
		ScholaLocalConnector.State.CLOSED, ScholaLocalConnector.State.ERROR:
			_end_training_session(state == ScholaLocalConnector.State.ERROR)


func _on_trainer_notification(message: Dictionary) -> void:
	match message.get("type"):
		"progress":
			trainer_progress.emit(message)
		"saved":
			_log("trainer saved policy to %s" % message.get("path"))
		"log":
			_log("python: %s" % message.get("message"))


func _end_training_session(failed: bool) -> void:
	if mode not in [Mode.WAITING, Mode.TRAINING, Mode.STOPPING]:
		return
	if failed:
		_kill_trainer()
	_set_mode(Mode.MANUAL)
	environment.reset_episode()
	_log("training session ended; keyboard controls the agent")


func _set_connection_state(state: ScholaLocalConnector.State, message: String) -> void:
	connection_state = state
	connection_state_changed.emit(state, message)
	var is_error := state == ScholaLocalConnector.State.ERROR
	_log("connection %s: %s" % [ScholaLocalConnector.STATE_NAMES[state], message], is_error)


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
