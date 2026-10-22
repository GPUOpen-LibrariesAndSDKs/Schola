class_name ScholaHud
extends CanvasLayer
## In-game Schola panel.

@export var runner: ScholaDemoRunner
@export var panel_width := 470
@export var history_height := 220

const MODE_TEXT := {
	ScholaDemoRunner.Mode.MANUAL: ["MANUAL", "Keyboard controls agent (arrow keys)", Color(0.45, 0.45, 0.5)],
	ScholaDemoRunner.Mode.WAITING: ["WAITING", "Waiting for the Python training client...", Color(0.75, 0.55, 0.15)],
	ScholaDemoRunner.Mode.TRAINING: ["TRAINING", "Training client controls agent", Color(0.2, 0.45, 0.9)],
	ScholaDemoRunner.Mode.STOPPING: ["STOPPING", "Training client is saving its policy...", Color(0.75, 0.55, 0.15)],
	ScholaDemoRunner.Mode.POLICY: ["INFERENCE", "Saved policy controls agent", Color(0.55, 0.3, 0.85)],
}
const CONNECTION_TEXT := {
	ScholaDemoRunner.Mode.MANUAL: "disconnected",
	ScholaDemoRunner.Mode.WAITING: "waiting for trainer",
	ScholaDemoRunner.Mode.TRAINING: "training",
	ScholaDemoRunner.Mode.STOPPING: "stopping",
	ScholaDemoRunner.Mode.POLICY: "policy loaded",
}
const OUTCOME_COLORS := {
	"running": Color(0.8, 0.8, 0.8),
	"success": Color(0.3, 0.85, 0.4),
	"failure": Color(0.95, 0.3, 0.3),
	"timeout": Color(0.95, 0.65, 0.2),
}
const ACTION_NAMES := ["up", "down", "left", "right"]
const MAX_LOG_LINES := 60

var _mode_panel: PanelContainer
var _mode_title: Label
var _mode_detail: Label
var _connection: Label
var _start_button: Button
var _stop_button: Button
var _load_button: Button
var _load_onnx_button: Button
var _speed_label: Label
var _telemetry := {}
var _graph: ScholaRewardGraph
var _log: RichTextLabel
var _log_lines: PackedStringArray = []
var _last_source := ""


func _ready() -> void:
	_build()
	var environment := runner.environment
	runner.mode_changed.connect(_on_mode_changed)
	runner.log_message.connect(_on_log)
	runner.action_chosen.connect(_on_action_chosen)
	runner.trainer_progress.connect(_on_trainer_progress)
	environment.episode_started.connect(func(_episode): _refresh_telemetry())
	environment.step_completed.connect(func(_result): _refresh_telemetry())
	environment.episode_ended.connect(_on_episode_ended)
	_on_mode_changed(runner.mode)
	_refresh_telemetry()


func _build() -> void:
	var root := PanelContainer.new()
	root.anchor_left = 1.0
	root.anchor_right = 1.0
	root.anchor_bottom = 1.0
	root.offset_left = -panel_width
	add_child(root)

	var scroll := ScrollContainer.new()
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	root.add_child(scroll)
	var margin := MarginContainer.new()
	margin.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	for side in ["left", "right", "top", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, 10)
	scroll.add_child(margin)
	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 6)
	margin.add_child(box)

	var title := Label.new()
	title.text = "Schola D2 · Grid World"
	title.add_theme_font_size_override("font_size", 20)
	box.add_child(title)

	_mode_panel = PanelContainer.new()
	var mode_box := VBoxContainer.new()
	_mode_title = Label.new()
	_mode_title.add_theme_font_size_override("font_size", 18)
	_mode_detail = Label.new()
	mode_box.add_child(_mode_title)
	mode_box.add_child(_mode_detail)
	_mode_panel.add_child(mode_box)
	box.add_child(_mode_panel)

	box.add_child(_section("Training controls"))
	var buttons := HBoxContainer.new()
	_start_button = _button("Start Training", runner.start_training)
	_stop_button = _button("Stop / Reset Run", runner.stop_run)
	_load_button = _button("Load Saved Policy", runner.load_saved_policy)
	_load_onnx_button = _button("Load ONNX Policy", runner.load_onnx_policy)
	for b in [_start_button, _stop_button]:
		buttons.add_child(b)
	box.add_child(buttons)
	var policy_buttons := HBoxContainer.new()
	for b in [_load_button, _load_onnx_button]:
		policy_buttons.add_child(b)
	box.add_child(policy_buttons)
	var row := HBoxContainer.new()
	_connection = Label.new()
	_connection.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	row.add_child(_connection)
	row.add_child(_button("Test US2 validation", runner.run_validation_check))
	box.add_child(row)
	var speed_row := HBoxContainer.new()
	_speed_label = Label.new()
	_speed_label.custom_minimum_size.x = 130
	var slider := HSlider.new()
	slider.min_value = 0.0
	slider.max_value = 3.0
	slider.step = 0.01
	slider.value = log(runner.steps_per_second) / log(10.0)
	slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	slider.value_changed.connect(_on_speed_changed)
	speed_row.add_child(_speed_label)
	speed_row.add_child(slider)
	box.add_child(speed_row)
	_on_speed_changed(slider.value)

	box.add_child(_section("Environment setup (Inspector)"))
	var agent := runner.environment.get_agent()
	var environment := runner.environment
	var setup := _grid(box)
	_pair(setup, "Schola nodes", "%s (%s)  ›  %s (%s)" % [
		environment.name, _script_class(environment), environment.get_path_to(agent), _script_class(agent),
	])
	_pair(setup, "Observation space", agent.observation_space.describe())
	_pair(setup, "Action space", "%s: up, down, left, right" % agent.action_space.describe())
	var config := environment.get_config()
	var settings := PackedStringArray()
	for key in config:
		var value: Variant = config[key]
		settings.append("%s %s" % [String(key).replace("_", " "), "%+.1f" % value if typeof(value) == TYPE_FLOAT else str(value)])
	_pair(setup, "Settings", "  ·  ".join(settings))
	for error in agent.get_configuration_errors():
		_pair(setup, "Config error", error).add_theme_color_override("font_color", OUTCOME_COLORS.failure)

	box.add_child(_section("Live RL telemetry"))
	var telemetry := _grid(box)
	for key in ["Episode", "Step", "Cumulative reward", "Last observation", "Action", "Terminal state", "Trainer"]:
		_telemetry[key] = _pair(telemetry, key, "-")

	_build_history_panel()


func _build_history_panel() -> void:
	var panel := PanelContainer.new()
	panel.anchor_top = 1.0
	panel.anchor_right = 1.0
	panel.anchor_bottom = 1.0
	panel.offset_top = -history_height
	panel.offset_right = -panel_width
	add_child(panel)
	var margin := MarginContainer.new()
	for side in ["left", "right", "top", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, 8)
	panel.add_child(margin)
	var columns := HBoxContainer.new()
	columns.add_theme_constant_override("separation", 10)
	margin.add_child(columns)

	var history := VBoxContainer.new()
	history.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	history.add_child(_section("Episode reward history"))
	_graph = ScholaRewardGraph.new()
	_graph.size_flags_vertical = Control.SIZE_EXPAND_FILL
	history.add_child(_graph)
	var legend := Label.new()
	legend.text = "blue: training · purple: saved policy · grey: keyboard · gold: rolling avg"
	legend.add_theme_font_size_override("font_size", 11)
	legend.add_theme_color_override("font_color", Color(0.65, 0.65, 0.65))
	history.add_child(legend)
	columns.add_child(history)

	var log_box := VBoxContainer.new()
	log_box.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	log_box.add_child(_section("Log"))
	_log = RichTextLabel.new()
	_log.bbcode_enabled = true
	_log.scroll_following = true
	_log.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_log.add_theme_font_size_override("normal_font_size", 12)
	log_box.add_child(_log)
	columns.add_child(log_box)


func _script_class(node: Node) -> String:
	var script: Script = node.get_script()
	if script == null or script.get_global_name() == &"":
		return node.get_class()
	return script.get_global_name()


func _section(text: String) -> Label:
	var label := Label.new()
	label.text = text.to_upper()
	label.add_theme_font_size_override("font_size", 12)
	label.add_theme_color_override("font_color", Color(0.55, 0.7, 1.0))
	return label


func _button(text: String, callback: Callable) -> Button:
	var button := Button.new()
	button.text = text
	button.focus_mode = Control.FOCUS_NONE
	button.pressed.connect(callback)
	return button


func _grid(parent: Control) -> GridContainer:
	var grid := GridContainer.new()
	grid.columns = 2
	grid.add_theme_constant_override("h_separation", 12)
	parent.add_child(grid)
	return grid


func _pair(grid: GridContainer, key: String, value: String) -> Label:
	var key_label := Label.new()
	key_label.text = key
	key_label.add_theme_color_override("font_color", Color(0.65, 0.65, 0.65))
	key_label.size_flags_vertical = Control.SIZE_SHRINK_BEGIN
	var value_label := Label.new()
	value_label.text = value
	value_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	value_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	value_label.custom_minimum_size.x = 250
	grid.add_child(key_label)
	grid.add_child(value_label)
	return value_label


func _on_mode_changed(mode: ScholaDemoRunner.Mode) -> void:
	var info: Array = MODE_TEXT[mode]
	_mode_title.text = info[0]
	_mode_detail.text = info[1]
	if mode == ScholaDemoRunner.Mode.POLICY:
		var source := runner.policy_source
		_mode_detail.text = "%s%s controls agent" % [source.left(1).to_upper(), source.substr(1)]
	var style := StyleBoxFlat.new()
	style.bg_color = info[2]
	style.set_corner_radius_all(6)
	style.set_content_margin_all(8)
	_mode_panel.add_theme_stylebox_override("panel", style)
	_connection.text = "Connection: " + CONNECTION_TEXT[mode]
	var busy := mode in [ScholaDemoRunner.Mode.WAITING, ScholaDemoRunner.Mode.TRAINING, ScholaDemoRunner.Mode.STOPPING]
	_start_button.disabled = busy
	_load_button.disabled = busy
	_load_onnx_button.disabled = busy
	if mode == ScholaDemoRunner.Mode.TRAINING:
		_telemetry["Trainer"].text = "connected"
	elif mode == ScholaDemoRunner.Mode.MANUAL:
		_telemetry["Trainer"].text = "-"


func _on_speed_changed(value: float) -> void:
	runner.steps_per_second = roundf(pow(10.0, value))
	_speed_label.text = "Speed: %d steps/s" % runner.steps_per_second


func _on_action_chosen(action: Variant, source: String) -> void:
	_last_source = source
	var label := str(action)
	if typeof(action) == TYPE_INT and action >= 0 and action < ACTION_NAMES.size():
		label = "%d (%s)" % [action, ACTION_NAMES[action]]
	_telemetry["Action"].text = "%s  from %s" % [label, source]


func _on_trainer_progress(stats: Dictionary) -> void:
	_telemetry["Trainer"].text = "episode %d / %d · epsilon %.2f · %d states" % [
		int(stats.get("episode", 0)), int(stats.get("episodes", 0)),
		float(stats.get("epsilon", 0.0)), int(stats.get("states", 0)),
	]


func _on_episode_ended(outcome: String, total_reward: float) -> void:
	_graph.add_episode(total_reward, _last_source)
	_refresh_telemetry()


func _refresh_telemetry() -> void:
	var environment := runner.environment
	_telemetry["Episode"].text = str(environment.episode)
	_telemetry["Step"].text = "%d / %d" % [environment.get_episode_step(), environment.max_episode_steps]
	_telemetry["Cumulative reward"].text = "%+.2f" % environment.cumulative_reward
	_telemetry["Last observation"].text = _format_observation(environment.last_observation)
	var state: Label = _telemetry["Terminal state"]
	state.text = environment.outcome
	state.add_theme_color_override("font_color", OUTCOME_COLORS.get(environment.outcome, Color.WHITE))


func _format_observation(value: Variant) -> String:
	if typeof(value) != TYPE_PACKED_FLOAT32_ARRAY:
		return str(value)
	var parts := PackedStringArray()
	for v in value:
		parts.append(str(snappedf(v, 0.01)))
	return "[%s]  (x, y, dx, dy)" % ", ".join(parts)


func _on_log(text: String, is_error: bool) -> void:
	var stamp := Time.get_time_string_from_system()
	var line := "[color=#888]%s[/color] %s" % [stamp, text]
	if is_error:
		line = "[color=#888]%s[/color] [color=#f66]%s[/color]" % [stamp, text]
	_log_lines.append(line)
	if _log_lines.size() > MAX_LOG_LINES:
		_log_lines = _log_lines.slice(_log_lines.size() - MAX_LOG_LINES)
	_log.text = "\n".join(_log_lines)
