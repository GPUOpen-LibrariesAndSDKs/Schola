@tool
extends VBoxContainer
## Editor dock showing each Schola environment, its agents and their US2 space checks.

const REFRESH_SECONDS := 1.0

var _text: RichTextLabel
var _elapsed := 0.0


func _ready() -> void:
	var refresh := Button.new()
	refresh.text = "Refresh"
	refresh.pressed.connect(_refresh)
	add_child(refresh)
	_text = RichTextLabel.new()
	_text.bbcode_enabled = true
	_text.fit_content = false
	_text.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_text.selection_enabled = true
	add_child(_text)
	_refresh()


func _process(delta: float) -> void:
	if not is_visible_in_tree():
		return
	_elapsed += delta
	if _elapsed >= REFRESH_SECONDS:
		_elapsed = 0.0
		_refresh()


func _refresh() -> void:
	var root := EditorInterface.get_edited_scene_root()
	if root == null:
		_text.text = "Open a scene to see its Schola setup."
		return
	var environments: Array[Node] = []
	_collect(root, environments)
	if environments.is_empty():
		_text.text = "No ScholaDemoEnvironment in this scene."
		return
	var lines := PackedStringArray()
	for environment in environments:
		_describe_environment(environment, lines)
	_text.text = "\n".join(lines)


func _collect(node: Node, out: Array[Node]) -> void:
	if node is ScholaDemoEnvironment:
		out.append(node)
	for child in node.get_children():
		_collect(child, out)


func _describe_environment(environment: ScholaDemoEnvironment, lines: PackedStringArray) -> void:
	lines.append("[b]%s[/b]  (environment node)" % environment.name)
	var config := environment.get_config()
	for key in config:
		lines.append("  %s: %s" % [String(key).capitalize(), str(config[key])])
	var agents := environment.get_agents()
	if agents.is_empty():
		lines.append("  [color=#f66]no ScholaDemoAgent node below this environment[/color]")
	for agent in agents:
		lines.append("")
		lines.append("[b]%s[/b]  (agent node, id \"%s\")" % [environment.get_path_to(agent), agent.agent_id])
		_describe_space(lines, "observation", agent.observation_space)
		_describe_space(lines, "action", agent.action_space)
	lines.append("")


func _describe_space(lines: PackedStringArray, label: String, space: ScholaSpace) -> void:
	if space == null:
		lines.append("  [color=#f66]✗ %s space not set[/color]" % label)
		return
	var error := space.get_definition_error()
	if error != "":
		lines.append("  [color=#f66]✗ %s: %s[/color]" % [label, error])
		return
	lines.append("  [color=#6d6]✓[/color] %s: %s" % [label, space.describe()])
	lines.append("     flattened size %d, default value %s" % [space.get_flattened_size(), str(space.make_default_value())])
	lines.append("     [color=#999]to Python: %s[/color]" % JSON.stringify(space.to_dictionary()))
