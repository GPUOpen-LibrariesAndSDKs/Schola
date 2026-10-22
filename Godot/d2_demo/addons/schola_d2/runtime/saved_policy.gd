class_name ScholaSavedPolicy
extends RefCounted
## Stand-in for US7 inference: runs the trainer's saved Q-table in Godot.

const FORMAT := "schola-d2-qtable"

var episodes_trained := 0
var _q_table: Dictionary = {}


func load_policy(p_path: String, p_agent: ScholaDemoAgent) -> String:
	if not FileAccess.file_exists(p_path):
		return "no saved policy at %s; train first" % p_path
	var data: Variant = JSON.parse_string(FileAccess.get_file_as_string(p_path))
	if typeof(data) != TYPE_DICTIONARY or data.get("format") != FORMAT:
		return "%s is not a %s file" % [p_path, FORMAT]
	if _normalized(data.get("observation_space")) != _normalized(p_agent.observation_space.to_dictionary()):
		return "policy observation space does not match the agent's %s" % p_agent.observation_space.describe()
	if _normalized(data.get("action_space")) != _normalized(p_agent.action_space.to_dictionary()):
		return "policy action space does not match the agent's %s" % p_agent.action_space.describe()
	_q_table = data.get("q", {})
	episodes_trained = int(data.get("episodes", 0))
	return ""


func act(p_observation: PackedFloat32Array) -> int:
	var values: Variant = _q_table.get(state_key(p_observation))
	if values == null:
		return 0
	var best := 0
	for i in range(1, values.size()):
		if values[i] > values[best]:
			best = i
	return best


func state_count() -> int:
	return _q_table.size()


# Must match state_key() in python/trainer.py.
static func state_key(p_observation: PackedFloat32Array) -> String:
	var parts := PackedStringArray()
	for value in p_observation:
		parts.append(str(int(round(value))))
	return ",".join(parts)


static func _normalized(p_value: Variant) -> String:
	return JSON.stringify(JSON.parse_string(JSON.stringify(p_value)), "", true)
