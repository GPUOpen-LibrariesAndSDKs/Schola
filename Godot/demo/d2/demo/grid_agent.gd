@tool
class_name GridAgent
extends ScholaDemoAgent
## Observation: Box [x, y, goal dx, goal dy]. Action: Discrete(4) up, down, left, right.

const MOVES: Array[Vector2i] = [Vector2i.UP, Vector2i.DOWN, Vector2i.LEFT, Vector2i.RIGHT]

@onready var player := get_parent() as GridPlayer


func _observe() -> Variant:
	var goal := player.get_arena().get_goal_cell()
	return PackedFloat32Array([player.cell.x, player.cell.y, goal.x - player.cell.x, goal.y - player.cell.y])


func _act(action: Variant) -> void:
	player.move(MOVES[action])


func _get_configuration_warnings() -> PackedStringArray:
	var warnings := get_configuration_errors()
	if not get_parent() is GridPlayer:
		warnings.append("GridAgent must be a child of a GridPlayer.")
	return warnings
