@tool
class_name GridPlayer
extends Node3D
## The agent's body: tracks its cell and animates moves.

const HOP_HEIGHT := 0.3

var cell := Vector2i.ZERO
var _move_tween: Tween


func get_arena() -> GridArena:
	return get_parent() as GridArena


func move(p_direction: Vector2i) -> void:
	var arena := get_arena()
	cell = arena.clamp_cell(cell + p_direction)
	rotation.y = atan2(-p_direction.x, -p_direction.y)
	_hop_to(arena.cell_center(cell))


func place(p_cell: Vector2i) -> void:
	cell = p_cell
	if _move_tween != null:
		_move_tween.kill()
	position = get_arena().cell_center(cell)
	rotation.y = 0.0
	scale = Vector3.ONE * 0.3
	_move_tween = create_tween()
	_move_tween.tween_property(self, "scale", Vector3.ONE, 0.15)


func _hop_to(target: Vector3) -> void:
	if _move_tween != null:
		_move_tween.kill()
	scale = Vector3.ONE
	var duration := 0.15
	var runner := get_tree().get_first_node_in_group(ScholaDemoRunner.GROUP) as ScholaDemoRunner
	if runner != null:
		duration = minf(0.15, 0.8 / runner.steps_per_second)
	if duration < 0.02:
		position = target
		return
	var start := position
	_move_tween = create_tween()
	_move_tween.tween_method(func(t: float): position = start.lerp(target, t) + Vector3.UP * sin(t * PI) * HOP_HEIGHT, 0.0, 1.0, duration)
