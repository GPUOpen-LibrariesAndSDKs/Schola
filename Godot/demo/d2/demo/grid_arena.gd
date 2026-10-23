@tool
class_name GridArena
extends Node3D
## The grid level. Cell (x, y) is the floor square at world (x, z).

const FLOOR := "floor"
const GOAL := "goal"
const HAZARD := "hazard"

const OUTCOME_STYLE := {
	ScholaDemoEnvironment.SUCCESS: ["GOAL REACHED", Color(0.35, 1.0, 0.5)],
	ScholaDemoEnvironment.FAILURE: ["HIT HAZARD", Color(1.0, 0.35, 0.35)],
	ScholaDemoEnvironment.TIMEOUT: ["TIMEOUT", Color(1.0, 0.7, 0.25)],
}

@export var grid_size := Vector2i(6, 6):
	set(value):
		grid_size = value
		_rebuild_floor()
@export var cell_size := 1.0:
	set(value):
		cell_size = value
		_rebuild_floor()

@onready var _banner: Label3D = $Banner
@onready var _outcome_light: OmniLight3D = $OutcomeLight

var _banner_tween: Tween
var _light_tween: Tween


func _ready() -> void:
	if Engine.is_editor_hint():
		return
	_set_banner_alpha(0.0)
	_outcome_light.light_energy = 0.0


func get_goal_cell() -> Vector2i:
	for marker in _markers(GridMarker.Kind.GOAL):
		return cell_of(marker.position)
	return Vector2i(-1, -1)


func get_hazard_cells() -> Array[Vector2i]:
	var cells: Array[Vector2i] = []
	for marker in _markers(GridMarker.Kind.HAZARD):
		cells.append(cell_of(marker.position))
	return cells


func get_cell_kind(cell: Vector2i) -> String:
	if cell == get_goal_cell():
		return GOAL
	if get_hazard_cells().has(cell):
		return HAZARD
	return FLOOR


func random_free_cell(rng: RandomNumberGenerator) -> Vector2i:
	var free: Array[Vector2i] = []
	for y in grid_size.y:
		for x in grid_size.x:
			if get_cell_kind(Vector2i(x, y)) == FLOOR:
				free.append(Vector2i(x, y))
	return free[rng.randi_range(0, free.size() - 1)]


func clamp_cell(cell: Vector2i) -> Vector2i:
	return cell.clamp(Vector2i.ZERO, grid_size - Vector2i.ONE)


func cell_of(local_position: Vector3) -> Vector2i:
	return Vector2i(floori(local_position.x / cell_size), floori(local_position.z / cell_size))


func cell_center(cell: Vector2i) -> Vector3:
	return Vector3((cell.x + 0.5) * cell_size, 0.0, (cell.y + 0.5) * cell_size)


func show_outcome(outcome: String, total_reward: float) -> void:
	var style: Array = OUTCOME_STYLE.get(outcome, ["DONE", Color.WHITE])
	if _banner_tween != null:
		_banner_tween.kill()
	_banner.text = "%s\n%+.1f" % [style[0], total_reward]
	_banner.modulate = style[1]
	_set_banner_alpha(1.0)

	if _light_tween != null:
		_light_tween.kill()
	_outcome_light.light_color = style[1]
	for child in get_children():
		if child is GridPlayer:
			_outcome_light.position = child.position + Vector3.UP * 1.5
	_outcome_light.light_energy = 6.0
	_light_tween = create_tween()
	_light_tween.tween_property(_outcome_light, "light_energy", 0.0, 0.6)


func hide_outcome(_episode: int) -> void:
	if _banner_tween != null:
		_banner_tween.kill()
	_banner_tween = create_tween()
	_banner_tween.tween_method(_set_banner_alpha, _banner.modulate.a, 0.0, 0.25)


func _markers(kind: GridMarker.Kind) -> Array[GridMarker]:
	var markers: Array[GridMarker] = []
	for child in get_children():
		if child is GridMarker and child.kind == kind:
			markers.append(child)
	return markers


func _rebuild_floor() -> void:
	if is_node_ready() and has_node("Floor"):
		$Floor.rebuild()


func _set_banner_alpha(alpha: float) -> void:
	_banner.modulate.a = alpha
	_banner.outline_modulate.a = alpha


func _get_configuration_warnings() -> PackedStringArray:
	var warnings := PackedStringArray()
	var goals := _markers(GridMarker.Kind.GOAL)
	if goals.size() != 1:
		warnings.append("Add exactly one GridMarker with kind Goal (found %d)." % goals.size())
	for child in get_children():
		if child is GridMarker:
			var cell := cell_of(child.position)
			if cell != clamp_cell(cell):
				warnings.append("%s is outside the %dx%d grid." % [child.name, grid_size.x, grid_size.y])
	return warnings
