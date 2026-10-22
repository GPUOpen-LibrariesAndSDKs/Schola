@tool
class_name GridEnvironment
extends ScholaDemoEnvironment
## Rewards, start cells and end conditions for the grid.

@export_group("Rewards")
@export var goal_reward := 10.0
@export var failure_penalty := -10.0
@export var step_penalty := -0.1
@export_group("Start")
@export var random_start := true
@export var start_cell := Vector2i(0, 5)

@onready var arena := $Arena as GridArena


func _reset_episode(_seed: Variant, _options: Dictionary) -> void:
	for agent in get_agents():
		var cell := arena.random_free_cell(get_rng()) if random_start else start_cell
		(agent as GridAgent).player.place(cell)


func _get_reward(agent: ScholaDemoAgent) -> float:
	var kind := _cell_kind(agent)
	if kind == GridArena.GOAL:
		return step_penalty + goal_reward
	if kind == GridArena.HAZARD:
		return step_penalty + failure_penalty
	return step_penalty


func _is_terminated(agent: ScholaDemoAgent) -> bool:
	return _cell_kind(agent) != GridArena.FLOOR


func _get_info(agent: ScholaDemoAgent) -> Dictionary:
	var kind := _cell_kind(agent)
	if kind == GridArena.GOAL:
		return {"outcome": SUCCESS}
	if kind == GridArena.HAZARD:
		return {"outcome": FAILURE}
	return {"outcome": RUNNING}


func get_config() -> Dictionary:
	var config := super()
	config.merge({
		"goal_reward": goal_reward,
		"failure_penalty": failure_penalty,
		"step_penalty": step_penalty,
	})
	return config


func _cell_kind(agent: ScholaDemoAgent) -> String:
	return arena.get_cell_kind((agent as GridAgent).player.cell)
