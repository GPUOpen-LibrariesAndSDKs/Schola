@tool
@icon("res://addons/schola_d2/icons/schola_agent.svg")
class_name ScholaDemoAgent
extends Node
## Stand-in for the US5 ScholaAgent node.

@export var agent_id := "agent_0"
@export var observation_space: ScholaSpace:
	set(value):
		observation_space = value
		update_configuration_warnings()
@export var action_space: ScholaSpace:
	set(value):
		action_space = value
		update_configuration_warnings()


func _observe() -> Variant:
	return null


func _act(_action: Variant) -> void:
	pass


func observe_checked() -> Dictionary:
	var value: Variant = _observe()
	var error := "observation_space is not set"
	if observation_space != null:
		error = observation_space.get_validation_error(value)
	return {"value": value, "error": error}


func get_configuration_errors() -> PackedStringArray:
	var errors := PackedStringArray()
	if observation_space == null:
		errors.append("observation_space is not set")
	elif observation_space.get_definition_error() != "":
		errors.append("observation_space: " + observation_space.get_definition_error())
	if action_space == null:
		errors.append("action_space is not set")
	elif action_space.get_definition_error() != "":
		errors.append("action_space: " + action_space.get_definition_error())
	return errors


func _get_configuration_warnings() -> PackedStringArray:
	return get_configuration_errors()
