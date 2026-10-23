@tool
@icon("res://addons/schola_d2/icons/schola_agent.svg")
class_name ScholaDemoAgent
extends ScholaAgent
## Demo lifecycle helper layered on the native US5 ScholaAgent configuration node.


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
