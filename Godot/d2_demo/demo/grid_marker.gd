@tool
class_name GridMarker
extends Node3D
## Marks the cell under it as the goal or a hazard.

enum Kind { GOAL, HAZARD }

@export var kind := Kind.HAZARD:
	set(value):
		kind = value
		update_configuration_warnings()


func _get_configuration_warnings() -> PackedStringArray:
	if not get_parent() is GridArena:
		return PackedStringArray(["GridMarker must be a child of a GridArena."])
	return PackedStringArray()
