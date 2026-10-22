@tool
extends EditorPlugin
## Adds the Schola setup dock.

const SetupDock := preload("res://addons/schola_d2/editor/setup_dock.gd")

var _dock: Control


func _enter_tree() -> void:
	_dock = SetupDock.new()
	_dock.name = "Schola"
	add_control_to_dock(DOCK_SLOT_RIGHT_UL, _dock)


func _exit_tree() -> void:
	if _dock != null:
		remove_control_from_docks(_dock)
		_dock.queue_free()
		_dock = null
