extends SceneTree

const MAIN_SCENE := preload("res://demo/main.tscn")
const ENVIRONMENT_PATH := NodePath("ArenaView/SubViewport/ScholaEnvironment")


func _init() -> void:
	var exit_code := 0
	var scene := MAIN_SCENE.instantiate()
	root.add_child(scene)
	var environment := scene.get_node(ENVIRONMENT_PATH) as ScholaEnvironment
	if environment == null:
		push_error("US5 smoke check: scene did not instantiate a ScholaEnvironment.")
		exit_code = 1
	else:
		var errors := environment.get_configuration_errors()
		if not errors.is_empty():
			push_error("US5 smoke check: invalid scene configuration: %s" % ", ".join(errors))
			exit_code = 1
		var agents := environment.get_agents()
		if agents.size() != 1:
			push_error("US5 smoke check: expected one agent, found %d." % agents.size())
			exit_code = 1
		elif agents[0].agent_id != "agent_0":
			push_error("US5 smoke check: expected agent_id agent_0, found %s." % agents[0].agent_id)
			exit_code = 1
		elif environment.max_episode_steps != 40:
			push_error("US5 smoke check: expected a 40-step limit, found %d." % environment.max_episode_steps)
			exit_code = 1
	scene.queue_free()
	quit(exit_code)
