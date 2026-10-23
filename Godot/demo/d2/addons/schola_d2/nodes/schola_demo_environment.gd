@tool
@icon("res://addons/schola_d2/icons/schola_environment.svg")
class_name ScholaDemoEnvironment
extends ScholaEnvironment
## Demo lifecycle bridge layered on the native US5 ScholaEnvironment node.

signal episode_started(episode: int)
signal step_completed(result: Dictionary)
signal episode_ended(outcome: String, total_reward: float)

const RUNNING := "running"
const SUCCESS := "success"
const FAILURE := "failure"
const TIMEOUT := "timeout"

var episode := 0
var episode_step := 0
var cumulative_reward := 0.0
var outcome := RUNNING
var last_observation: Variant = null

var _rng := RandomNumberGenerator.new()


func _reset_episode(_seed: Variant, _options: Dictionary) -> void:
	pass


func _get_reward(_agent: ScholaAgent) -> float:
	push_error("%s must override _get_reward()" % name)
	return 0.0


func _is_terminated(_agent: ScholaAgent) -> bool:
	return false


func _is_truncated(_agent: ScholaAgent) -> bool:
	return false


func _get_info(_agent: ScholaAgent) -> Dictionary:
	return {}


func get_episode_step() -> int:
	return episode_step


func get_rng() -> RandomNumberGenerator:
	return _rng


func get_agent() -> ScholaDemoAgent:
	var agents := super.get_agents()
	if agents.is_empty():
		return null
	return agents[0] as ScholaDemoAgent


func is_episode_over() -> bool:
	return outcome != RUNNING


func get_config() -> Dictionary:
	return {"max_episode_steps": max_episode_steps}


func reset_episode(seed: Variant = null, options := {}) -> Dictionary:
	if seed != null:
		_rng.seed = int(seed)
	episode += 1
	episode_step = 0
	cumulative_reward = 0.0
	outcome = RUNNING
	_reset_episode(seed, options)
	var observed := get_agent().observe_checked()
	last_observation = observed.value
	episode_started.emit(episode)
	return {"observation": observed.value, "error": observed.error}


func step(action: Variant) -> Dictionary:
	var agent := get_agent()
	var action_error: String = agent.action_space.get_validation_error(action)
	if action_error != "":
		return {"error": "action rejected by %s: %s" % [agent.action_space.describe(), action_error]}
	if is_episode_over():
		return {"error": "step called after the episode ended; reset first"}

	agent._act(action)
	episode_step += 1
	var observed := agent.observe_checked()
	var reward := _get_reward(agent)
	var terminated := _is_terminated(agent)
	var step_limit_hit := max_episode_steps > 0 and episode_step >= max_episode_steps
	var truncated := not terminated and (_is_truncated(agent) or step_limit_hit)
	var info := _get_info(agent)
	if terminated:
		outcome = str(info.get("outcome", "terminated"))
	elif truncated:
		outcome = TIMEOUT
	cumulative_reward += reward
	last_observation = observed.value

	var result := {
		"observation": observed.value,
		"reward": reward,
		"terminated": terminated,
		"truncated": truncated,
		"outcome": outcome,
		"info": info,
		"action": action,
		"error": observed.error,
	}
	step_completed.emit(result)
	if terminated or truncated:
		episode_ended.emit(outcome, cumulative_reward)
	return result
