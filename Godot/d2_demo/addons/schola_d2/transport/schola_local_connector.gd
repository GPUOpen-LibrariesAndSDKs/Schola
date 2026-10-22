class_name ScholaLocalConnector
extends RefCounted
## D2 stand-in for US3's Connector (Godot/src/core/connector/connector.h): newline-delimited JSON
## over localhost TCP. Godot listens and Python connects. As in Schola's GymService, Python starts
## every exchange and Godot answers it. This class only moves and checks messages; the caller decides
## what each request does. Message schema: Godot/docs/local_protocol.md.

## Emitted on every connection state change, with a message suitable for the UI.
signal state_changed(state: State, message: String)
## Python's one-way demo messages ("progress", "saved", "log"), which get no reply.
signal notification_received(message: Dictionary)

enum State { IDLE, WAITING, CONNECTED, TRAINING, CLOSED, ERROR }

const STATE_NAMES := {
	State.IDLE: "idle",
	State.WAITING: "waiting",
	State.CONNECTED: "connected",
	State.TRAINING: "training",
	State.CLOSED: "closed",
	State.ERROR: "error",
}
const AUTORESET_MODES := ["SAME_STEP", "NEXT_STEP", "DISABLED"]
const NOTIFICATIONS := ["progress", "saved", "log"]

var state := State.IDLE
## How long to wait for Python to connect and send start.
var start_timeout_sec := 45.0
## How long Python may go without sending anything during training. 0 disables it.
var step_timeout_sec := 60.0

var _server: TCPServer
var _peer: StreamPeerTCP
var _address := ""
var _port := 0
var _buffer := PackedByteArray()
var _requests: Array[Dictionary] = []
var _awaiting_reply := false
var _started := false
var _definition: Variant = null
var _definition_requested := false
var _stop_message := ""
var _deadline_from_msec := 0


func open(p_port: int, p_address := "127.0.0.1") -> bool:
	_address = p_address
	_port = p_port
	_server = TCPServer.new()
	var err := _server.listen(p_port, p_address)
	if err != OK:
		_server = null
		_finish(State.ERROR, "Could not listen on %s:%d (%s). Another program may be using the port; change port on the Runner node." % [p_address, p_port, error_string(err)])
		return false
	_deadline_from_msec = Time.get_ticks_msec()
	_set_state(State.WAITING, "Waiting for Python on %s:%d." % [p_address, p_port])
	return true


## True while listening or connected.
func is_active() -> bool:
	return state in [State.WAITING, State.CONNECTED, State.TRAINING]


## Accepts the connection, reads Python's messages, and enforces the timeouts. Call every frame.
func poll() -> void:
	if not is_active():
		return
	if _peer == null:
		_accept_connection()
		return
	_peer.poll()
	if _peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		if _stop_message != "":
			_finish(State.CLOSED, "Python disconnected after the stop request.")
		else:
			_finish(State.ERROR, "Python disconnected during training. Check the trainer's output for an error.")
		return
	var available := _peer.get_available_bytes()
	if available > 0:
		var received := _peer.get_data(available)
		if received[0] == OK:
			_buffer.append_array(received[1])
	var newline := _buffer.find(10)
	while newline != -1 and is_active():
		var line := _buffer.slice(0, newline).get_string_from_utf8()
		_buffer = _buffer.slice(newline + 1)
		if line.strip_edges() != "":
			_handle_line(line)
		newline = _buffer.find(10)
	_check_timeouts()


## Returns the next request from Python, or an empty Dictionary. Each request must be answered with
## its reply method, or with fail(), before the next one is returned. Requests have a "kind":
## "start" with "autoreset_mode"; "reset" with "environments" ({id: {"seed", "options"}}); or
## "step" with "actions" (one {agent_id: action} per environment).
func poll_request() -> Dictionary:
	if _awaiting_reply or _requests.is_empty():
		return {}
	_awaiting_reply = true
	return _requests.pop_front()


func reply_start() -> void:
	_started = true
	_reply({"type": "start_result"})


## Answers Python's definition request now and for any later one in this session. p_environments has
## one {agent_id: {"observation_space", "action_space", "agent_type"}} per environment.
func publish_definition(p_environments: Array) -> void:
	_definition = p_environments
	if _definition_requested:
		_send({"type": "definition_result", "environments": _definition})
		_definition_requested = false
	_set_state(State.TRAINING, "Training session started.")


## p_initial_state is {environment_id: {agent_id: {"observation", "info"}}}.
func reply_reset(p_initial_state: Dictionary) -> void:
	_reply({"type": "reset_result", "initial_state": p_initial_state})


## p_state has one {agent_id: {"observation", "reward", "terminated", "truncated", "info"}} per
## environment. p_initial_state carries environments reset in the same step, if any.
func reply_step(p_state: Array, p_initial_state := {}) -> void:
	_reply({"type": "step_result", "state": p_state, "initial_state": p_initial_state})


## Answers Python's pending request with an error, then ends the session.
func fail(p_code: String, p_message: String) -> void:
	_send_error(p_code, p_message)
	_finish(State.ERROR, p_message)


## Asks Python to stop: its current and later requests are answered with a CLOSED error carrying
## p_message, so it can save and close the session itself.
func request_stop(p_message: String) -> void:
	_stop_message = p_message
	if _awaiting_reply:
		_send_error("CLOSED", p_message)
		_awaiting_reply = false
	for request in _requests:
		_send_error("CLOSED", p_message)
	_requests.clear()


## Ends the session without waiting for Python. p_notify false skips state_changed, for shutdown.
func close(p_message := "Connection closed by Godot.", p_notify := true) -> void:
	if not is_active():
		return
	if p_notify:
		_finish(State.CLOSED, p_message)
	else:
		_disconnect()
		state = State.CLOSED


## Converts packed arrays to plain Arrays so JSON can encode them.
static func to_json_value(p_value: Variant) -> Variant:
	if typeof(p_value) >= TYPE_PACKED_BYTE_ARRAY and typeof(p_value) <= TYPE_PACKED_VECTOR4_ARRAY:
		return Array(p_value)
	return p_value


## The protocol's info maps hold strings, matching AgentState.info in Proto/State.proto.
static func info_to_strings(p_info: Dictionary) -> Dictionary:
	var strings := {}
	for key in p_info:
		strings[str(key)] = str(p_info[key])
	return strings


func _accept_connection() -> void:
	if _server != null and _server.is_connection_available():
		_peer = _server.take_connection()
		_peer.set_no_delay(true)
		_server.stop()
		_server = null
		_deadline_from_msec = Time.get_ticks_msec()
		_set_state(State.CONNECTED, "Python connected; waiting for start.")
	elif _timed_out(start_timeout_sec):
		_finish(State.ERROR, "Python did not connect to %s:%d within %s s. Check that the trainer is running and uses port %d." % [_address, _port, start_timeout_sec, _port])


func _check_timeouts() -> void:
	if not is_active() or _awaiting_reply or not _requests.is_empty():
		return
	if not _started and _timed_out(start_timeout_sec):
		fail("TIMEOUT", "Python connected but did not send start within %s s." % start_timeout_sec)
	elif _started and step_timeout_sec > 0 and _timed_out(step_timeout_sec):
		fail("TIMEOUT", "No message from Python for %s s; the training process may have stopped." % step_timeout_sec)


func _timed_out(p_seconds: float) -> bool:
	return Time.get_ticks_msec() - _deadline_from_msec > p_seconds * 1000.0


func _handle_line(p_line: String) -> void:
	_deadline_from_msec = Time.get_ticks_msec()
	var json := JSON.new()
	var message: Variant = json.data if json.parse(p_line) == OK else null
	if typeof(message) != TYPE_DICTIONARY:
		fail("INVALID_ARGUMENT", "Python sent a message that is not a JSON object: %s" % p_line.left(120))
		return
	var type: Variant = message.get("type")
	if type in NOTIFICATIONS:
		notification_received.emit(message)
		return
	if type not in ["start", "definition", "update"]:
		fail("INVALID_ARGUMENT", "Python sent an unknown message type %s; expected start, definition or update." % JSON.stringify(type))
		return
	if _awaiting_reply or not _requests.is_empty():
		fail("FAILED_PRECONDITION", "Python sent %s before Godot answered its previous request; send one request at a time." % type)
		return
	if _stop_message != "" and not _is_close(message):
		_send_error("CLOSED", _stop_message)
		return
	match type:
		"start":
			_handle_start(message)
		"definition":
			_handle_definition()
		"update":
			_handle_update(message)


func _handle_start(p_message: Dictionary) -> void:
	if _started:
		fail("FAILED_PRECONDITION", "Python sent start twice; start opens the session once.")
		return
	var mode: Variant = p_message.get("autoreset_mode")
	if mode not in AUTORESET_MODES:
		fail("INVALID_ARGUMENT", "start.autoreset_mode must be one of %s; got %s." % [", ".join(AUTORESET_MODES), JSON.stringify(mode)])
		return
	_requests.append({"kind": "start", "autoreset_mode": mode})


func _handle_definition() -> void:
	if not _started:
		fail("FAILED_PRECONDITION", "Python asked for the definition before start; send start first.")
	elif _definition != null:
		_send({"type": "definition_result", "environments": _definition})
	else:
		_definition_requested = true


func _handle_update(p_message: Dictionary) -> void:
	if _is_close(p_message):
		var status: String = p_message["status"]
		_send({"type": "close_result"})
		if status == "ERROR":
			_finish(State.ERROR, "Python reported an error: %s" % p_message.get("message", "no details"))
		else:
			_finish(State.CLOSED, "Python closed the training session.")
		return
	if _definition == null:
		fail("FAILED_PRECONDITION", "Python sent an update before the session started; send start and definition first.")
		return
	if p_message.has("reset"):
		_decode_reset(p_message["reset"])
	elif p_message.has("step"):
		_decode_step(p_message["step"])
	elif p_message.has("status"):
		fail("INVALID_ARGUMENT", "update.status must be CLOSED or ERROR; got %s." % JSON.stringify(p_message["status"]))
	else:
		fail("INVALID_ARGUMENT", "An update must contain reset, step or status.")


func _decode_reset(p_reset: Variant) -> void:
	if typeof(p_reset) != TYPE_DICTIONARY or typeof(p_reset.get("environments", {})) != TYPE_DICTIONARY:
		fail("INVALID_ARGUMENT", "update.reset.environments must be an object keyed by environment ID.")
		return
	var environments := {}
	var given: Dictionary = p_reset.get("environments", {})
	for key in given:
		var id := _environment_id(key)
		if id == -1:
			return
		var settings: Variant = given[key]
		if typeof(settings) != TYPE_DICTIONARY:
			fail("INVALID_ARGUMENT", "Reset settings for environment %s must be an object." % key)
			return
		var seed: Variant = settings.get("seed")
		if seed != null and typeof(seed) != TYPE_FLOAT and typeof(seed) != TYPE_INT:
			fail("INVALID_ARGUMENT", "The seed for environment %s must be a number or null." % key)
			return
		environments[id] = {"seed": seed, "options": settings.get("options", {})}
	_requests.append({"kind": "reset", "environments": environments})


func _decode_step(p_step: Variant) -> void:
	var actions: Variant = null
	if typeof(p_step) == TYPE_DICTIONARY:
		actions = p_step.get("actions")
	if typeof(actions) != TYPE_ARRAY or actions.size() != _definition.size():
		fail("INVALID_ARGUMENT", "update.step.actions must be a list with one entry per environment (%d)." % _definition.size())
		return
	for index in actions.size():
		var environment_actions: Variant = actions[index]
		if typeof(environment_actions) != TYPE_DICTIONARY:
			fail("INVALID_ARGUMENT", "Actions for environment %d must be an object keyed by agent ID." % index)
			return
		for agent_id in environment_actions:
			if not _definition[index].has(agent_id):
				fail("INVALID_ARGUMENT", "Environment %d has no agent %s; its agents are %s." % [index, JSON.stringify(agent_id), _definition[index].keys()])
				return
	_requests.append({"kind": "step", "actions": actions})


func _environment_id(p_key: Variant) -> int:
	var text := str(p_key)
	if not text.is_valid_int() or int(text) < 0 or int(text) >= _definition.size():
		fail("INVALID_ARGUMENT", "Unknown environment ID %s; valid IDs are 0 to %d." % [JSON.stringify(p_key), _definition.size() - 1])
		return -1
	return int(text)


func _is_close(p_message: Dictionary) -> bool:
	return p_message.get("type") == "update" and p_message.get("status") in ["CLOSED", "ERROR"]


func _reply(p_message: Dictionary) -> void:
	_awaiting_reply = false
	_send(p_message)


func _send_error(p_code: String, p_message: String) -> void:
	_awaiting_reply = false
	_send({"type": "error", "code": p_code, "message": p_message})


func _send(p_message: Dictionary) -> void:
	if _peer != null and _peer.get_status() == StreamPeerTCP.STATUS_CONNECTED:
		_peer.put_data((JSON.stringify(p_message) + "\n").to_utf8_buffer())


func _finish(p_state: State, p_message: String) -> void:
	_disconnect()
	_set_state(p_state, p_message)


func _disconnect() -> void:
	if _peer != null:
		_peer.disconnect_from_host()
		_peer = null
	if _server != null:
		_server.stop()
		_server = null
	_buffer.clear()
	_requests.clear()
	_awaiting_reply = false


func _set_state(p_state: State, p_message: String) -> void:
	state = p_state
	state_changed.emit(p_state, p_message)
