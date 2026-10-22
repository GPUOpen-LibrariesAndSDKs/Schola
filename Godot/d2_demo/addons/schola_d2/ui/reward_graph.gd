class_name ScholaRewardGraph
extends Control
## Episode reward history with a rolling average.

@export var max_episodes := 400
@export var rolling_window := 20

var _rewards: PackedFloat32Array = []
var _averages: PackedFloat32Array = []
var _sources: PackedStringArray = []

const SOURCE_COLORS := {
	"Python trainer": Color(0.35, 0.6, 1.0, 0.55),
	"saved policy": Color(0.75, 0.45, 1.0, 0.7),
	"keyboard": Color(0.7, 0.7, 0.7, 0.55),
}


func add_episode(reward: float, source: String) -> void:
	_rewards.append(reward)
	_sources.append(source)
	var start := maxi(0, _rewards.size() - rolling_window)
	var total := 0.0
	for i in range(start, _rewards.size()):
		total += _rewards[i]
	_averages.append(total / (_rewards.size() - start))
	if _rewards.size() > max_episodes:
		_rewards = _rewards.slice(1)
		_averages = _averages.slice(1)
		_sources = _sources.slice(1)
	queue_redraw()


func clear() -> void:
	_rewards.clear()
	_averages.clear()
	_sources.clear()
	queue_redraw()


func last_average() -> float:
	return _averages[-1] if not _averages.is_empty() else 0.0


func _draw() -> void:
	var rect := Rect2(Vector2.ZERO, size)
	draw_rect(rect, Color(0.08, 0.09, 0.11))
	var font := get_theme_default_font()
	if _rewards.is_empty():
		draw_string(font, Vector2(8, size.y / 2), "No finished episodes yet", HORIZONTAL_ALIGNMENT_LEFT, -1, 13, Color(0.6, 0.6, 0.6))
		return

	var low := 0.0
	var high := 0.0
	for value in _rewards:
		low = minf(low, value)
		high = maxf(high, value)
	if is_equal_approx(low, high):
		high = low + 1.0
	var pad := 6.0
	var plot := Rect2(pad, pad, size.x - 2 * pad, size.y - 2 * pad)
	var to_y := func(v: float) -> float: return plot.end.y - (v - low) / (high - low) * plot.size.y
	var bar_width := plot.size.x / max_episodes

	var zero_y: float = to_y.call(0.0)
	draw_line(Vector2(plot.position.x, zero_y), Vector2(plot.end.x, zero_y), Color(0.4, 0.4, 0.4), 1.0)
	for i in _rewards.size():
		var x := plot.position.x + i * bar_width
		var y: float = to_y.call(_rewards[i])
		var color: Color = SOURCE_COLORS.get(_sources[i], Color.GRAY)
		draw_rect(Rect2(x, minf(y, zero_y), maxf(bar_width - 1.0, 1.0), absf(zero_y - y)), color)
	var points := PackedVector2Array()
	for i in _averages.size():
		points.append(Vector2(plot.position.x + (i + 0.5) * bar_width, to_y.call(_averages[i])))
	if points.size() >= 2:
		draw_polyline(points, Color(1.0, 0.8, 0.3), 2.0, true)

	draw_string(font, Vector2(plot.position.x + 2, plot.position.y + 11), "%.1f" % high, HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(0.7, 0.7, 0.7))
	draw_string(font, Vector2(plot.position.x + 2, plot.end.y - 2), "%.1f" % low, HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(0.7, 0.7, 0.7))
	draw_string(font, Vector2(plot.end.x - 150, plot.position.y + 11), "rolling avg (%d): %.2f" % [rolling_window, _averages[-1]], HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(1.0, 0.8, 0.3))
