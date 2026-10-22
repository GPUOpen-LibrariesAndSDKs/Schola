@tool
extends Node3D
## Builds the checkerboard floor tiles at runtime.

const TILE_HEIGHT := 0.1

var _tile_materials: Array[StandardMaterial3D] = []


func _ready() -> void:
	for color in [Color(0.22, 0.23, 0.27), Color(0.27, 0.28, 0.33)]:
		var material := StandardMaterial3D.new()
		material.albedo_color = color
		material.roughness = 0.9
		_tile_materials.append(material)
	rebuild()


func rebuild() -> void:
	var grid := get_parent() as GridArena
	if grid == null or _tile_materials.is_empty():
		return
	for child in get_children():
		child.queue_free()

	var tile := BoxMesh.new()
	tile.size = Vector3(grid.cell_size * 0.96, TILE_HEIGHT, grid.cell_size * 0.96)
	for y in grid.grid_size.y:
		for x in grid.grid_size.x:
			var cell := MeshInstance3D.new()
			cell.mesh = tile
			cell.material_override = _tile_materials[(x + y) % 2]
			cell.position = grid.cell_center(Vector2i(x, y)) + Vector3.DOWN * TILE_HEIGHT / 2
			add_child(cell)

	var base := MeshInstance3D.new()
	var base_mesh := BoxMesh.new()
	base_mesh.size = Vector3(grid.grid_size.x * grid.cell_size + 0.4, 0.3, grid.grid_size.y * grid.cell_size + 0.4)
	base.mesh = base_mesh
	var base_material := StandardMaterial3D.new()
	base_material.albedo_color = Color(0.1, 0.11, 0.13)
	base.material_override = base_material
	base.position = Vector3(grid.grid_size.x * grid.cell_size / 2, -0.25, grid.grid_size.y * grid.cell_size / 2)
	add_child(base)
