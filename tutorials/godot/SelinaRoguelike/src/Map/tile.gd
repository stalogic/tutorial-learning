class_name Tile
extends Sprite2D

var _definition: TileDefinition

func _init(grid_position: Vector2i, tile_definition: TileDefinition) -> void:
	self.centered = false
	self.position = Grid.grid_to_world(grid_position)
	self.set_tile_type(tile_definition)

func set_tile_type(tile_definition: TileDefinition) -> void:
	self._definition = tile_definition
	self.texture = self._definition.texture
	self.modulate = self._definition.color_dark

func is_walkable() -> bool:
	return self._definition.is_walkable
	
func is_transparent() -> bool:
	return self._definition.is_transparent
