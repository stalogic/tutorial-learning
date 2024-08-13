class_name Entity
extends Sprite2D

var grid_position: Vector2i:
	set(value):
		grid_position = value
		position = Grid.grid_to_world(grid_position)

func _init(start_position: Vector2i, entity_definition: EntityDefinition) -> void:
	self.centered = false
	self.grid_position = start_position
	self.texture = entity_definition.texture
	self.modulate = entity_definition.color
	
func move(move_offset: Vector2i) -> void:
	self.grid_position += move_offset
	
