class_name Map
extends Node2D

@export var fov_radius: int = 8

var map_data: MapData

@onready var dungeon_generator = $DungeonGenerator
@onready var field_of_view = %FieldOfView

func update_fov(player_position: Vector2i) -> void:
	field_of_view.update_fov(map_data, player_position, fov_radius)

func generate(player: Entity):
	map_data = dungeon_generator.generate_dungeon(player)
	_replace_tiles()

func _replace_tiles() -> void:
	for tile in map_data.tiles:
		add_child(tile)

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(_delta):
	pass
