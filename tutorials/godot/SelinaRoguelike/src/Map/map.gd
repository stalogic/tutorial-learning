class_name Map
extends Node2D

@export var fov_radius: int = 8

var map_data: MapData

@onready var dungeon_generator = $DungeonGenerator
@onready var tiles = $Tiles
@onready var entities = $Entities
@onready var field_of_view = $FieldOfView

func update_fov(player_position: Vector2i) -> void:
	field_of_view.update_fov(map_data, player_position, fov_radius)
	
	for entity in map_data.entities:
		entity.visible = map_data.get_tile(entity.grid_position).is_in_view

func generate(player: Entity):
	map_data = dungeon_generator.generate_dungeon(player)
	_place_tiles()
	_place_entities()

func _place_tiles() -> void:
	for tile in map_data.tiles:
		tiles.add_child(tile)
		
func _place_entities() -> void:
	for entity in map_data.entities:
		entities.add_child(entity)

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(_delta):
	pass
