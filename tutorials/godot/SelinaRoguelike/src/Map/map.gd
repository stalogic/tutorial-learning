class_name Map
extends Node2D

@onready var dungeon_generator = $DungeonGenerator

var map_data: MapData

func generate(player: Entity):
	map_data = dungeon_generator.generate_dungeon(player)
	_replace_tiles()

func _replace_tiles() -> void:
	for tile in map_data.tiles:
		add_child(tile)

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(_delta):
	pass
