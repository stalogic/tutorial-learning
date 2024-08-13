class_name Map
extends Node2D

@export var map_width: int = 80
@export var map_height: int = 45

var map_data: MapData

# Called when the node enters the scene tree for the first time.
func _ready():
	map_data = MapData.new(map_width, map_height)
	_replace_tiles()

func _replace_tiles() -> void:
	for tile in map_data.tiles:
		add_child(tile)

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass
