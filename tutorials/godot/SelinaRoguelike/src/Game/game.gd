class_name Game
extends Node2D

const player_definition: EntityDefinition = preload("res://src/Assets/Definnitions/Entities/Actors/entity_definition_player.tres")

@onready var player: Entity
@onready var event_handler: EventHandler = $EventHandler
@onready var entities: Node2D = $Entities
@onready var map: Map = $Map


# Called when the node enters the scene tree for the first time.
func _ready():
	player = Entity.new(Vector2i.ZERO, player_definition)
	var camera: Camera2D = $Camera2D
	remove_child(camera)
	player.add_child(camera)
	entities.add_child(player)
	map.generate(player)

func _physics_process(_delta: float) -> void:
	var action: Action = event_handler.get_action()
	if action:
		action.perform(self, player)

func get_map_data() -> MapData:
	return map.map_data
