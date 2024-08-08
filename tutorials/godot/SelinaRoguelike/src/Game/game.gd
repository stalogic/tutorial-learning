extends Node2D

var player_grid_pos := Vector2i.ZERO
@onready var player: Sprite2D = $Player
@onready var event_handler: EventHandler = $EventHandler


# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(_delta: float) -> void:
	var action = event_handler.get_action()
	if action is MovementAction:
		player_grid_pos += action.offset
		player.position = Grid.grid_to_world(player_grid_pos)
	else:
		get_tree().quit()
