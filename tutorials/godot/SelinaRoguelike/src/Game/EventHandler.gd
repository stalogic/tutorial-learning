class_name EventHandler
extends Node

@export_category("Move Config")
@export var long_press_threshold: int = 30
@export var long_press_frequency: int = 5

var press_count: int = 0
var long_press: bool = false


func get_action() -> Action:
	var action: Action = null
	
	if (Input.is_action_just_released("up")
		or Input.is_action_just_released("down")
		or Input.is_action_just_released("left")
		or Input.is_action_just_released("right")):
			press_count = 0
			long_press = false
	
	if (Input.is_action_pressed("up")
		or Input.is_action_pressed("down") 
		or Input.is_action_pressed("left")
		or Input.is_action_pressed("right")):
			
		press_count += 1
		if not long_press:
			if press_count >= long_press_threshold:
				long_press = true	
			
	if long_press and (press_count % long_press_frequency == 0):
		if Input.is_action_pressed("up"):
			action = MovementAction.new(0, -1)
		elif Input.is_action_pressed("down"):
			action = MovementAction.new(0, 1)
		elif Input.is_action_pressed("left"):
			action = MovementAction.new(-1, 0)
		elif Input.is_action_pressed("right"):
			action = MovementAction.new(1, 0)
	else:
		if Input.is_action_just_pressed("up"):
			action = MovementAction.new(0, -1)
		elif Input.is_action_just_pressed("down"):
			action = MovementAction.new(0, 1)
		elif Input.is_action_just_pressed("left"):
			action = MovementAction.new(-1, 0)
		elif Input.is_action_just_pressed("right"):
			action = MovementAction.new(1, 0)
		
	if Input.is_action_just_pressed("ui_cancel"):
		action = EscapeAction.new()
	
	return action
