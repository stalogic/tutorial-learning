extends BaseInputHandler

const long_press_threshold: int = 20
const long_press_frequency: int = 5
var press_count: int = 0
var long_press: bool = false


const directions = {
	"up": Vector2i.UP,
	"down": Vector2i.DOWN,
	"left": Vector2i.LEFT,
	"right": Vector2i.RIGHT,
	"up_left": Vector2i.UP + Vector2i.LEFT,
	"up_right": Vector2i.UP + Vector2i.RIGHT,
	"down_left": Vector2i.DOWN + Vector2i.LEFT,
	"down_right": Vector2i.DOWN + Vector2i.RIGHT,
}

const inventory_menu_scene := preload("res://src/GUI/inventory_menu.tscn")

func get_item(window_title:String, inventory:InventoryComponent) -> Entity:
	var inventory_menu: InventoryMenu = inventory_menu_scene.instantiate()
	add_child(inventory_menu)
	inventory_menu.build(window_title, inventory)
	get_parent().transition_to(InputHandler.InputHandlers.DUMMY)
	var selected_item: Entity = await inventory_menu.item_selected
	await get_tree().physics_frame
	get_parent().call_deferred("transition_to", InputHandler.InputHandlers.MAIN_GAME)
	return selected_item

func _just_released() -> bool:
	for key in directions:
		if Input.is_action_just_released(key):
			return true
	return false
	
func _pressed() -> bool:
	for key in directions:
		if Input.is_action_pressed(key):
			return true
	return false

func get_action(player: Entity) -> Action:
	var action: Action = null
	
	if _just_released():
			press_count = 0
			long_press = false
	
	if _pressed():
		press_count += 1
		if not long_press:
			if press_count >= long_press_threshold:
				long_press = true
			
	if long_press and (press_count % long_press_frequency == 0):
		for direction in directions:
			if Input.is_action_pressed(direction):
				var offset = directions[direction]
				action = BumpAction.new(player, offset.x, offset.y)
	else:
		for direction in directions:
			if Input.is_action_just_pressed(direction):
				var offset = directions[direction]
				action = BumpAction.new(player, offset.x, offset.y)
				
	if Input.is_action_just_pressed("wait"):
		action = WaitAction.new(player)
		
	if Input.is_action_just_pressed("pickup"):
		action = PickupAction.new(player)
		
	if Input.is_action_just_pressed("drop"):
		var selected_item: Entity = await get_item("Select an item to drop", player.inventory_component)
		action = DropItemAction.new(player, selected_item)
		
	if Input.is_action_just_pressed("activate"):
		var selected_item: Entity = await get_item("Select an item to use", player.inventory_component)
		action = ItemAction.new(player, selected_item)
		
	if Input.is_action_just_pressed("view_history"):
		get_parent().transition_to(InputHandler.InputHandlers.HISTORY_VIEWER)
		
	if Input.is_action_just_pressed("quit") or Input.is_action_just_pressed("ui_back"):
		action = EscapeAction.new(player)
	
	return action
