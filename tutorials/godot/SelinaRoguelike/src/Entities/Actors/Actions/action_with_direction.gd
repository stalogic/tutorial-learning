class_name ActionWithDirection
extends Action

var offset: Vector2i

func _init(entity_: Entity, dx: int, dy: int) -> void:
	super._init(entity_)
	offset = Vector2i(dx, dy)
	
func perform() -> bool:
	return false

func get_destination() -> Vector2i:
	return entity.grid_position + offset
	
func get_blocking_entity_at_destination() -> Entity:
	return get_map_data().get_blocking_entity_at_locaion(get_destination())

func get_target_actor() -> Entity:
	return get_map_data().get_actor_at_location(get_destination())
