class_name Action
extends RefCounted

var entity: Entity

func _init(entity_: Entity) -> void:
	self.entity = entity_

func perform() -> void:
	pass
	
func get_map_data() -> MapData:
	return entity.map_data
