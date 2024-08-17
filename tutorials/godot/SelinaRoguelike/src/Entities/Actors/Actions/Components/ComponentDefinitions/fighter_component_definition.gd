class_name FighterComponentDefinition
extends Resource

@export_category("Stats")
@export var max_hp: int
@export var power: int
@export var defense: int

@export_category("Visuals")
@export var death_texture: AtlasTexture = preload("res://src/Assets/default_death_texture.tres")
@export var death_color: Color = Color.DARK_RED
