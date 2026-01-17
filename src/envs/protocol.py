"""
Message protocol for OSRS RL communication
"""
from dataclasses import dataclass, field
from typing import List, Optional
import json

# Action types (must match Java ActionExecutor)
class Actions:
    NOOP = 0
    WALK_TO = 1
    CLICK_INVENTORY = 2
    ATTACK_NPC = 3
    INTERACT_OBJECT = 4
    TOGGLE_PRAYER = 5
    TOGGLE_RUN = 6
    SPECIAL_ATTACK = 7


@dataclass
class EntityInfo:
    id: int = 0
    name: str = ""
    x: int = 0
    y: int = 0
    hp: int = 0
    max_hp: int = 0
    animation: int = -1
    distance: int = 0
    
    @classmethod
    def from_dict(cls, data: dict) -> 'EntityInfo':
        return cls(
            id=data.get('id', 0),
            name=data.get('name', ''),
            x=data.get('x', 0),
            y=data.get('y', 0),
            hp=data.get('hp', 0),
            max_hp=data.get('maxHp', 0),  # Java sends camelCase
            animation=data.get('animation', -1),
            distance=data.get('distance', 0)
        )


@dataclass
class ObjectInfo:
    id: int = 0
    name: str = ""
    x: int = 0
    y: int = 0
    distance: int = 0
    actions: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ObjectInfo':
        return cls(
            id=data.get('id', 0),
            name=data.get('name', ''),
            x=data.get('x', 0),
            y=data.get('y', 0),
            distance=data.get('distance', 0),
            actions=data.get('actions', []) or []
        )


@dataclass
class GameState:
    """Complete game state from plugin"""
    tick: int = 0
    
    # Player
    player_x: int = 0
    player_y: int = 0
    player_plane: int = 0
    player_hp: int = 0
    player_max_hp: int = 0
    player_prayer: int = 0
    player_max_prayer: int = 0
    player_energy: int = 0
    player_animation: int = -1
    player_is_moving: bool = False
    player_in_combat: bool = False
    
    # Target
    has_target: bool = False
    target_hp: int = 0
    target_max_hp: int = 0
    target_x: int = 0
    target_y: int = 0
    target_animation: int = -1
    
    # Inventory (28 slots)
    inventory_ids: List[int] = field(default_factory=lambda: [-1] * 28)
    inventory_quantities: List[int] = field(default_factory=lambda: [0] * 28)
    
    # Equipment (11 slots)
    equipment_ids: List[int] = field(default_factory=lambda: [-1] * 11)
    
    # Prayers (29)
    active_prayers: List[bool] = field(default_factory=lambda: [False] * 29)
    
    # Skills (23)
    skill_levels: List[int] = field(default_factory=lambda: [1] * 23)
    skill_xp: List[int] = field(default_factory=lambda: [0] * 23)
    
    # Nearby entities
    nearby_npcs: List[EntityInfo] = field(default_factory=list)
    nearby_players: List[EntityInfo] = field(default_factory=list)
    nearby_objects: List[ObjectInfo] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'GameState':
        """Parse from JSON dict (Java sends camelCase)"""
        state = cls()
        
        state.tick = data.get('tick', 0)
        
        # Player (camelCase from Java)
        state.player_x = data.get('playerX', 0)
        state.player_y = data.get('playerY', 0)
        state.player_plane = data.get('playerPlane', 0)
        state.player_hp = data.get('playerHp', 0)
        state.player_max_hp = data.get('playerMaxHp', 0)
        state.player_prayer = data.get('playerPrayer', 0)
        state.player_max_prayer = data.get('playerMaxPrayer', 0)
        state.player_energy = data.get('playerEnergy', 0)
        state.player_animation = data.get('playerAnimation', -1)
        state.player_is_moving = data.get('playerIsMoving', False)
        state.player_in_combat = data.get('playerInCombat', False)
        
        # Target
        state.has_target = data.get('hasTarget', False)
        state.target_hp = data.get('targetHp', 0)
        state.target_max_hp = data.get('targetMaxHp', 0)
        state.target_x = data.get('targetX', 0)
        state.target_y = data.get('targetY', 0)
        state.target_animation = data.get('targetAnimation', -1)
        
        # Arrays
        state.inventory_ids = data.get('inventoryIds', [-1] * 28) or [-1] * 28
        state.inventory_quantities = data.get('inventoryQuantities', [0] * 28) or [0] * 28
        state.equipment_ids = data.get('equipmentIds', [-1] * 11) or [-1] * 11
        state.active_prayers = data.get('activePrayers', [False] * 29) or [False] * 29
        state.skill_levels = data.get('skillLevels', [1] * 23) or [1] * 23
        state.skill_xp = data.get('skillXp', [0] * 23) or [0] * 23
        
        # Entities - use from_dict to handle camelCase
        state.nearby_npcs = [
            EntityInfo.from_dict(npc) for npc in (data.get('nearbyNpcs') or [])
        ]
        state.nearby_players = [
            EntityInfo.from_dict(p) for p in (data.get('nearbyPlayers') or [])
        ]
        state.nearby_objects = [
            ObjectInfo.from_dict(o) for o in (data.get('nearbyObjects') or [])
        ]
        
        return state
