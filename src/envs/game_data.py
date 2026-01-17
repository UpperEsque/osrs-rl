"""
OSRS Game Data - Items, NPCs, Locations, etc.
"""
import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


@dataclass
class Item:
    id: int
    name: str
    examine: str = ""
    members: bool = False
    lowalch: int = 0
    highalch: int = 0
    value: int = 0


@dataclass
class NPC:
    name: str
    combat_level: int = 0
    hitpoints: int = 0
    max_hit: int = 0
    attack_style: str = "Unknown"
    slayer_category: str = ""
    slayer_xp: float = 0


@dataclass
class Location:
    name: str
    center: List[int]
    type: str
    features: List[str]
    teleport: str = ""


class GameData:
    """
    Central access point for all OSRS game data.
    
    Usage:
        data = GameData()
        item = data.get_item(4151)  # Abyssal whip
        npc = data.get_npc("Abyssal demon")
        loc = data.get_location("Slayer Tower")
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance
    
    def __init__(self):
        if not self._loaded:
            self._load_data()
            self._loaded = True
    
    def _load_data(self):
        """Load all data files"""
        self.items: Dict[int, Item] = {}
        self.items_by_name: Dict[str, Item] = {}
        self.npcs: Dict[str, NPC] = {}
        self.locations: Dict[str, Location] = {}
        self.objects: Dict[str, Dict] = {}
        self.skills: Dict[int, Dict] = {}
        self.xp_table: List[int] = []
        self.prayers: Dict[int, Dict] = {}
        self.combat: Dict[str, Any] = {}
        
        # Load items
        items_path = os.path.join(DATA_DIR, 'items.json')
        if os.path.exists(items_path):
            with open(items_path) as f:
                raw = json.load(f)
                for item_id, data in raw.items():
                    item = Item(
                        id=int(item_id),
                        name=data.get('name', ''),
                        examine=data.get('examine', ''),
                        members=data.get('members', False),
                        lowalch=data.get('lowalch', 0),
                        highalch=data.get('highalch', 0),
                        value=data.get('value', 0),
                    )
                    self.items[int(item_id)] = item
                    self.items_by_name[item.name.lower()] = item
            print(f"[GameData] Loaded {len(self.items)} items")
        
        # Load NPCs
        npcs_path = os.path.join(DATA_DIR, 'npcs.json')
        if os.path.exists(npcs_path):
            with open(npcs_path) as f:
                raw = json.load(f)
                for name, data in raw.items():
                    npc = NPC(
                        name=name,
                        combat_level=data.get('combat_level', 0),
                        hitpoints=data.get('hitpoints', 0),
                        max_hit=data.get('max_hit', 0),
                        attack_style=data.get('attack_style', 'Unknown'),
                        slayer_category=data.get('slayer_category', ''),
                        slayer_xp=data.get('slayer_xp', 0),
                    )
                    self.npcs[name.lower()] = npc
            print(f"[GameData] Loaded {len(self.npcs)} NPCs")
        
        # Load locations
        locations_path = os.path.join(DATA_DIR, 'locations.json')
        if os.path.exists(locations_path):
            with open(locations_path) as f:
                raw = json.load(f)
                for name, data in raw.items():
                    loc = Location(
                        name=name,
                        center=data.get('center', [0, 0]),
                        type=data.get('type', ''),
                        features=data.get('features', []),
                        teleport=data.get('teleport', ''),
                    )
                    self.locations[name.lower()] = loc
            print(f"[GameData] Loaded {len(self.locations)} locations")
        
        # Load objects
        objects_path = os.path.join(DATA_DIR, 'objects.json')
        if os.path.exists(objects_path):
            with open(objects_path) as f:
                self.objects = json.load(f)
            print(f"[GameData] Loaded {len(self.objects)} objects")
        
        # Load skills
        skills_path = os.path.join(DATA_DIR, 'skills.json')
        if os.path.exists(skills_path):
            with open(skills_path) as f:
                data = json.load(f)
                self.skills = {int(k): v for k, v in data.get('skills', {}).items()}
                self.xp_table = data.get('xp_table', [])
            print(f"[GameData] Loaded {len(self.skills)} skills")
        
        # Load prayers
        prayers_path = os.path.join(DATA_DIR, 'prayers.json')
        if os.path.exists(prayers_path):
            with open(prayers_path) as f:
                raw = json.load(f)
                self.prayers = {int(k): v for k, v in raw.items()}
            print(f"[GameData] Loaded {len(self.prayers)} prayers")
        
        # Load combat
        combat_path = os.path.join(DATA_DIR, 'combat.json')
        if os.path.exists(combat_path):
            with open(combat_path) as f:
                self.combat = json.load(f)
            print(f"[GameData] Loaded combat data")
    
    # Item methods
    def get_item(self, item_id: int) -> Optional[Item]:
        """Get item by ID"""
        return self.items.get(item_id)
    
    def get_item_by_name(self, name: str) -> Optional[Item]:
        """Get item by name (case-insensitive)"""
        return self.items_by_name.get(name.lower())
    
    def search_items(self, query: str) -> List[Item]:
        """Search items by partial name"""
        query = query.lower()
        return [item for item in self.items.values() if query in item.name.lower()]
    
    # NPC methods
    def get_npc(self, name: str) -> Optional[NPC]:
        """Get NPC by name (case-insensitive)"""
        return self.npcs.get(name.lower())
    
    def search_npcs(self, query: str) -> List[NPC]:
        """Search NPCs by partial name"""
        query = query.lower()
        return [npc for npc in self.npcs.values() if query in npc.name.lower()]
    
    def get_npcs_by_slayer_category(self, category: str) -> List[NPC]:
        """Get all NPCs in a slayer category"""
        return [npc for npc in self.npcs.values() if npc.slayer_category.lower() == category.lower()]
    
    # Location methods
    def get_location(self, name: str) -> Optional[Location]:
        """Get location by name"""
        return self.locations.get(name.lower())
    
    def get_locations_by_type(self, loc_type: str) -> List[Location]:
        """Get all locations of a type (city, woodcutting, mining, etc.)"""
        return [loc for loc in self.locations.values() if loc.type == loc_type]
    
    def get_nearest_location(self, x: int, y: int, loc_type: Optional[str] = None) -> Optional[Location]:
        """Find nearest location to coordinates"""
        best = None
        best_dist = float('inf')
        
        for loc in self.locations.values():
            if loc_type and loc.type != loc_type:
                continue
            
            dx = loc.center[0] - x
            dy = loc.center[1] - y
            dist = (dx*dx + dy*dy) ** 0.5
            
            if dist < best_dist:
                best_dist = dist
                best = loc
        
        return best
    
    # Skill methods
    def get_skill_name(self, skill_id: int) -> str:
        """Get skill name by ID"""
        skill = self.skills.get(skill_id)
        return skill['name'] if skill else f"Unknown({skill_id})"
    
    def get_level_for_xp(self, xp: int) -> int:
        """Get level for given XP amount"""
        for level, required in enumerate(self.xp_table):
            if xp < required:
                return level - 1
        return 99
    
    def get_xp_for_level(self, level: int) -> int:
        """Get XP required for level"""
        if 0 <= level < len(self.xp_table):
            return self.xp_table[level]
        return 0
    
    # Prayer methods
    def get_prayer(self, prayer_id: int) -> Optional[Dict]:
        """Get prayer by ID"""
        return self.prayers.get(prayer_id)
    
    def get_protection_prayer(self, attack_style: str) -> Optional[int]:
        """Get the protection prayer ID for an attack style"""
        style_map = {
            'magic': 16,   # Protect from Magic
            'ranged': 17,  # Protect from Missiles
            'melee': 18,   # Protect from Melee
        }
        return style_map.get(attack_style.lower())
    
    # Object methods
    def get_object(self, name: str) -> Optional[Dict]:
        """Get object by name"""
        return self.objects.get(name)
    
    def get_objects_by_skill(self, skill: str) -> List[Dict]:
        """Get all objects for a skill"""
        return [
            {**obj, 'name': name}
            for name, obj in self.objects.items()
            if obj.get('skill') == skill
        ]


# Singleton instance
_game_data: Optional[GameData] = None

def get_game_data() -> GameData:
    """Get the singleton GameData instance"""
    global _game_data
    if _game_data is None:
        _game_data = GameData()
    return _game_data
