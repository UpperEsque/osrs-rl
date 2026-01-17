#!/usr/bin/env python3
"""
Download OSRS game data from Wiki API and other sources
"""
import json
import os
import requests
import time

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def download_items():
    """Download all items from OSRS Wiki"""
    print("Downloading items...")
    
    # OSRS Wiki API for item data
    url = "https://prices.runescape.wiki/api/v1/osrs/mapping"
    
    headers = {
        'User-Agent': 'OSRS-RL-Bot/1.0 (Training purposes)'
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch items: {response.status_code}")
        return
    
    items_raw = response.json()
    
    # Process into a cleaner format
    items = {}
    for item in items_raw:
        item_id = item.get('id')
        if item_id:
            items[item_id] = {
                'id': item_id,
                'name': item.get('name', ''),
                'examine': item.get('examine', ''),
                'members': item.get('members', False),
                'lowalch': item.get('lowalch', 0),
                'highalch': item.get('highalch', 0),
                'limit': item.get('limit', 0),
                'value': item.get('value', 0),
            }
    
    # Save
    filepath = os.path.join(DATA_DIR, 'items.json')
    with open(filepath, 'w') as f:
        json.dump(items, f, indent=2)
    
    print(f"Saved {len(items)} items to {filepath}")
    return items


def download_npcs():
    """Download NPC data from OSRS Wiki"""
    print("Downloading NPCs...")
    
    # Use the wiki's SMW API to get NPC data
    url = "https://oldschool.runescape.wiki/api.php"
    
    all_npcs = {}
    offset = 0
    batch_size = 500
    
    while True:
        params = {
            'action': 'ask',
            'query': f'[[Category:Monsters]]|?Combat level|?Hitpoints|?Max hit|?Attack style|?Slayer category|?Slayer experience|limit={batch_size}|offset={offset}',
            'format': 'json'
        }
        
        headers = {'User-Agent': 'OSRS-RL-Bot/1.0'}
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            if response.status_code != 200:
                print(f"Failed at offset {offset}: {response.status_code}")
                break
                
            data = response.json()
            results = data.get('query', {}).get('results', {})
            
            if not results:
                break
            
            for name, info in results.items():
                printouts = info.get('printouts', {})
                npc_data = {
                    'name': name.replace('_', ' '),
                    'combat_level': printouts.get('Combat level', [0])[0] if printouts.get('Combat level') else 0,
                    'hitpoints': printouts.get('Hitpoints', [0])[0] if printouts.get('Hitpoints') else 0,
                    'max_hit': printouts.get('Max hit', [0])[0] if printouts.get('Max hit') else 0,
                    'attack_style': printouts.get('Attack style', ['Unknown'])[0] if printouts.get('Attack style') else 'Unknown',
                    'slayer_category': printouts.get('Slayer category', [''])[0] if printouts.get('Slayer category') else '',
                    'slayer_xp': printouts.get('Slayer experience', [0])[0] if printouts.get('Slayer experience') else 0,
                }
                all_npcs[name] = npc_data
            
            print(f"  Fetched {len(results)} NPCs (total: {len(all_npcs)})")
            
            if len(results) < batch_size:
                break
                
            offset += batch_size
            time.sleep(1)  # Be nice to the API
            
        except Exception as e:
            print(f"Error at offset {offset}: {e}")
            break
    
    # Save
    filepath = os.path.join(DATA_DIR, 'npcs.json')
    with open(filepath, 'w') as f:
        json.dump(all_npcs, f, indent=2)
    
    print(f"Saved {len(all_npcs)} NPCs to {filepath}")
    return all_npcs


def download_objects():
    """Download game object data (trees, rocks, etc.)"""
    print("Downloading objects...")
    
    # Common skilling objects with their IDs
    # These are hardcoded as there's no good API for this
    objects = {
        # Trees
        "Tree": {"id": 1276, "type": "tree", "skill": "woodcutting", "level": 1, "xp": 25},
        "Oak tree": {"id": 10820, "type": "tree", "skill": "woodcutting", "level": 15, "xp": 37.5},
        "Willow tree": {"id": 10819, "type": "tree", "skill": "woodcutting", "level": 30, "xp": 67.5},
        "Maple tree": {"id": 10832, "type": "tree", "skill": "woodcutting", "level": 45, "xp": 100},
        "Yew tree": {"id": 10822, "type": "tree", "skill": "woodcutting", "level": 60, "xp": 175},
        "Magic tree": {"id": 10834, "type": "tree", "skill": "woodcutting", "level": 75, "xp": 250},
        "Redwood tree": {"id": 29668, "type": "tree", "skill": "woodcutting", "level": 90, "xp": 380},
        
        # Rocks
        "Copper rock": {"id": 10943, "type": "rock", "skill": "mining", "level": 1, "xp": 17.5},
        "Tin rock": {"id": 10941, "type": "rock", "skill": "mining", "level": 1, "xp": 17.5},
        "Iron rock": {"id": 10945, "type": "rock", "skill": "mining", "level": 15, "xp": 35},
        "Coal rock": {"id": 10947, "type": "rock", "skill": "mining", "level": 30, "xp": 50},
        "Gold rock": {"id": 10949, "type": "rock", "skill": "mining", "level": 40, "xp": 65},
        "Mithril rock": {"id": 10951, "type": "rock", "skill": "mining", "level": 55, "xp": 80},
        "Adamantite rock": {"id": 10953, "type": "rock", "skill": "mining", "level": 70, "xp": 95},
        "Runite rock": {"id": 10955, "type": "rock", "skill": "mining", "level": 85, "xp": 125},
        
        # Fishing spots (these are NPCs technically)
        "Fishing spot (net)": {"id": 1530, "type": "fishing", "skill": "fishing", "level": 1, "xp": 10},
        "Fishing spot (fly)": {"id": 1526, "type": "fishing", "skill": "fishing", "level": 20, "xp": 50},
        "Fishing spot (cage)": {"id": 1519, "type": "fishing", "skill": "fishing", "level": 40, "xp": 90},
        "Fishing spot (harpoon)": {"id": 1520, "type": "fishing", "skill": "fishing", "level": 35, "xp": 80},
        
        # Banks
        "Bank booth": {"id": 10583, "type": "bank", "skill": None, "level": 0, "xp": 0},
        "Bank chest": {"id": 12308, "type": "bank", "skill": None, "level": 0, "xp": 0},
        
        # Furnace/Anvil
        "Furnace": {"id": 16469, "type": "furnace", "skill": "smithing", "level": 1, "xp": 0},
        "Anvil": {"id": 2097, "type": "anvil", "skill": "smithing", "level": 1, "xp": 0},
        
        # Cooking
        "Range": {"id": 9682, "type": "range", "skill": "cooking", "level": 1, "xp": 0},
        "Fire": {"id": 26185, "type": "fire", "skill": "cooking", "level": 1, "xp": 0},
    }
    
    filepath = os.path.join(DATA_DIR, 'objects.json')
    with open(filepath, 'w') as f:
        json.dump(objects, f, indent=2)
    
    print(f"Saved {len(objects)} objects to {filepath}")
    return objects


def download_locations():
    """Download key location data"""
    print("Creating locations database...")
    
    locations = {
        # Cities
        "Lumbridge": {
            "center": [3222, 3218],
            "type": "city",
            "features": ["bank", "furnace", "range", "trees", "goblins"],
            "teleport": "Home teleport"
        },
        "Varrock": {
            "center": [3213, 3428],
            "type": "city", 
            "features": ["bank", "anvil", "grand_exchange", "trees"],
            "teleport": "Varrock teleport"
        },
        "Falador": {
            "center": [2964, 3378],
            "type": "city",
            "features": ["bank", "mining_guild_nearby", "furnace"],
            "teleport": "Falador teleport"
        },
        "Edgeville": {
            "center": [3093, 3493],
            "type": "city",
            "features": ["bank", "furnace", "wilderness_nearby", "yew_trees"],
            "teleport": "Amulet of glory"
        },
        "Ferox Enclave": {
            "center": [3142, 3633],
            "type": "safe_zone",
            "features": ["bank", "pool_of_refreshment", "pvp_supplies"],
            "teleport": "Ring of dueling"
        },
        
        # Skilling locations
        "Draynor Village Willows": {
            "center": [3087, 3235],
            "type": "woodcutting",
            "features": ["willow_trees", "bank_nearby"],
            "teleport": "Amulet of glory (Draynor)"
        },
        "Seers Village Maples": {
            "center": [2722, 3499],
            "type": "woodcutting",
            "features": ["maple_trees", "bank_very_close"],
            "teleport": "Camelot teleport"
        },
        "Woodcutting Guild": {
            "center": [1658, 3505],
            "type": "woodcutting",
            "features": ["all_trees", "bank", "invisible_boost"],
            "teleport": "Skills necklace"
        },
        "Mining Guild": {
            "center": [3046, 9756],
            "type": "mining",
            "features": ["coal", "mithril", "adamant", "invisible_boost"],
            "teleport": "Skills necklace"
        },
        "Motherlode Mine": {
            "center": [3760, 5670],
            "type": "mining",
            "features": ["pay_dirt", "prospector_outfit"],
            "teleport": "Skills necklace"
        },
        
        # Combat locations
        "Cows (Lumbridge)": {
            "center": [3253, 3270],
            "type": "combat",
            "features": ["cows", "cowhides", "low_level"],
            "monsters": ["Cow"]
        },
        "Chickens (Lumbridge)": {
            "center": [3235, 3295],
            "type": "combat",
            "features": ["chickens", "feathers", "very_low_level"],
            "monsters": ["Chicken"]
        },
        "Rock Crabs": {
            "center": [2707, 3714],
            "type": "combat",
            "features": ["rock_crabs", "afk", "low_defence"],
            "monsters": ["Rock crab"]
        },
        "Sand Crabs": {
            "center": [1726, 3463],
            "type": "combat",
            "features": ["sand_crabs", "afk", "low_defence"],
            "monsters": ["Sand crab"]
        },
        "Ammonite Crabs": {
            "center": [3706, 3880],
            "type": "combat",
            "features": ["ammonite_crabs", "fossil_island", "best_afk"],
            "monsters": ["Ammonite crab"]
        },
        
        # Slayer locations
        "Slayer Tower": {
            "center": [3429, 3538],
            "type": "slayer",
            "features": ["crawling_hands", "banshees", "infernal_mages", "bloodvelds", "gargoyles", "nechryaels", "abyssal_demons"],
            "teleport": "Slayer ring"
        },
        "Catacombs of Kourend": {
            "center": [1666, 10049],
            "type": "slayer",
            "features": ["many_monsters", "ancient_shards", "totem_pieces"],
            "teleport": "Kourend portal"
        },
    }
    
    filepath = os.path.join(DATA_DIR, 'locations.json')
    with open(filepath, 'w') as f:
        json.dump(locations, f, indent=2)
    
    print(f"Saved {len(locations)} locations to {filepath}")
    return locations


def download_skills():
    """Create skills database"""
    print("Creating skills database...")
    
    skills = {
        0: {"name": "Attack", "type": "combat", "trainable": True},
        1: {"name": "Defence", "type": "combat", "trainable": True},
        2: {"name": "Strength", "type": "combat", "trainable": True},
        3: {"name": "Hitpoints", "type": "combat", "trainable": True},
        4: {"name": "Ranged", "type": "combat", "trainable": True},
        5: {"name": "Prayer", "type": "combat", "trainable": True},
        6: {"name": "Magic", "type": "combat", "trainable": True},
        7: {"name": "Cooking", "type": "gathering", "trainable": True},
        8: {"name": "Woodcutting", "type": "gathering", "trainable": True},
        9: {"name": "Fletching", "type": "artisan", "trainable": True},
        10: {"name": "Fishing", "type": "gathering", "trainable": True},
        11: {"name": "Firemaking", "type": "artisan", "trainable": True},
        12: {"name": "Crafting", "type": "artisan", "trainable": True},
        13: {"name": "Smithing", "type": "artisan", "trainable": True},
        14: {"name": "Mining", "type": "gathering", "trainable": True},
        15: {"name": "Herblore", "type": "artisan", "trainable": True},
        16: {"name": "Agility", "type": "support", "trainable": True},
        17: {"name": "Thieving", "type": "support", "trainable": True},
        18: {"name": "Slayer", "type": "combat", "trainable": True},
        19: {"name": "Farming", "type": "gathering", "trainable": True},
        20: {"name": "Runecraft", "type": "artisan", "trainable": True},
        21: {"name": "Hunter", "type": "gathering", "trainable": True},
        22: {"name": "Construction", "type": "artisan", "trainable": True},
    }
    
    # XP table
    xp_table = [0]
    for level in range(1, 127):
        xp = int(sum(int(l + 300 * 2 ** (l / 7)) for l in range(1, level)) / 4)
        xp_table.append(xp)
    
    data = {
        "skills": skills,
        "xp_table": xp_table
    }
    
    filepath = os.path.join(DATA_DIR, 'skills.json')
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved skills data to {filepath}")
    return data


def download_prayers():
    """Create prayers database"""
    print("Creating prayers database...")
    
    prayers = {
        0: {"name": "Thick Skin", "level": 1, "drain": 3, "effect": "+5% Defence"},
        1: {"name": "Burst of Strength", "level": 4, "drain": 3, "effect": "+5% Strength"},
        2: {"name": "Clarity of Thought", "level": 7, "drain": 3, "effect": "+5% Attack"},
        3: {"name": "Sharp Eye", "level": 8, "drain": 3, "effect": "+5% Ranged"},
        4: {"name": "Mystic Will", "level": 9, "drain": 3, "effect": "+5% Magic"},
        5: {"name": "Rock Skin", "level": 10, "drain": 6, "effect": "+10% Defence"},
        6: {"name": "Superhuman Strength", "level": 13, "drain": 6, "effect": "+10% Strength"},
        7: {"name": "Improved Reflexes", "level": 16, "drain": 6, "effect": "+10% Attack"},
        8: {"name": "Rapid Restore", "level": 19, "drain": 1, "effect": "2x stat restore"},
        9: {"name": "Rapid Heal", "level": 22, "drain": 2, "effect": "2x HP restore"},
        10: {"name": "Protect Item", "level": 25, "drain": 2, "effect": "Keep 1 extra item"},
        11: {"name": "Hawk Eye", "level": 26, "drain": 6, "effect": "+10% Ranged"},
        12: {"name": "Mystic Lore", "level": 27, "drain": 6, "effect": "+10% Magic"},
        13: {"name": "Steel Skin", "level": 28, "drain": 12, "effect": "+15% Defence"},
        14: {"name": "Ultimate Strength", "level": 31, "drain": 12, "effect": "+15% Strength"},
        15: {"name": "Incredible Reflexes", "level": 34, "drain": 12, "effect": "+15% Attack"},
        16: {"name": "Protect from Magic", "level": 37, "drain": 12, "effect": "Block magic"},
        17: {"name": "Protect from Missiles", "level": 40, "drain": 12, "effect": "Block ranged"},
        18: {"name": "Protect from Melee", "level": 43, "drain": 12, "effect": "Block melee"},
        19: {"name": "Eagle Eye", "level": 44, "drain": 12, "effect": "+15% Ranged"},
        20: {"name": "Mystic Might", "level": 45, "drain": 12, "effect": "+15% Magic"},
        21: {"name": "Retribution", "level": 46, "drain": 3, "effect": "Damage on death"},
        22: {"name": "Redemption", "level": 49, "drain": 6, "effect": "Heal on low HP"},
        23: {"name": "Smite", "level": 52, "drain": 18, "effect": "Drain enemy prayer"},
        24: {"name": "Preserve", "level": 55, "drain": 2, "effect": "50% slower stat drain"},
        25: {"name": "Chivalry", "level": 60, "drain": 24, "effect": "+15% Att, +18% Str, +20% Def"},
        26: {"name": "Piety", "level": 70, "drain": 24, "effect": "+20% Att, +23% Str, +25% Def"},
        27: {"name": "Rigour", "level": 74, "drain": 24, "effect": "+20% Ranged, +23% damage"},
        28: {"name": "Augury", "level": 77, "drain": 24, "effect": "+25% Magic, +25% Def"},
    }
    
    filepath = os.path.join(DATA_DIR, 'prayers.json')
    with open(filepath, 'w') as f:
        json.dump(prayers, f, indent=2)
    
    print(f"Saved {len(prayers)} prayers to {filepath}")
    return prayers


def download_combat_styles():
    """Create combat styles database"""
    print("Creating combat styles database...")
    
    combat = {
        "attack_styles": {
            "accurate": {"attack_type": "melee", "xp": "attack", "bonus": "+3 attack"},
            "aggressive": {"attack_type": "melee", "xp": "strength", "bonus": "+3 strength"},
            "defensive": {"attack_type": "melee", "xp": "defence", "bonus": "+3 defence"},
            "controlled": {"attack_type": "melee", "xp": "shared", "bonus": "+1 all"},
            "rapid": {"attack_type": "ranged", "xp": "ranged", "bonus": "faster speed"},
            "accurate_ranged": {"attack_type": "ranged", "xp": "ranged", "bonus": "+3 ranged"},
            "longrange": {"attack_type": "ranged", "xp": "ranged+defence", "bonus": "+2 attack range"},
        },
        "combat_triangle": {
            "melee": {"strong_against": "ranged", "weak_against": "magic"},
            "ranged": {"strong_against": "magic", "weak_against": "melee"},
            "magic": {"strong_against": "melee", "weak_against": "ranged"},
        },
        "special_attacks": {
            "Dragon dagger": {"spec_cost": 25, "effect": "2 rapid hits with increased accuracy"},
            "Granite maul": {"spec_cost": 50, "effect": "Instant attack"},
            "Armadyl godsword": {"spec_cost": 50, "effect": "+25% accuracy, +37.5% damage"},
            "Dragon claws": {"spec_cost": 50, "effect": "4 hits with accuracy roll"},
            "Volatile nightmare staff": {"spec_cost": 55, "effect": "Large magic hit"},
        }
    }
    
    filepath = os.path.join(DATA_DIR, 'combat.json')
    with open(filepath, 'w') as f:
        json.dump(combat, f, indent=2)
    
    print(f"Saved combat data to {filepath}")
    return combat


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("=" * 60)
    print("OSRS Game Data Downloader")
    print("=" * 60)
    print()
    
    download_items()
    print()
    
    download_npcs()
    print()
    
    download_objects()
    print()
    
    download_locations()
    print()
    
    download_skills()
    print()
    
    download_prayers()
    print()
    
    download_combat_styles()
    print()
    
    print("=" * 60)
    print("Download complete!")
    print(f"Data saved to: {DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
