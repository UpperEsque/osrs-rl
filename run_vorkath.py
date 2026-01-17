#!/usr/bin/env python3
"""Run trained Vorkath model on real game"""
import socket
import json
import sys
import glob
import time
import numpy as np

sys.path.insert(0, '/home/azdelic/osrs-rl/python')

# Find the best Vorkath model
model_paths = glob.glob("/home/azdelic/osrs-rl/models/vorkath*/best_model.zip")
if not model_paths:
    print("No Vorkath model found!")
    sys.exit(1)

model_path = sorted(model_paths)[-1]
print(f"Loading model: {model_path}")

from stable_baselines3 import PPO
model = PPO.load(model_path)

# Get the expected observation size
obs_size = model.observation_space.shape[0]
print(f"Model expects observation size: {obs_size}")
print("Model loaded!")

# Connect to plugin
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(5)
sock.connect(('localhost', 5555))
print("Connected to RuneLite plugin!")

# Action names for display
ACTION_NAMES = [
    "WAIT", "ATTACK_VORKATH", "MOVE_AWAY", "MOVE_TO_VORKATH",
    "PRAYER_MAGE", "PRAYER_RANGE", "PRAYER_OFF",
    "EAT_FOOD", "DRINK_PRAYER", "DRINK_ANTIFIRE", "DRINK_ANTIVENOM",
    "CAST_CRUMBLE", "SPECIAL_ATTACK", "EQUIP_SPEC", "EQUIP_MAIN"
]

def get_state():
    """Get state from plugin"""
    buffer = ""
    while True:
        data = sock.recv(8192).decode('utf-8')
        if not data:
            return None
        buffer += data
        try:
            msg = json.loads(buffer)
            return msg.get('data', msg)
        except:
            continue

def send_action(action_id):
    """Send action to plugin"""
    action_msg = json.dumps({"type": "action", "action": action_id}) + "\n"
    sock.send(action_msg.encode())

def state_to_obs(state, obs_size):
    """Convert game state to model observation"""
    obs = np.zeros(obs_size, dtype=np.float32)
    
    # === PLAYER STATS (indices 0-19) ===
    obs[0] = state.get('playerHp', 99) / 99.0
    obs[1] = state.get('playerMaxHp', 99) / 99.0
    obs[2] = state.get('playerPrayer', 99) / 99.0
    obs[3] = state.get('playerMaxPrayer', 99) / 99.0
    obs[4] = state.get('playerEnergy', 100) / 100.0
    obs[5] = 1.0 if state.get('playerAnimation', -1) != -1 else 0.0
    obs[6] = 1.0 if state.get('playerIsMoving', False) else 0.0
    obs[7] = 1.0 if state.get('playerInCombat', False) else 0.0
    
    # Position (normalized around Vorkath arena ~2272, 4052)
    obs[8] = (state.get('playerX', 2272) - 2272) / 50.0
    obs[9] = (state.get('playerY', 4052) - 4052) / 50.0
    obs[10] = state.get('playerPlane', 0) / 3.0
    
    # === TARGET INFO (indices 20-39) ===
    obs[20] = 1.0 if state.get('hasTarget', False) else 0.0
    obs[21] = state.get('targetHp', 0) / 750.0
    obs[22] = state.get('targetMaxHp', 750) / 750.0
    obs[23] = (state.get('targetX', 0) - state.get('playerX', 0)) / 20.0
    obs[24] = (state.get('targetY', 0) - state.get('playerY', 0)) / 20.0
    obs[25] = state.get('targetAnimation', 0) / 10000.0
    
    # === PRAYERS (indices 40-69) ===
    prayers = state.get('activePrayers', [False] * 29)
    for i, active in enumerate(prayers[:29]):
        obs[40 + i] = 1.0 if active else 0.0
    
    # === INVENTORY (indices 70-129) ===
    inv_ids = state.get('inventoryIds', [])
    inv_qtys = state.get('inventoryQuantities', [])
    
    # Food count
    food_ids = [385, 391, 7946, 379, 365, 373, 361]  # Sharks, manta, monkfish, etc.
    obs[70] = sum(1 for i in inv_ids if i in food_ids) / 28.0
    
    # Prayer potion count
    prayer_ids = [2434, 139, 141, 143, 3024, 3026, 3028, 3030]
    obs[71] = sum(1 for i in inv_ids if i in prayer_ids) / 8.0
    
    # Antifire count
    antifire_ids = [2452, 2454, 2456, 2458, 21978, 21981, 21984, 21987]
    obs[72] = sum(1 for i in inv_ids if i in antifire_ids) / 4.0
    
    # Antivenom count
    antivenom_ids = [12913, 12915, 12917, 12919, 12905, 12907, 12909, 12911]
    obs[73] = sum(1 for i in inv_ids if i in antivenom_ids) / 4.0
    
    # Raw inventory slots (28 slots * 2 = 56 values)
    for i, item_id in enumerate(inv_ids[:28]):
        obs[80 + i] = item_id / 30000.0 if item_id > 0 else 0.0
    for i, qty in enumerate(inv_qtys[:28]):
        obs[108 + i] = qty / 1000.0 if qty > 0 else 0.0
    
    # === EQUIPMENT (indices 130-149) ===
    equip = state.get('equipmentIds', [])
    for i, item_id in enumerate(equip[:11]):
        obs[130 + i] = item_id / 30000.0 if item_id > 0 else 0.0
    
    # === SKILLS (indices 150-179) ===
    skills = state.get('skillLevels', [])
    for i, level in enumerate(skills[:23]):
        obs[150 + i] = level / 99.0
    
    # === NEARBY NPCS (indices 180-249) ===
    npcs = state.get('nearbyNpcs', [])
    for i, npc in enumerate(npcs[:10]):
        base = 180 + i * 7
        obs[base] = npc.get('id', 0) / 10000.0
        obs[base + 1] = (npc.get('x', 0) - state.get('playerX', 0)) / 20.0
        obs[base + 2] = (npc.get('y', 0) - state.get('playerY', 0)) / 20.0
        obs[base + 3] = npc.get('hp', 0) / 750.0
        obs[base + 4] = npc.get('maxHp', 1) / 750.0
        obs[base + 5] = npc.get('animation', 0) / 10000.0
        obs[base + 6] = npc.get('distance', 0) / 20.0
    
    # === NEARBY OBJECTS (indices 250-319) ===
    objects = state.get('nearbyObjects', [])
    for i, obj in enumerate(objects[:10]):
        base = 250 + i * 7
        obs[base] = obj.get('id', 0) / 50000.0
        obs[base + 1] = (obj.get('x', 0) - state.get('playerX', 0)) / 20.0
        obs[base + 2] = (obj.get('y', 0) - state.get('playerY', 0)) / 20.0
        obs[base + 3] = obj.get('distance', 0) / 20.0
    
    return obs

print("\n" + "="*50)
print("VORKATH RL AGENT - LIVE")
print("="*50)
print("Make sure you're at Vorkath and the fight is starting!")
print("Press Ctrl+C to stop\n")

try:
    tick = 0
    while True:
        state = get_state()
        if not state:
            print("Lost connection!")
            break
        
        obs = state_to_obs(state, obs_size)
        action, _ = model.predict(obs, deterministic=True)
        action_id = int(action)
        
        # Display status
        hp = state.get('playerHp', 0)
        prayer = state.get('playerPrayer', 0)
        target_hp = state.get('targetHp', 0)
        has_target = state.get('hasTarget', False)
        pos = f"({state.get('playerX', 0)}, {state.get('playerY', 0)})"
        
        action_name = ACTION_NAMES[action_id] if action_id < len(ACTION_NAMES) else f"ACTION_{action_id}"
        
        print(f"Tick {tick}: HP={hp} Prayer={prayer} VorkathHP={target_hp} Target={has_target} Pos={pos} -> {action_name}")
        
        # Send action to game
        send_action(action_id)
        
        tick += 1
        time.sleep(0.6)  # Game tick rate
        
except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    sock.close()
