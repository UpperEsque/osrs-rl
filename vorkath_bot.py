#!/usr/bin/env python3
"""Vorkath bot v4 - fireball dodging"""
import socket
import json
import time

WIN_IP = "172.17.112.1"

# Vorkath animations  
VORKATH_MAGIC = 7950      # Pray mage
VORKATH_RANGED = 7952     # Pray range
VORKATH_FIREBALL = 7960   # Fireball - MOVE AWAY!
VORKATH_ACID = 7957       # Acid phase
VORKATH_SPAWN = 7959      # Zombified spawn

# Item IDs
SHARK = 385
PRAYER_POT = [2434, 139, 141, 143]

# Prayer indices
PROTECT_MAGIC = 16
PROTECT_RANGE = 17

def find_slot(inv, items):
    if isinstance(items, int):
        items = [items]
    for i, item in enumerate(inv):
        if item in items:
            return i
    return -1

def count_items(inv, items):
    if isinstance(items, int):
        items = [items]
    return sum(1 for item in inv if item in items)

def main():
    print(f"Connecting to {WIN_IP}:5555...")
    print("Click on RuneLite in 3 seconds!")
    time.sleep(3)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    sock.connect((WIN_IP, 5555))
    print("Connected!\n")

    print("="*50)
    print("VORKATH BOT v4 - FIREBALL DODGE")
    print("="*50 + "\n")

    tick = 0
    last_hp = 99
    last_vork_anim = -1
    dodge_ticks = 0
    
    try:
        while True:
            buffer = ""
            while True:
                data = sock.recv(8192).decode('utf-8')
                buffer += data
                try:
                    msg = json.loads(buffer)
                    state = msg.get('data', msg)
                    break
                except:
                    continue

            hp = state.get('playerHp', 99)
            max_hp = state.get('playerMaxHp', 99)
            prayer = state.get('playerPrayer', 99)
            inv = state.get('inventoryIds', [])
            prayers_active = state.get('activePrayers', [])
            player_x = state.get('playerX', 0)
            player_y = state.get('playerY', 0)
            in_combat = state.get('playerInCombat', False)
            
            # Get Vorkath info
            vorkath = None
            for npc in state.get('nearbyNpcs', []):
                name = npc.get('name', '').lower()
                if 'vorkath' in name:
                    vorkath = npc
                    break
            
            vork_hp = vorkath.get('hp', 0) if vorkath else 0
            vork_anim = vorkath.get('animation', -1) if vorkath else -1
            vork_id = vorkath.get('id', -1) if vorkath else -1
            
            has_mage = len(prayers_active) > PROTECT_MAGIC and prayers_active[PROTECT_MAGIC]
            has_range = len(prayers_active) > PROTECT_RANGE and prayers_active[PROTECT_RANGE]
            
            action = None
            action_name = "WAIT"
            alert = ""
            
            # Detect NEW fireball attack
            if vork_anim == VORKATH_FIREBALL and last_vork_anim != VORKATH_FIREBALL:
                dodge_ticks = 3  # Dodge for 3 ticks
                alert = " !!! FIREBALL - DODGING !!!"
            
            last_vork_anim = vork_anim
            
            # PRIORITY 1: DODGE FIREBALL - move 2 tiles west/east
            if dodge_ticks > 0:
                # Alternate direction based on tick
                if tick % 2 == 0:
                    action = {"type": "action", "action": [1, player_x + 2, player_y]}
                    action_name = ">>> DODGE EAST <<<"
                else:
                    action = {"type": "action", "action": [1, player_x - 2, player_y]}
                    action_name = ">>> DODGE WEST <<<"
                dodge_ticks -= 1

            # PRIORITY 2: Eat if HP critical
            elif hp > 0 and hp < 40:
                slot = find_slot(inv, SHARK)
                if slot >= 0:
                    action = {"type": "action", "action": [2, slot]}
                    action_name = f"!! EAT !! (slot {slot})"

            # PRIORITY 3: Prayer switching
            elif vork_anim == VORKATH_RANGED and not has_range:
                action = {"type": "action", "action": [5, PROTECT_RANGE]}
                action_name = "SWITCH TO RANGE"
            
            elif vork_anim == VORKATH_MAGIC and not has_mage:
                action = {"type": "action", "action": [5, PROTECT_MAGIC]}
                action_name = "SWITCH TO MAGE"

            # PRIORITY 4: Eat if HP < 55
            elif hp > 0 and hp < 55:
                slot = find_slot(inv, SHARK)
                if slot >= 0:
                    action = {"type": "action", "action": [2, slot]}
                    action_name = f"EAT (slot {slot})"

            # PRIORITY 5: Prayer pot
            elif prayer < 30:
                slot = find_slot(inv, PRAYER_POT)
                if slot >= 0:
                    action = {"type": "action", "action": [2, slot]}
                    action_name = f"PRAYER POT (slot {slot})"

            # PRIORITY 6: Attack
            elif vorkath and vork_hp > 0 and not in_combat and dodge_ticks == 0:
                action = {"type": "action", "action": [3, vork_id]}
                action_name = "ATTACK VORKATH"

            # Status
            pray_str = "MAGE" if has_mage else "RANGE" if has_range else "OFF"
            food_count = count_items(inv, SHARK)
            
            dmg = ""
            if hp < last_hp and last_hp > 0 and hp > 0:
                dmg = f" [TOOK {last_hp - hp} DMG]"
            last_hp = hp

            print(f"T{tick}: HP={hp}/{max_hp} Pray={prayer} [{pray_str}] Vork={vork_hp} anim={vork_anim} Food={food_count} -> {action_name}{dmg}{alert}")

            if action:
                sock.send((json.dumps(action) + "\n").encode())

            tick += 1
            time.sleep(0.6)

    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
