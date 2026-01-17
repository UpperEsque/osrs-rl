#!/usr/bin/env python3
"""Rule-based Vorkath bot"""
import socket
import json
import time

WIN_IP = "172.17.112.1"

def find_item_slot(inv_ids, item_list):
    """Find first inventory slot containing any item from list"""
    for i, item in enumerate(inv_ids):
        if item in item_list:
            return i
    return -1

def main():
    print(f"Connecting to {WIN_IP}:5555...")
    print("You have 3 seconds - CLICK ON RUNELITE!")
    time.sleep(3)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    sock.connect((WIN_IP, 5555))
    print("Connected!")

    print("\n" + "="*50)
    print("VORKATH RULE-BASED BOT")
    print("="*50)
    print("Press Ctrl+C to stop\n")

    # Item IDs
    FOOD = [385, 391, 7946, 379, 3144, 373, 7060]  # Shark, manta, etc
    PRAYER_POT = [2434, 139, 141, 143, 3024, 3026, 3028, 3030]
    ANTIFIRE = [2452, 2454, 2456, 2458, 21978, 21981, 21984, 21987]
    ANTIVENOM = [12913, 12915, 12917, 12919, 12905, 12907, 12909, 12911]

    tick = 0
    last_action = ""
    
    try:
        while True:
            # Get state
            buffer = ""
            while True:
                data = sock.recv(8192).decode('utf-8')
                if not data:
                    raise Exception("Disconnected")
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
            max_prayer = state.get('playerMaxPrayer', 99)
            inv_ids = state.get('inventoryIds', [])
            player_x = state.get('playerX', 0)
            player_y = state.get('playerY', 0)
            
            # Find Vorkath
            vorkath_hp = 0
            vorkath_id = -1
            for npc in state.get('nearbyNpcs', []):
                name = npc.get('name', '').lower()
                if 'vorkath' in name:
                    vorkath_hp = npc.get('hp', 0)
                    vorkath_id = npc.get('id', -1)
                    break

            action = None
            action_name = "WAIT"

            # Priority 1: Eat if low HP
            if hp < 50:
                slot = find_item_slot(inv_ids, FOOD)
                if slot >= 0:
                    action = {"type": "action", "action": [2, slot]}
                    action_name = f"EAT (slot {slot})"

            # Priority 2: Prayer pot if low prayer
            elif prayer < 30:
                slot = find_item_slot(inv_ids, PRAYER_POT)
                if slot >= 0:
                    action = {"type": "action", "action": [2, slot]}
                    action_name = f"DRINK_PRAYER (slot {slot})"

            # Priority 3: Attack Vorkath if not in combat
            elif vorkath_hp > 0 and not state.get('playerInCombat', False):
                action = {"type": "action", "action": [3, vorkath_id]}
                action_name = f"ATTACK_VORKATH (id {vorkath_id})"

            # Log status
            print(f"T{tick}: HP={hp}/{max_hp} Pray={prayer} Vork={vorkath_hp} -> {action_name}")

            # Send action
            if action:
                sock.send((json.dumps(action) + "\n").encode())
                last_action = action_name

            tick += 1
            time.sleep(0.6)

    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
