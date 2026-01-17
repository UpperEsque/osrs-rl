#!/usr/bin/env python3
"""Show all game data from RuneLite"""
import socket
import json
import time

WIN_IP = "172.17.112.1"

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(10)
sock.connect((WIN_IP, 5555))
print("Connected!\n")

tick = 0
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

        print("=" * 60)
        print(f"TICK {tick}")
        print("=" * 60)
        
        print(f"\n--- PLAYER ---")
        print(f"Position: ({state.get('playerX')}, {state.get('playerY')}) plane={state.get('playerPlane')}")
        print(f"HP: {state.get('playerHp')}/{state.get('playerMaxHp')}")
        print(f"Prayer: {state.get('playerPrayer')}/{state.get('playerMaxPrayer')}")
        print(f"Energy: {state.get('playerEnergy')}")
        print(f"Animation: {state.get('playerAnimation')}")
        print(f"Moving: {state.get('playerIsMoving')}")
        print(f"In Combat: {state.get('playerInCombat')}")
        
        print(f"\n--- TARGET ---")
        print(f"Has Target: {state.get('hasTarget')}")
        print(f"Target HP: {state.get('targetHp')}/{state.get('targetMaxHp')}")
        print(f"Target Pos: ({state.get('targetX')}, {state.get('targetY')})")
        print(f"Target Anim: {state.get('targetAnimation')}")
        
        print(f"\n--- PRAYERS ---")
        prayers = state.get('activePrayers', [])
        active = [i for i, p in enumerate(prayers) if p]
        print(f"Active prayers: {active if active else 'None'}")
        
        print(f"\n--- INVENTORY ---")
        inv = state.get('inventoryIds', [])
        inv_qty = state.get('inventoryQuantities', [])
        for i, (item, qty) in enumerate(zip(inv, inv_qty)):
            if item > 0:
                print(f"  Slot {i}: ID={item} x{qty}")
        
        print(f"\n--- NEARBY NPCS ---")
        for npc in state.get('nearbyNpcs', [])[:5]:
            print(f"  {npc.get('name')} (id={npc.get('id')}) HP={npc.get('hp')}/{npc.get('maxHp')} dist={npc.get('distance')} anim={npc.get('animation')}")
        
        print(f"\n--- EQUIPMENT ---")
        equip = state.get('equipmentIds', [])
        slots = ['Head', 'Cape', 'Neck', 'Weapon', 'Body', 'Shield', 'Legs', 'Gloves', 'Boots', 'Ring', 'Ammo']
        for i, item in enumerate(equip):
            if item > 0 and i < len(slots):
                print(f"  {slots[i]}: {item}")
        
        print("\n")
        tick += 1
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopped")
finally:
    sock.close()
