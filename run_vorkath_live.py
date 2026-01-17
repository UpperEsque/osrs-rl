#!/usr/bin/env python3
import socket
import json
import time
import numpy as np
from stable_baselines3 import PPO

WIN_IP = "10.255.255.254"
MODEL_PATH = "/home/azdelic/osrs-rl/models/vorkath_detailed_20260117_000020/best_model.zip"

print("Loading model...")
model = PPO.load(MODEL_PATH)
print(f"Model loaded! Obs space: {model.observation_space.shape}")

print("\nCLICK ON RUNELITE NOW! Starting in 3 seconds...")
time.sleep(3)

print(f"Connecting to {WIN_IP}:5555...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(10)
sock.connect((WIN_IP, 5555))
print("Connected!")

ACTION_NAMES = ["NOOP", "WALK_TO", "CLICK_INV", "ATTACK_NPC", "INTERACT_OBJ", 
                "TOGGLE_PRAYER", "TOGGLE_RUN", "SPEC_ATTACK"]

def state_to_obs(state):
    obs = np.zeros(model.observation_space.shape[0], dtype=np.float32)
    obs[0] = state.get('playerHp', 99) / 99.0
    obs[1] = state.get('playerPrayer', 99) / 99.0
    obs[2] = state.get('playerEnergy', 100) / 100.0
    obs[3] = state.get('targetHpPercent', 0) / 100.0
    obs[4] = 1.0 if state.get('hasTarget', False) else 0.0
    obs[5] = (state.get('playerX', 2272) - 2272) / 20.0
    obs[6] = (state.get('playerY', 4052) - 4052) / 20.0
    return obs

print("\n" + "="*50)
print("VORKATH AGENT RUNNING - Go fight Vorkath!")
print("="*50 + "\n")

try:
    tick = 0
    while True:
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

        obs = state_to_obs(state)
        action, _ = model.predict(obs, deterministic=True)
        
        hp = state.get('playerHp', 0)
        prayer = state.get('playerPrayer', 0)
        target_hp = state.get('targetHpPercent', 0)
        
        print(f"T{tick}: HP={hp} Pray={prayer} Target={target_hp}% -> Action {action}")
        
        # Send action to plugin
        cmd = {"type": "action", "action": [int(action)]}
        sock.send((json.dumps(cmd) + "\n").encode())
        
        tick += 1
        time.sleep(0.6)

except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    sock.close()
