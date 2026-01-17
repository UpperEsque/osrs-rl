#!/usr/bin/env python3
"""
Vorkath Model Server for RuneLite Plugin
"""
import sys
sys.path.insert(0, 'python')

import json
import socket
import numpy as np
from stable_baselines3 import PPO
import glob

# Find best Vorkath model
model_paths = glob.glob("models/vorkath_*/best_model.zip") + \
              glob.glob("models/vorkath_detailed_*/best_model.zip") + \
              glob.glob("models/vorkath_lstm_*/best_model.zip")

if not model_paths:
    print("No Vorkath model found!")
    sys.exit(1)

model_path = sorted(model_paths)[-1]
print(f"Loading model: {model_path}")
model = PPO.load(model_path)
print("Model loaded!")

# Action mapping to game actions
ACTIONS = {
    0: "WAIT",
    1: "ATTACK_RANGED",
    2: "ATTACK_SPEC",
    3: "PRAY_MAGE",
    4: "PRAY_RANGE",
    5: "PRAY_OFF",
    6: "TOGGLE_RIGOUR",
    7: "EAT_FOOD",
    8: "DRINK_PRAYER",
    9: "DRINK_ANTIFIRE",
    10: "MOVE_NORTH",
    11: "MOVE_SOUTH",
    12: "MOVE_EAST",
    13: "MOVE_WEST",
    14: "WALK_AROUND",
    15: "CAST_CRUMBLE_UNDEAD",
}

def start_server(port=5050):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', port))
    server.listen(5)
    print(f"Server listening on port {port}...")
    
    while True:
        conn, addr = server.accept()
        print(f"Connection from {addr}")
        
        try:
            while True:
                data = conn.recv(4096).decode('utf-8')
                if not data:
                    break
                
                # Parse observation from plugin
                obs = json.loads(data)
                obs_array = np.array(obs['observation'], dtype=np.float32)
                
                # Get action from model
                action, _ = model.predict(obs_array, deterministic=True)
                action_name = ACTIONS.get(int(action), "WAIT")
                
                # Send response
                response = {
                    "action": int(action),
                    "action_name": action_name,
                }
                conn.send((json.dumps(response) + "\n").encode('utf-8'))
                print(f"Obs -> Action: {action_name}")
                
        except Exception as e:
            print(f"Error: {e}")
        finally:
            conn.close()

if __name__ == "__main__":
    start_server(5050)
