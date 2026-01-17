#!/usr/bin/env python3
"""Run trained model on live OSRS game"""
import socket
import json
import time
import numpy as np
import subprocess
import argparse
from pathlib import Path

def get_windows_ip():
    """Get Windows host IP from WSL"""
    try:
        result = subprocess.getoutput("cat /etc/resolv.conf | grep nameserver | awk '{print $2}'")
        return result.strip()
    except:
        return "localhost"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model.zip")
    parser.add_argument("--ip", type=str, default=None, help="Windows IP (auto-detect if not set)")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    # Load model
    from stable_baselines3 import PPO
    print(f"Loading model: {args.model}")
    model = PPO.load(args.model)
    obs_size = model.observation_space.shape[0]
    print(f"Model loaded. Observation size: {obs_size}")

    # Get IP
    win_ip = args.ip or get_windows_ip()
    
    print(f"\n*** CLICK ON RUNELITE WINDOW NOW! ***")
    print("Starting in 3 seconds...")
    time.sleep(3)

    # Connect
    print(f"Connecting to {win_ip}:{args.port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    sock.connect((win_ip, args.port))
    print("Connected!\n")

    def state_to_obs(state):
        """Convert game state to model observation"""
        obs = np.zeros(obs_size, dtype=np.float32)
        obs[0] = state.get('playerHp', 99) / 99.0
        obs[1] = state.get('playerPrayer', 99) / 99.0
        obs[2] = state.get('playerEnergy', 100) / 100.0
        obs[3] = state.get('targetHpPercent', 0) / 100.0
        obs[4] = 1.0 if state.get('hasTarget', False) else 0.0
        # Add more mappings as needed
        return obs

    print("="*50)
    print("AGENT RUNNING - Press Ctrl+C to stop")
    print("="*50 + "\n")

    tick = 0
    try:
        while True:
            # Receive state
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
                except json.JSONDecodeError:
                    continue

            # Get action from model
            obs = state_to_obs(state)
            action, _ = model.predict(obs, deterministic=True)

            # Log
            hp = state.get('playerHp', 0)
            prayer = state.get('playerPrayer', 0)
            target = state.get('targetHpPercent', 0)
            print(f"T{tick}: HP={hp} Pray={prayer} Target={target}% -> Action {action}")

            # Send action
            cmd = {"type": "action", "action": [int(action)]}
            sock.send((json.dumps(cmd) + "\n").encode())

            tick += 1
            time.sleep(0.6)

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
