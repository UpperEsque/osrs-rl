#!/usr/bin/env python3
"""
Vorkath RL Client - Connects to RuneLite plugin and runs trained model

Usage:
    python vorkath_client.py --model models/vorkath_detailed_xxx/best_model.zip
"""
import sys
import json
import socket
import time
import argparse
import glob
import numpy as np

def find_best_model():
    """Find the best Vorkath model"""
    patterns = [
        "models/vorkath_lstm_*/best_model.zip",
        "models/vorkath_detailed_*/best_model.zip",
        "models/vorkath_frames*/best_model.zip",
        "models/vorkath_*/best_model.zip",
    ]
    for pattern in patterns:
        paths = glob.glob(pattern)
        if paths:
            return sorted(paths)[-1]
    return None

class VorkathClient:
    def __init__(self, host='localhost', port=5050):
        self.host = host
        self.port = port
        self.sock = None
        self.model = None
        
    def connect(self, timeout=30):
        """Connect to RuneLite plugin"""
        print(f"Connecting to {self.host}:{self.port}...")
        
        start = time.time()
        while time.time() - start < timeout:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                self.sock.settimeout(5.0)
                print("Connected!")
                return True
            except (ConnectionRefusedError, socket.timeout):
                time.sleep(1)
        
        print("Connection failed!")
        return False
    
    def load_model(self, model_path):
        """Load trained model"""
        try:
            from stable_baselines3 import PPO
            print(f"Loading model: {model_path}")
            self.model = PPO.load(model_path)
            print("Model loaded!")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def run(self):
        """Main loop - receive state, send action"""
        if not self.model:
            print("No model loaded!")
            return
        
        buffer = ""
        action_names = [
            "WAIT", "ATTACK_RANGED", "ATTACK_SPEC", "PRAY_MAGE", "PRAY_RANGE",
            "PRAY_OFF", "TOGGLE_RIGOUR", "EAT_FOOD", "DRINK_PRAYER", "DRINK_ANTIFIRE",
            "DRINK_ANTIVENOM", "MOVE_NORTH", "MOVE_SOUTH", "MOVE_EAST", "MOVE_WEST",
            "WALK_AROUND", "CAST_CRUMBLE_UNDEAD"
        ]
        
        print("\n" + "="*50)
        print("VORKATH RL CLIENT RUNNING")
        print("="*50)
        print("Press Ctrl+C to stop\n")
        
        tick = 0
        try:
            while True:
                # Receive state
                try:
                    data = self.sock.recv(4096).decode('utf-8')
                    if not data:
                        print("Disconnected from server")
                        break
                    
                    buffer += data
                    
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if not line.strip():
                            continue
                        
                        try:
                            msg = json.loads(line)
                            obs = np.array(msg.get('observation', []), dtype=np.float32)
                            
                            if len(obs) > 0:
                                # Get action from model
                                action, _ = self.model.predict(obs, deterministic=True)
                                action_id = int(action)
                                action_name = action_names[action_id] if action_id < len(action_names) else f"ACTION_{action_id}"
                                
                                # Send action
                                response = {"action": [action_id]}
                                self.sock.send((json.dumps(response) + '\n').encode('utf-8'))
                                
                                tick += 1
                                if tick % 5 == 0:  # Log every 5 ticks
                                    hp = obs[0] * 99 if len(obs) > 0 else 0
                                    vork_hp = obs[15] * 750 if len(obs) > 15 else 0
                                    print(f"Tick {tick:4d} | HP: {hp:3.0f} | Vorkath: {vork_hp:3.0f} | Action: {action_name}")
                        
                        except json.JSONDecodeError:
                            pass
                        except Exception as e:
                            print(f"Error: {e}")
                
                except socket.timeout:
                    continue
                    
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            self.close()
    
    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None

def main():
    parser = argparse.ArgumentParser(description='Vorkath RL Client')
    parser.add_argument('--model', type=str, default=None, help='Path to model')
    parser.add_argument('--host', type=str, default='localhost', help='Plugin host')
    parser.add_argument('--port', type=int, default=5050, help='Plugin port')
    
    args = parser.parse_args()
    
    # Find model
    model_path = args.model or find_best_model()
    if not model_path:
        print("No model found!")
        print("Train one first or specify with --model")
        sys.exit(1)
    
    # Create client
    client = VorkathClient(args.host, args.port)
    
    # Load model
    if not client.load_model(model_path):
        sys.exit(1)
    
    # Connect and run
    if client.connect():
        client.run()

if __name__ == "__main__":
    main()
