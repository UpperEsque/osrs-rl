#!/usr/bin/env python3
"""
Train walking agent on real OSRS - with continuous walking
"""
import sys
sys.path.insert(0, './python')

from osrs_rl.client import GameClient
from osrs_rl.protocol import Actions
import time
import math

def main():
    print("=" * 50)
    print("OSRS RL - Walking Training (Real Game)")
    print("=" * 50)
    
    client = GameClient(port=5555)
    
    if not client.connect(timeout=30):
        print("Failed to connect!")
        return
    
    print("✓ Connected!")
    
    # Get initial state
    state = client.get_state(timeout=5)
    if not state:
        print("No state received")
        return
    
    start_x, start_y = state.player_x, state.player_y
    print(f"Starting position: ({start_x}, {start_y})")
    
    # Set a target 10 tiles east
    target_x = start_x + 10
    target_y = start_y
    print(f"Target position: ({target_x}, {target_y})")
    
    print("\nWalking to target...")
    
    for i in range(30):
        state = client.get_state(timeout=2)
        if not state:
            continue
            
        dx = target_x - state.player_x
        dy = target_y - state.player_y
        dist = math.sqrt(dx*dx + dy*dy)
        
        print(f"Tick {state.tick}: Pos=({state.player_x}, {state.player_y}) Distance={dist:.1f} Moving={state.player_is_moving}")
        
        if dist < 2:
            print("\n✓ Reached target!")
            break
        
        # Send walk command if not moving or every few ticks
        if not state.player_is_moving or i % 3 == 0:
            # Walk towards target in smaller steps
            step_x = state.player_x + min(5, max(-5, dx))
            step_y = state.player_y + min(5, max(-5, dy))
            print(f"  -> Sending walk to ({step_x}, {step_y})")
            client.walk_to(step_x, step_y)
        
        time.sleep(0.6)  # Wait roughly 1 game tick
    
    client.disconnect()
    print("\nDone!")

if __name__ == "__main__":
    main()
