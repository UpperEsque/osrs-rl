#!/usr/bin/env python3
"""
Test connection to the plugin
"""
import sys
sys.path.insert(0, './python')

from osrs_rl.client import GameClient

def main():
    print("Connecting to OSRS RL plugin...")
    
    client = GameClient(port=5555)
    
    if not client.connect(timeout=10):
        print("Failed to connect! Make sure:")
        print("  1. RuneLite is running")
        print("  2. OSRS RL plugin is enabled")
        print("  3. You're logged into the game")
        return
    
    print("Connected! Waiting for game state...")
    
    for i in range(10):
        state = client.get_state(timeout=2)
        if state:
            print(f"\nTick {state.tick}:")
            print(f"  Position: ({state.player_x}, {state.player_y})")
            print(f"  HP: {state.player_hp}/{state.player_max_hp}")
            print(f"  Prayer: {state.player_prayer}/{state.player_max_prayer}")
            print(f"  Animation: {state.player_animation}")
            print(f"  Moving: {state.player_is_moving}")
            print(f"  Nearby NPCs: {len(state.nearby_npcs)}")
            print(f"  Nearby Objects: {len(state.nearby_objects)}")
            
            # Test walk action
            if i == 5:
                print("\n  Sending walk action...")
                client.walk_to(state.player_x + 3, state.player_y)
        else:
            print("No state received")
    
    client.disconnect()
    print("\nDone!")

if __name__ == "__main__":
    main()
