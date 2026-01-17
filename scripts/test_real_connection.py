#!/usr/bin/env python3
"""Test connection to real RuneLite plugin"""
import sys
sys.path.insert(0, './python')

from osrs_rl.client import GameClient

def main():
    print("=" * 50)
    print("OSRS RL - Real Game Connection Test")
    print("=" * 50)
    print("\nMake sure:")
    print("  1. RuneLite is running")
    print("  2. You're logged into OSRS")
    print("  3. OSRS-RL plugin is enabled")
    print("\nConnecting...")
    
    client = GameClient(port=5555)
    
    if not client.connect(timeout=30):
        print("\n❌ Failed to connect!")
        print("Check that the plugin is enabled in RuneLite")
        return
    
    print("✓ Connected!\n")
    print("Receiving game state...")
    
    for i in range(20):
        state = client.get_state(timeout=3)
        if state:
            dist_to_lumb = ((state.player_x - 3222)**2 + (state.player_y - 3218)**2)**0.5
            print(f"\nTick {state.tick}:")
            print(f"  Position: ({state.player_x}, {state.player_y}, plane={state.player_plane})")
            print(f"  HP: {state.player_hp}/{state.player_max_hp}")
            print(f"  Prayer: {state.player_prayer}/{state.player_max_prayer}")
            print(f"  Run Energy: {state.player_energy}%")
            print(f"  Animation: {state.player_animation}")
            print(f"  In Combat: {state.player_in_combat}")
            print(f"  Nearby NPCs: {len(state.nearby_npcs)}")
            print(f"  Nearby Players: {len(state.nearby_players)}")
            print(f"  Nearby Objects: {len(state.nearby_objects)}")
            
            if state.nearby_npcs:
                print(f"  First NPC: {state.nearby_npcs[0].name} (dist={state.nearby_npcs[0].distance})")
        else:
            print("No state received (waiting...)")
    
    client.disconnect()
    print("\n✓ Test complete!")

if __name__ == "__main__":
    main()
