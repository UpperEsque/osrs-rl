#!/usr/bin/env python3
"""Test connection to the OSRS-RL plugin"""
import socket
import json

def test_connection():
    HOST = 'localhost'
    PORT = 5555
    
    print(f"Connecting to {HOST}:{PORT}...")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((HOST, PORT))
        print("✓ Connected!")
        
        # Read data
        buffer = ""
        while True:
            data = sock.recv(4096).decode('utf-8')
            if not data:
                break
            buffer += data
            
            # Try to parse JSON
            try:
                # Find complete JSON objects
                if buffer.strip().startswith('{'):
                    # Try to parse
                    state = json.loads(buffer)
                    print("\n✓ Received game state!")
                    print(f"  Player X: {state.get('playerX', 'N/A')}")
                    print(f"  Player Y: {state.get('playerY', 'N/A')}")
                    print(f"  Player HP: {state.get('playerHp', 'N/A')}/{state.get('playerMaxHp', 'N/A')}")
                    print(f"  Nearby NPCs: {len(state.get('nearbyNpcs', []))}")
                    
                    if state.get('nearbyNpcs'):
                        for npc in state['nearbyNpcs'][:3]:
                            print(f"    - {npc.get('name', 'Unknown')} (dist: {npc.get('distance', '?')})")
                    
                    buffer = ""
                    break
            except json.JSONDecodeError:
                # Need more data
                continue
                
        sock.close()
        print("\n✓ Connection test successful!")
        return True
        
    except socket.timeout:
        print("✗ Connection timed out")
        return False
    except ConnectionRefusedError:
        print("✗ Connection refused - is the plugin running?")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    test_connection()
