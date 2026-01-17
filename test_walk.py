#!/usr/bin/env python3
import socket
import json
import time

WIN_IP = "172.17.112.1"

print("Starting in 3 seconds... CLICK ON RUNELITE WINDOW NOW!")
time.sleep(3)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(10)
print(f"Connecting to {WIN_IP}:5555...")
sock.connect((WIN_IP, 5555))
print("Connected!")

buffer = ""
while True:
    data = sock.recv(4096).decode('utf-8')
    buffer += data
    try:
        state = json.loads(buffer)
        x = state['data']['playerX']
        y = state['data']['playerY']
        print(f"Player at ({x}, {y})")
        break
    except:
        continue

target_x = x + 5
target_y = y
print(f"Sending WALK_TO ({target_x}, {target_y})...")

action_msg = json.dumps({"type": "action", "action": [1, target_x, target_y]}) + "\n"
sock.send(action_msg.encode())

for i in range(5):
    time.sleep(0.6)
    buffer = ""
    while True:
        data = sock.recv(4096).decode('utf-8')
        buffer += data
        try:
            state = json.loads(buffer)
            new_x = state['data']['playerX']
            new_y = state['data']['playerY']
            print(f"  Tick {i+1}: ({new_x}, {new_y})")
            break
        except:
            continue

sock.close()
print("Done!")
