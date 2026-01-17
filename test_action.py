#!/usr/bin/env python3
"""Test sending actions to the plugin"""
import socket
import json
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('localhost', 5555))
print("Connected!")

# Read initial state
buffer = ""
while True:
    data = sock.recv(4096).decode('utf-8')
    buffer += data
    try:
        state = json.loads(buffer)
        print(f"Got state - Player at ({state['data']['playerX']}, {state['data']['playerY']})")
        break
    except:
        continue

# Test sending an action (7 = EAT_FOOD)
print("\nSending EAT_FOOD action...")
action_msg = json.dumps({"type": "action", "action": 7}) + "\n"
sock.send(action_msg.encode())

time.sleep(1)

# Read response
buffer = ""
while True:
    data = sock.recv(4096).decode('utf-8')
    buffer += data
    try:
        response = json.loads(buffer)
        print(f"Response: {response}")
        break
    except:
        continue

sock.close()
print("Done!")
