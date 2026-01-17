#!/usr/bin/env python3
import socket
import json

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(5)
sock.connect(('localhost', 5555))

buffer = ""
while True:
    data = sock.recv(4096).decode('utf-8')
    if not data:
        break
    buffer += data
    try:
        state = json.loads(buffer)
        print("Fields in state:")
        for key, value in state.items():
            if isinstance(value, list):
                print(f"  {key}: [{len(value)} items]")
            else:
                print(f"  {key}: {value}")
        break
    except:
        continue

sock.close()
