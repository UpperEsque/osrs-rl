import websocket
import json

try:
    ws = websocket.create_connection("ws://localhost:5555")
    print("Connected to plugin!")
    
    # Get game state
    ws.send(json.dumps({"type": "get_state"}))
    response = json.loads(ws.recv())
    print("Response:", response)
    
    ws.close()
except Exception as e:
    print(f"Connection failed: {e}")
