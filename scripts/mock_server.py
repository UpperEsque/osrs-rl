#!/usr/bin/env python3
"""
Mock OSRS server for testing Python RL code without RuneLite
Simulates a simple 2D grid world with walking task
"""
import socket
import json
import threading
import time
import math

class MockOSRSServer:
    def __init__(self, port=5555):
        self.port = port
        self.running = False
        
        # Simulated player state
        self.reset_position()
        self.tick = 0
        
        # Target for walking task
        self.target_x = 3232
        self.target_y = 3228
        
        # Stats
        self.episodes = 0
        self.total_steps = 0
        self.best_distance = float('inf')
        
    def reset_position(self):
        self.player_x = 3222
        self.player_y = 3218
        
    def get_distance(self):
        dx = self.player_x - self.target_x
        dy = self.player_y - self.target_y
        return math.sqrt(dx*dx + dy*dy)
        
    def start(self):
        self.running = True
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(('localhost', self.port))
        self.server.listen(1)
        
        print(f"[MockServer] Listening on port {self.port}")
        print(f"[MockServer] Player starts at ({self.player_x}, {self.player_y})")
        print(f"[MockServer] Target at ({self.target_x}, {self.target_y})")
        print(f"[MockServer] Initial distance: {self.get_distance():.1f}")
        print("-" * 50)
        
        while self.running:
            try:
                conn, addr = self.server.accept()
                print(f"[MockServer] Client connected")
                self.handle_client(conn)
            except Exception as e:
                if self.running:
                    print(f"[MockServer] Error: {e}")
    
    def handle_client(self, conn):
        conn.settimeout(1.0)
        
        # Start sending ticks
        tick_thread = threading.Thread(target=self.send_ticks, args=(conn,))
        tick_thread.daemon = True
        tick_thread.start()
        
        # Read actions
        buffer = ""
        while self.running:
            try:
                data = conn.recv(4096).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        self.handle_message(line)
                        
            except socket.timeout:
                continue
            except Exception as e:
                break
        
        print(f"\n[MockServer] Session ended - {self.episodes} episodes, {self.total_steps} steps")
        print(f"[MockServer] Best distance achieved: {self.best_distance:.1f}")
    
    def handle_message(self, line):
        try:
            msg = json.loads(line)
            if msg.get('type') == 'action':
                action = msg.get('data', [0])
                self.execute_action(action)
            elif msg.get('type') == 'reset':
                self.reset()
        except Exception as e:
            pass
    
    def execute_action(self, action):
        if not action:
            return
            
        action_type = action[0]
        self.total_steps += 1
        
        if action_type == 0:  # NOOP
            pass
        elif action_type == 1:  # WALK_TO
            if len(action) >= 3:
                target_x, target_y = action[1], action[2]
                # Move towards target (simplified - move up to 5 tiles)
                dx = target_x - self.player_x
                dy = target_y - self.player_y
                
                if abs(dx) > 5:
                    dx = 5 if dx > 0 else -5
                if abs(dy) > 5:
                    dy = 5 if dy > 0 else -5
                
                self.player_x += dx
                self.player_y += dy
                
                dist = self.get_distance()
                if dist < self.best_distance:
                    self.best_distance = dist
                
                # Print progress every 100 steps
                if self.total_steps % 100 == 0:
                    print(f"[Step {self.total_steps}] Pos: ({self.player_x}, {self.player_y}) | Distance: {dist:.1f} | Best: {self.best_distance:.1f}")
    
    def reset(self):
        self.reset_position()
        self.tick = 0
        self.episodes += 1
        if self.episodes % 10 == 0:
            print(f"[Episode {self.episodes}] Best distance so far: {self.best_distance:.1f}")
    
    def send_ticks(self, conn):
        while self.running:
            try:
                state = self.get_state()
                msg = json.dumps({"type": "state", "data": state})
                conn.sendall((msg + '\n').encode('utf-8'))
                self.tick += 1
                time.sleep(0.05)  # Faster for training (50ms instead of 600ms)
            except:
                break
    
    def get_state(self):
        return {
            "tick": self.tick,
            "playerX": self.player_x,
            "playerY": self.player_y,
            "playerPlane": 0,
            "playerHp": 99,
            "playerMaxHp": 99,
            "playerPrayer": 99,
            "playerMaxPrayer": 99,
            "playerEnergy": 100,
            "playerAnimation": -1,
            "playerIsMoving": False,
            "playerInCombat": False,
            "hasTarget": False,
            "targetHp": 0,
            "targetMaxHp": 0,
            "targetX": 0,
            "targetY": 0,
            "targetAnimation": -1,
            "inventoryIds": [-1] * 28,
            "inventoryQuantities": [0] * 28,
            "equipmentIds": [-1] * 11,
            "activePrayers": [False] * 29,
            "skillLevels": [99] * 23,
            "skillXp": [13034431] * 23,
            "nearbyNpcs": [],
            "nearbyPlayers": [],
            "nearbyObjects": []
        }


if __name__ == "__main__":
    server = MockOSRSServer(port=5555)
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n[MockServer] Shutting down")
        server.running = False
