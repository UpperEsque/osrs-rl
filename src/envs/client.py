"""
Client for connecting to OSRS RL plugin
"""
import socket
import json
import threading
from typing import Optional, List, Callable
from queue import Queue, Empty

from .protocol import GameState, Actions


class GameClient:
    """
    Connects to the RuneLite plugin via TCP socket.
    
    Usage:
        client = GameClient(port=5555)
        client.connect()
        
        while True:
            state = client.get_state()  # blocks until state received
            action = agent.predict(state)
            client.send_action(action)
    """
    
    def __init__(self, host: str = "localhost", port: int = 5555):
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.connected = False
        
        self._state_queue: Queue[GameState] = Queue(maxsize=1)
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False
        
    def connect(self, timeout: float = 30.0) -> bool:
        """Connect to the plugin"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(timeout)
            self.socket.connect((self.host, self.port))
            self.socket.settimeout(None)
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            self.connected = True
            self._running = True
            
            # Start reader thread
            self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._reader_thread.start()
            
            print(f"[Client] Connected to {self.host}:{self.port}")
            return True
            
        except Exception as e:
            print(f"[Client] Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from plugin"""
        self._running = False
        self.connected = False
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        
        if self._reader_thread:
            self._reader_thread.join(timeout=1.0)
    
    def _read_loop(self):
        """Background thread reading state updates"""
        buffer = ""
        
        while self._running and self.connected:
            try:
                data = self.socket.recv(4096).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                
                # Process complete messages (newline delimited)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        self._handle_message(line)
                        
            except Exception as e:
                if self._running:
                    print(f"[Client] Read error: {e}")
                break
        
        self.connected = False
        print("[Client] Disconnected")
    
    def _handle_message(self, line: str):
        """Process a message from the plugin"""
        try:
            msg = json.loads(line)
            msg_type = msg.get('type')
            
            if msg_type == 'state':
                state = GameState.from_dict(msg.get('data', {}))
                
                # Replace old state (we only care about latest)
                try:
                    self._state_queue.get_nowait()
                except Empty:
                    pass
                self._state_queue.put(state)
                
        except json.JSONDecodeError as e:
            print(f"[Client] JSON error: {e}")
    
    def get_state(self, timeout: float = 5.0) -> Optional[GameState]:
        """
        Get the latest game state.
        Blocks until state is received or timeout.
        """
        try:
            return self._state_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def send_action(self, action: List[int]):
        """
        Send an action to the plugin.
        
        Args:
            action: [action_type, param1, param2, param3]
        """
        if not self.connected or not self.socket:
            return
        
        try:
            msg = json.dumps({"type": "action", "data": action})
            self.socket.sendall((msg + '\n').encode('utf-8'))
        except Exception as e:
            print(f"[Client] Send error: {e}")
            self.connected = False
    
    def send_reset(self):
        """Request environment reset"""
        if not self.connected or not self.socket:
            return
            
        try:
            msg = json.dumps({"type": "reset"})
            self.socket.sendall((msg + '\n').encode('utf-8'))
        except Exception as e:
            print(f"[Client] Send error: {e}")
    
    # Convenience methods for common actions
    def noop(self):
        self.send_action([Actions.NOOP])
    
    def walk_to(self, x: int, y: int):
        self.send_action([Actions.WALK_TO, x, y])
    
    def click_inventory(self, slot: int):
        self.send_action([Actions.CLICK_INVENTORY, slot])
    
    def attack_npc(self, npc_index: int):
        self.send_action([Actions.ATTACK_NPC, npc_index])
    
    def interact_object(self, object_id: int, action_index: int = 0):
        self.send_action([Actions.INTERACT_OBJECT, object_id, action_index])
    
    def toggle_prayer(self, prayer_id: int):
        self.send_action([Actions.TOGGLE_PRAYER, prayer_id])
    
    def toggle_run(self):
        self.send_action([Actions.TOGGLE_RUN])
    
    def special_attack(self):
        self.send_action([Actions.SPECIAL_ATTACK])
