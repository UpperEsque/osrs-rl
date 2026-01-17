"""
Walking Environment - Real Game Version
Learn to walk to target locations
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
import time

import sys
sys.path.insert(0, '..')
from osrs_rl.client import GameClient
from osrs_rl.protocol import GameState, Actions


class WalkingEnv(gym.Env):
    """
    Walking environment for OSRS.
    
    Observation (6 values):
        - player_x (normalized relative to start)
        - player_y (normalized relative to start)
        - target_dx (relative, normalized)
        - target_dy (relative, normalized)
        - is_moving (0 or 1)
        - distance_to_target (normalized)
    
    Actions (9):
        0: NOOP (wait)
        1-8: Walk in 8 directions (N, S, E, W, NE, NW, SE, SW)
    
    Reward:
        - Negative distance to target (encourages getting closer)
        - +10 bonus for reaching target
        - Small time penalty
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        max_episode_steps: int = 100,
        target_distance: int = 10,
        reach_threshold: int = 2,
    ):
        super().__init__()
        
        self.host = host
        self.port = port
        self.max_episode_steps = max_episode_steps
        self.target_distance = target_distance
        self.reach_threshold = reach_threshold
        
        self.client: Optional[GameClient] = None
        
        # State tracking
        self._current_state: Optional[GameState] = None
        self._prev_state: Optional[GameState] = None
        self._step_count = 0
        self._start_x = 0
        self._start_y = 0
        self._target_x = 0
        self._target_y = 0
        self._prev_distance = 0
        
        # 6 observation values
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        
        # 9 actions: NOOP + 8 directions
        self.action_space = spaces.Discrete(9)
        
        # Direction vectors for actions 1-8
        self._directions = [
            (0, 0),    # 0: NOOP
            (0, 5),    # 1: North (+Y)
            (0, -5),   # 2: South (-Y)
            (5, 0),    # 3: East (+X)
            (-5, 0),   # 4: West (-X)
            (5, 5),    # 5: Northeast
            (-5, 5),   # 6: Northwest
            (5, -5),   # 7: Southeast
            (-5, -5),  # 8: Southwest
        ]
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment with a new random target"""
        super().reset(seed=seed)
        
        # Connect if needed
        if self.client is None or not self.client.connected:
            self.client = GameClient(self.host, self.port)
            if not self.client.connect(timeout=30):
                raise RuntimeError("Failed to connect to OSRS plugin")
        
        # Get current state
        self._current_state = self.client.get_state(timeout=5.0)
        if self._current_state is None:
            raise RuntimeError("No state received")
        
        # Set start position
        self._start_x = self._current_state.player_x
        self._start_y = self._current_state.player_y
        
        # Generate random target within range
        if self.np_random is not None:
            angle = self.np_random.uniform(0, 2 * np.pi)
        else:
            angle = np.random.uniform(0, 2 * np.pi)
        
        self._target_x = int(self._start_x + self.target_distance * np.cos(angle))
        self._target_y = int(self._start_y + self.target_distance * np.sin(angle))
        
        self._prev_state = None
        self._step_count = 0
        self._prev_distance = self._get_distance(self._current_state)
        
        print(f"[WalkingEnv] New episode: start=({self._start_x}, {self._start_y}), target=({self._target_x}, {self._target_y}), dist={self._prev_distance:.1f}")
        
        obs = self._get_observation(self._current_state)
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action and return results"""
        
        # Convert action to walk command
        if action > 0:
            dx, dy = self._directions[action]
            target_x = self._current_state.player_x + dx
            target_y = self._current_state.player_y + dy
            self.client.walk_to(target_x, target_y)
        
        # Wait for game tick
        time.sleep(0.6)
        
        # Get new state
        self._prev_state = self._current_state
        self._current_state = self.client.get_state(timeout=2.0)
        
        if self._current_state is None:
            # Connection lost
            return self._get_observation(self._prev_state), -10.0, True, True, {}
        
        self._step_count += 1
        
        # Compute outputs
        obs = self._get_observation(self._current_state)
        reward = self._compute_reward(self._current_state, self._prev_state)
        terminated = self._check_terminated(self._current_state)
        truncated = self._step_count >= self.max_episode_steps
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self, state: GameState) -> np.ndarray:
        """Convert game state to observation array"""
        # Relative position from start (normalized)
        px = (state.player_x - self._start_x) / 20.0
        py = (state.player_y - self._start_y) / 20.0
        
        # Relative target position (normalized)
        dx = (self._target_x - state.player_x) / 20.0
        dy = (self._target_y - state.player_y) / 20.0
        
        # Is moving
        moving = 1.0 if state.player_is_moving else 0.0
        
        # Distance to target (normalized)
        dist = self._get_distance(state) / 20.0
        
        return np.array([px, py, dx, dy, moving, dist], dtype=np.float32).clip(-1, 1)
    
    def _compute_reward(self, state: GameState, prev_state: Optional[GameState]) -> float:
        """Compute reward for this step"""
        distance = self._get_distance(state)
        
        # Reached target - big bonus!
        if distance <= self.reach_threshold:
            print(f"[WalkingEnv] Reached target! Distance={distance:.1f}")
            return 10.0
        
        # Reward for getting closer
        distance_delta = self._prev_distance - distance
        reward = distance_delta  # Positive if closer, negative if further
        
        # Small time penalty to encourage efficiency
        reward -= 0.05
        
        self._prev_distance = distance
        
        return reward
    
    def _check_terminated(self, state: GameState) -> bool:
        """Check if episode should end (reached target)"""
        return self._get_distance(state) <= self.reach_threshold
    
    def _get_distance(self, state: GameState) -> float:
        """Euclidean distance to target"""
        dx = state.player_x - self._target_x
        dy = state.player_y - self._target_y
        return np.sqrt(dx*dx + dy*dy)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info dict"""
        return {
            'tick': self._current_state.tick if self._current_state else 0,
            'step': self._step_count,
            'distance': self._get_distance(self._current_state) if self._current_state else 0,
            'target': (self._target_x, self._target_y),
            'position': (self._current_state.player_x, self._current_state.player_y) if self._current_state else (0, 0),
        }
    
    def close(self):
        """Clean up"""
        if self.client:
            self.client.disconnect()
