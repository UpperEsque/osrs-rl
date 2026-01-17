"""
Base Gym environment for OSRS
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any

from osrs_rl.client import GameClient
from osrs_rl.protocol import GameState


class OSRSBaseEnv(gym.Env):
    """
    Base environment for OSRS RL.
    
    Subclasses should implement:
        - _get_observation(state) -> np.array
        - _compute_reward(state, prev_state) -> float
        - _check_done(state) -> bool
        - action_space definition
        - observation_space definition
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        max_episode_steps: int = 1000,
    ):
        super().__init__()
        
        self.client = GameClient(host, port)
        self.max_episode_steps = max_episode_steps
        
        self._current_state: Optional[GameState] = None
        self._prev_state: Optional[GameState] = None
        self._step_count = 0
        
        # Subclasses must define these
        self.observation_space: spaces.Space = None
        self.action_space: spaces.Space = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Connect if needed
        if not self.client.connected:
            if not self.client.connect():
                raise RuntimeError("Failed to connect to OSRS plugin")
        
        # Request reset from plugin
        self.client.send_reset()
        
        # Wait for initial state
        self._current_state = self.client.get_state(timeout=10.0)
        if self._current_state is None:
            raise RuntimeError("No state received after reset")
        
        self._prev_state = None
        self._step_count = 0
        
        obs = self._get_observation(self._current_state)
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action and return results"""
        
        # Convert action to plugin format
        action_data = self._action_to_plugin(action)
        self.client.send_action(action_data)
        
        # Wait for next state
        self._prev_state = self._current_state
        self._current_state = self.client.get_state(timeout=5.0)
        
        if self._current_state is None:
            # Connection lost
            return self._get_observation(self._prev_state), 0.0, True, True, {}
        
        self._step_count += 1
        
        # Compute outputs
        obs = self._get_observation(self._current_state)
        reward = self._compute_reward(self._current_state, self._prev_state)
        terminated = self._check_terminated(self._current_state)
        truncated = self._step_count >= self.max_episode_steps
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def close(self):
        """Clean up"""
        self.client.disconnect()
    
    # --- Methods to override in subclasses ---
    
    def _get_observation(self, state: GameState) -> np.ndarray:
        """Convert game state to observation array"""
        raise NotImplementedError
    
    def _compute_reward(self, state: GameState, prev_state: Optional[GameState]) -> float:
        """Compute reward for this step"""
        raise NotImplementedError
    
    def _check_terminated(self, state: GameState) -> bool:
        """Check if episode should end"""
        raise NotImplementedError
    
    def _action_to_plugin(self, action: int) -> list:
        """Convert discrete action to plugin format [type, p1, p2, p3]"""
        raise NotImplementedError
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info dict"""
        return {
            'tick': self._current_state.tick if self._current_state else 0,
            'step': self._step_count,
        }
