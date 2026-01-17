"""
LSTM Policy and Frame Stacking for OSRS RL

Benefits:
- Remembers attack patterns (Zulrah rotations, Akkha memory)
- Tracks phase sequences
- Better at predicting next attacks
"""
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple
from collections import deque


class FrameStackWrapper(gym.Wrapper):
    """
    Stack multiple frames to give the policy temporal information.
    Properly inherits from gymnasium.Wrapper for compatibility.
    """
    
    def __init__(self, env: gym.Env, n_frames: int = 4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
        
        # Update observation space
        obs_shape = env.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(obs_shape * n_frames,), 
            dtype=np.float32
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Fill frame buffer with initial observation
        for _ in range(self.n_frames):
            self.frames.append(obs)
        
        return self._get_stacked_obs(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_stacked_obs(), reward, terminated, truncated, info
    
    def _get_stacked_obs(self) -> np.ndarray:
        return np.concatenate(list(self.frames)).astype(np.float32)


class LSTMExtractor(BaseFeaturesExtractor):
    """
    LSTM-based feature extractor for sequential decision making.
    """
    
    def __init__(
        self, 
        observation_space: spaces.Box,
        features_dim: int = 256,
        lstm_hidden_dim: int = 128,
        num_lstm_layers: int = 1,
        mlp_hidden_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim)
        
        obs_dim = observation_space.shape[0]
        
        self.input_mlp = nn.Sequential(
            nn.Linear(obs_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
        )
        
        self.lstm = nn.LSTM(
            input_size=mlp_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
        )
        
        self.output_mlp = nn.Sequential(
            nn.Linear(lstm_hidden_dim, features_dim),
            nn.ReLU(),
        )
        
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self._hidden = None
        
    def _init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_dim, device=device)
        c = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_dim, device=device)
        return (h, c)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        device = observations.device
        
        x = self.input_mlp(observations)
        x = x.unsqueeze(1)
        
        if self._hidden is None or self._hidden[0].shape[1] != batch_size:
            self._hidden = self._init_hidden(batch_size, device)
        
        # Detach hidden state to prevent backprop through time across batches
        self._hidden = (self._hidden[0].detach(), self._hidden[1].detach())
        
        lstm_out, self._hidden = self.lstm(x, self._hidden)
        lstm_out = lstm_out.squeeze(1)
        features = self.output_mlp(lstm_out)
        
        return features
    
    def reset_hidden(self):
        self._hidden = None


def create_lstm_ppo(env, **kwargs) -> PPO:
    """Create a PPO model with LSTM policy."""
    default_kwargs = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "verbose": 1,
        "policy_kwargs": {
            "features_extractor_class": LSTMExtractor,
            "features_extractor_kwargs": {
                "features_dim": 256,
                "lstm_hidden_dim": 128,
            },
        },
    }
    default_kwargs.update(kwargs)
    
    return PPO("MlpPolicy", env, **default_kwargs)


if __name__ == "__main__":
    print("Testing Frame Stacking...")
    
    from osrs_rl.bosses.vorkath import VorkathEnv
    
    env = VorkathEnv()
    stacked = FrameStackWrapper(env, n_frames=4)
    
    obs, _ = stacked.reset()
    print(f"Original: {env.observation_space.shape} -> Stacked: {obs.shape}")
    
    for _ in range(5):
        action = stacked.action_space.sample()
        obs, reward, done, trunc, info = stacked.step(action)
        if done:
            break
    
    print("✓ Frame stacking working!")
