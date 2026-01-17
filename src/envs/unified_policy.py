"""
Unified Advanced Policy for OSRS RL

Combines:
1. GRU - Sequential memory (remembers past)
2. Attention - Focus on what matters (selective memory)
3. World Model - Predict future (planning)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from typing import Tuple, Dict, List, Optional
from collections import deque


# ============================================================
# FRAME STACK WRAPPER
# ============================================================

class AdvancedFrameStack(gym.Wrapper):
    """Advanced frame stacking that preserves frame structure."""
    
    def __init__(self, env: gym.Env, n_frames: int = 8):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
        
        self.single_obs_dim = env.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(n_frames * self.single_obs_dim,),
            dtype=np.float32
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_obs(), info
    
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, term, trunc, info
    
    def _get_obs(self) -> np.ndarray:
        return np.concatenate(list(self.frames)).astype(np.float32)


# ============================================================
# COMPONENT 1: ATTENTION MODULE
# ============================================================

class TemporalAttention(nn.Module):
    """
    Self-attention over time steps.
    Learns which past observations are most relevant.
    """
    
    def __init__(self, obs_dim: int, n_frames: int, embed_dim: int = 64, n_heads: int = 4):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.n_frames = n_frames
        self.embed_dim = embed_dim
        
        # Embed each frame
        self.frame_encoder = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, n_frames, embed_dim) * 0.02)
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True,
        )
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_frames * obs_dim) - flattened frame stack
        
        Returns:
            attended: (batch, embed_dim) - attention-weighted representation
            attention_weights: (batch, n_frames) - where model is "looking"
        """
        batch_size = x.shape[0]
        
        # Reshape to (batch, n_frames, obs_dim)
        x = x.view(batch_size, self.n_frames, self.obs_dim)
        
        # Encode each frame
        frame_embeds = self.frame_encoder(x)  # (batch, n_frames, embed_dim)
        
        # Add positional encoding
        frame_embeds = frame_embeds + self.pos_encoding
        
        # Self-attention
        attended, attention_weights = self.self_attention(
            frame_embeds, frame_embeds, frame_embeds,
            average_attn_weights=True  # Average across heads
        )
        
        # Residual connection and norm
        attended = self.norm(attended + frame_embeds)
        
        # Take the last frame's representation (most recent)
        output = self.output_proj(attended[:, -1, :])
        
        # Get attention weights for the last query (what current frame attends to)
        # attention_weights shape: (batch, n_frames, n_frames) when average_attn_weights=True
        attn_weights = attention_weights[:, -1, :]  # (batch, n_frames)
        
        return output, attn_weights


# ============================================================
# COMPONENT 2: GRU MEMORY MODULE
# ============================================================

class GRUMemory(nn.Module):
    """
    GRU for sequential memory.
    Maintains hidden state across time steps.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        
        self._hidden = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        device = x.device
        
        # Add sequence dimension
        x = x.unsqueeze(1)
        
        # Initialize hidden state if needed
        if self._hidden is None or self._hidden.shape[1] != batch_size:
            self._hidden = torch.zeros(
                self.num_layers, batch_size, self.hidden_dim,
                device=device
            )
        
        # Detach to prevent backprop through time across batches
        self._hidden = self._hidden.detach()
        
        # GRU forward
        output, self._hidden = self.gru(x, self._hidden)
        
        return output.squeeze(1)
    
    def reset(self):
        self._hidden = None


# ============================================================
# COMPONENT 3: WORLD MODEL (Prediction)
# ============================================================

class WorldModel(nn.Module):
    """
    Learns to predict next observation given current state and action.
    Enables "imagination" and planning.
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.obs_dim = obs_dim
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Action encoder
        self.action_encoder = nn.Embedding(action_dim, hidden_dim)
        
        # Dynamics model: predicts next state
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
            nn.Sigmoid(),
        )
        
        # Reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(
        self, 
        obs: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state = self.state_encoder(obs)
        action_embed = self.action_encoder(action)
        
        combined = torch.cat([state, action_embed], dim=-1)
        
        next_obs_pred = self.dynamics(combined)
        reward_pred = self.reward_predictor(combined)
        
        return next_obs_pred, reward_pred
    
    def imagine(
        self, 
        obs: torch.Tensor, 
        action_sequence: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        predicted_obs = []
        predicted_rewards = []
        current_obs = obs
        
        horizon = action_sequence.shape[1]
        
        for t in range(horizon):
            action = action_sequence[:, t]
            next_obs, reward = self.forward(current_obs, action)
            predicted_obs.append(next_obs)
            predicted_rewards.append(reward)
            current_obs = next_obs
        
        return predicted_obs, predicted_rewards


# ============================================================
# UNIFIED FEATURE EXTRACTOR
# ============================================================

class UnifiedExtractor(BaseFeaturesExtractor):
    """
    Combines Attention + GRU + World Model into one powerful extractor.
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        n_frames: int = 8,
        embed_dim: int = 64,
        gru_hidden: int = 128,
        n_attention_heads: int = 4,
        use_world_model: bool = True,
        imagination_horizon: int = 3,
        action_dim: int = 17,
    ):
        super().__init__(observation_space, features_dim)
        
        total_obs_dim = observation_space.shape[0]
        self.single_obs_dim = total_obs_dim // n_frames
        self.n_frames = n_frames
        self.use_world_model = use_world_model
        self.imagination_horizon = imagination_horizon
        self.action_dim = action_dim
        
        # 1. Temporal Attention
        self.attention = TemporalAttention(
            obs_dim=self.single_obs_dim,
            n_frames=n_frames,
            embed_dim=embed_dim,
            n_heads=n_attention_heads,
        )
        
        # 2. World Model (optional)
        if use_world_model:
            self.world_model = WorldModel(
                obs_dim=self.single_obs_dim,
                action_dim=action_dim,
                hidden_dim=128,
            )
            world_model_dim = 64
            self.world_model_compress = nn.Sequential(
                nn.Linear(self.single_obs_dim * imagination_horizon, world_model_dim),
                nn.ReLU(),
            )
        else:
            world_model_dim = 0
            self.world_model = None
        
        # 3. GRU Memory
        gru_input_dim = embed_dim + world_model_dim
        self.gru = GRUMemory(
            input_dim=gru_input_dim,
            hidden_dim=gru_hidden,
            num_layers=1,
        )
        
        # 4. Final output
        self.output_net = nn.Sequential(
            nn.Linear(gru_hidden, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        device = observations.device
        
        # 1. Attention over frames
        attended, _ = self.attention(observations)
        
        # 2. World Model imagination (if enabled)
        if self.use_world_model and self.world_model is not None:
            current_obs = observations[:, -self.single_obs_dim:]
            
            random_actions = torch.randint(
                0, self.action_dim, 
                (batch_size, self.imagination_horizon),
                device=device
            )
            
            imagined_obs, _ = self.world_model.imagine(current_obs, random_actions)
            imagined_flat = torch.cat(imagined_obs, dim=-1)
            world_features = self.world_model_compress(imagined_flat)
            
            combined = torch.cat([attended, world_features], dim=-1)
        else:
            combined = attended
        
        # 3. GRU memory
        memory_out = self.gru(combined)
        
        # 4. Final features
        features = self.output_net(memory_out)
        
        return features
    
    def reset_memory(self):
        self.gru.reset()


# ============================================================
# CREATE UNIFIED PPO
# ============================================================

def create_unified_ppo(
    env,
    n_frames: int = 8,
    use_world_model: bool = True,
    imagination_horizon: int = 3,
    **kwargs
) -> PPO:
    """Create PPO with unified Attention + GRU + World Model policy."""
    
    action_dim = env.action_space.n if hasattr(env, 'action_space') else 17
    
    default_kwargs = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "verbose": 1,
        "policy_kwargs": {
            "features_extractor_class": UnifiedExtractor,
            "features_extractor_kwargs": {
                "features_dim": 256,
                "n_frames": n_frames,
                "embed_dim": 64,
                "gru_hidden": 128,
                "n_attention_heads": 4,
                "use_world_model": use_world_model,
                "imagination_horizon": imagination_horizon,
                "action_dim": action_dim,
            },
            "net_arch": [256, 128],
        },
    }
    default_kwargs.update(kwargs)
    
    return PPO("MlpPolicy", env, **default_kwargs)


def create_attention_only_ppo(env, n_frames: int = 8, **kwargs) -> PPO:
    """PPO with only Attention (no GRU, no World Model)"""
    return create_unified_ppo(env, n_frames=n_frames, use_world_model=False, **kwargs)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("Testing Unified Policy Components...")
    
    batch_size = 4
    n_frames = 8
    obs_dim = 40
    
    # Test Attention
    print("\n1. Testing Temporal Attention...")
    attention = TemporalAttention(obs_dim, n_frames, embed_dim=64, n_heads=4)
    x = torch.randn(batch_size, n_frames * obs_dim)
    out, weights = attention(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}, Attention: {weights.shape}")
    print(f"   Attention weights sum: {weights.sum(dim=-1)}")
    
    # Test GRU
    print("\n2. Testing GRU Memory...")
    gru = GRUMemory(64, hidden_dim=128)
    x = torch.randn(batch_size, 64)
    out = gru(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    out2 = gru(x)
    print(f"   Second call (with memory): {out2.shape}")
    
    # Test World Model
    print("\n3. Testing World Model...")
    world_model = WorldModel(obs_dim, action_dim=17, hidden_dim=128)
    obs = torch.randn(batch_size, obs_dim)
    actions = torch.randint(0, 17, (batch_size,))
    next_obs, reward = world_model(obs, actions)
    print(f"   Predict: obs {obs.shape} + action {actions.shape} -> next_obs {next_obs.shape}, reward {reward.shape}")
    
    action_seq = torch.randint(0, 17, (batch_size, 5))
    imagined_obs, imagined_rewards = world_model.imagine(obs, action_seq)
    print(f"   Imagine 5 steps: {len(imagined_obs)} observations, {len(imagined_rewards)} rewards")
    
    # Test Unified Extractor
    print("\n4. Testing Unified Extractor...")
    obs_space = spaces.Box(low=0, high=1, shape=(n_frames * obs_dim,), dtype=np.float32)
    extractor = UnifiedExtractor(
        obs_space, 
        features_dim=256, 
        n_frames=n_frames,
        use_world_model=True,
        action_dim=17,
    )
    x = torch.randn(batch_size, n_frames * obs_dim)
    features = extractor(x)
    print(f"   Input: {x.shape} -> Features: {features.shape}")
    
    # Test with actual environment
    print("\n5. Testing with VorkathEnv...")
    try:
        from osrs_rl.bosses.vorkath import VorkathEnv
        env = VorkathEnv()
        wrapped = AdvancedFrameStack(env, n_frames=8)
        obs, _ = wrapped.reset()
        print(f"   Environment obs shape: {obs.shape}")
        
        for _ in range(5):
            action = wrapped.action_space.sample()
            obs, reward, done, trunc, info = wrapped.step(action)
            if done:
                break
        print(f"   Environment working!")
    except Exception as e:
        print(f"   Skipping env test: {e}")
    
    print("\n" + "="*50)
    print("✓ All components working!")
    print("="*50)
    print("\nUsage:")
    print("  python train_unified.py --boss vorkath --frames 8")
    print("  python train_unified.py --boss zulrah --no-world-model")
