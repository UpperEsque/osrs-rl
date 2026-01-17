#!/usr/bin/env python3
"""
Train with Frame Stacking for sequence memory.

Best for:
- Zulrah (rotation memory)
- Vorkath (attack pattern memory)
- Akkha (attack sequence memory)
"""
import sys
sys.path.insert(0, 'python')

from osrs_rl.bosses.zulrah import ZulrahEnv
from osrs_rl.bosses.vorkath import VorkathEnv
from osrs_rl.bosses.toa import WardensEnv, AkkhaEnv
from osrs_rl.lstm_policy import FrameStackWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
from datetime import datetime
import argparse


ENVS = {
    "zulrah": ZulrahEnv,
    "vorkath": VorkathEnv,
    "wardens": WardensEnv,
    "akkha": AkkhaEnv,
}


def make_stacked_env(EnvClass, n_frames, rank, seed=0):
    """Create a frame-stacked environment for SubprocVecEnv"""
    def _init():
        env = EnvClass()
        env = FrameStackWrapper(env, n_frames=n_frames)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_with_frames(
    boss: str = "vorkath",
    n_frames: int = 4,
    timesteps: int = 2_000_000,
    n_envs: int = 8,
):
    """Train with frame stacking for temporal memory"""
    
    EnvClass = ENVS[boss]
    
    run_name = f"{boss}_lstm_f{n_frames}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = f"models/{run_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Get observation dimensions
    test_env = EnvClass()
    orig_obs_dim = test_env.observation_space.shape[0]
    stacked_obs_dim = orig_obs_dim * n_frames
    test_env.close()
    
    print("="*60)
    print(f"TRAINING WITH FRAME STACKING (LSTM-like)")
    print("="*60)
    print(f"  Boss: {boss}")
    print(f"  Frames stacked: {n_frames}")
    print(f"  Observation: {orig_obs_dim} → {stacked_obs_dim}")
    print(f"  Envs: {n_envs} | Steps: {timesteps:,}")
    print(f"  Save: {save_dir}")
    print("="*60 + "\n")
    
    # Create parallel environments
    env = SubprocVecEnv([make_stacked_env(EnvClass, n_frames, i) for i in range(n_envs)])
    eval_env = SubprocVecEnv([make_stacked_env(EnvClass, n_frames, 100)])
    
    callbacks = [
        CheckpointCallback(
            save_freq=max(50000 // n_envs, 1000), 
            save_path=save_dir, 
            name_prefix="ppo"
        ),
        EvalCallback(
            eval_env, 
            best_model_save_path=save_dir, 
            eval_freq=max(25000 // n_envs, 500), 
            n_eval_episodes=20
        ),
    ]
    
    # Larger network for stacked observations
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.015,  # Slightly higher entropy for exploration
        verbose=1,
        tensorboard_log=f"{save_dir}/tb",
        policy_kwargs={
            "net_arch": [512, 256, 256],  # Bigger network for more input
        },
    )
    
    print(f"Network input: {stacked_obs_dim} → [512, 256, 256] → {env.action_space.n} actions")
    print(f"Parameters: {sum(p.numel() for p in model.policy.parameters()):,}\n")
    
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)
    model.save(f"{save_dir}/final_model")
    
    print(f"\n✓ Saved to {save_dir}")
    
    env.close()
    eval_env.close()
    
    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--boss", type=str, default="vorkath", choices=list(ENVS.keys()))
    parser.add_argument("--frames", type=int, default=4, help="Number of frames to stack (2-8)")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--envs", type=int, default=8)
    args = parser.parse_args()
    
    train_with_frames(args.boss, args.frames, args.timesteps, args.envs)
