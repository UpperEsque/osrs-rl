#!/usr/bin/env python3
"""
Frame Stack Training - The simple but effective approach (88% kill rate)
No LSTM, no Attention, no World Model - just stack frames and learn!
"""
import sys
import torch
import os
from datetime import datetime
import argparse

print("="*60)
print("FRAME STACK TRAINING (Simple & Effective)")
print("="*60)
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    DEVICE = "cuda"
else:
    print("⚠ No GPU, using CPU")
    DEVICE = "cpu"
print("="*60 + "\n")

from osrs_rl.bosses.vorkath import VorkathEnv
from osrs_rl.bosses.zulrah import ZulrahEnv
from osrs_rl.bosses.toa import WardensEnv, AkkhaEnv
from osrs_rl.lstm_policy import FrameStackWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

ENVS = {
    "vorkath": VorkathEnv,
    "zulrah": ZulrahEnv,
    "wardens": WardensEnv,
    "akkha": AkkhaEnv,
}

def make_env(EnvClass, n_frames, rank, seed=0):
    def _init():
        env = EnvClass()
        env = FrameStackWrapper(env, n_frames=n_frames)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def train(
    boss="vorkath",
    n_frames=4,
    timesteps=5_000_000,
    n_envs=8,
):
    EnvClass = ENVS[boss]
    
    run_name = f"{boss}_framestack_f{n_frames}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = f"models/{run_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Get env info
    test_env = EnvClass()
    obs_dim = test_env.observation_space.shape[0]
    action_dim = test_env.action_space.n
    test_env.close()
    
    print(f"Boss: {boss.upper()}")
    print(f"Frames: {n_frames} ({obs_dim} → {obs_dim * n_frames})")
    print(f"Actions: {action_dim}")
    print(f"Device: {DEVICE}")
    print(f"Envs: {n_envs}")
    print(f"Steps: {timesteps:,}")
    print(f"Save: {save_dir}")
    print()
    
    # Create environments
    env = DummyVecEnv([make_env(EnvClass, n_frames, i) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_env(EnvClass, n_frames, 100)])
    
    callbacks = [
        CheckpointCallback(
            save_freq=max(100000 // n_envs, 1000),
            save_path=save_dir,
            name_prefix="model"
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=save_dir,
            eval_freq=max(50000 // n_envs, 500),
            n_eval_episodes=30
        ),
    ]
    
    # Simple but effective architecture
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
        ent_coef=0.015,
        verbose=1,
        device=DEVICE,
        tensorboard_log=f"{save_dir}/tb",
        policy_kwargs={
            "net_arch": [512, 256, 256],  # Bigger network for more input
        },
    )
    
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Network: {obs_dim * n_frames} → [512, 256, 256] → {action_dim}\n")
    
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)
    model.save(f"{save_dir}/final_model")
    
    print(f"\n✓ Training complete! Saved to {save_dir}")
    
    env.close()
    eval_env.close()
    
    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--boss", type=str, default="vorkath", choices=list(ENVS.keys()))
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--envs", type=int, default=8)
    args = parser.parse_args()
    
    train(args.boss, args.frames, args.timesteps, args.envs)
