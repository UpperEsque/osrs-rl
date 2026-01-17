#!/usr/bin/env python3
"""GPU-accelerated training for Vast.ai"""
import sys
sys.path.insert(0, '.')

import torch
import os
from datetime import datetime
import argparse

print("="*60)
print("HARDWARE CHECK")
print("="*60)
if torch.cuda.is_available():
    print(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    DEVICE = "cuda"
else:
    print("✗ No GPU, using CPU")
    DEVICE = "cpu"
print()

from osrs_rl.bosses.vorkath import VorkathEnv
from osrs_rl.bosses.zulrah import ZulrahEnv
from osrs_rl.bosses.toa import WardensEnv, AkkhaEnv
from osrs_rl.unified_policy import create_unified_ppo, AdvancedFrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
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
        env = AdvancedFrameStack(env, n_frames=n_frames)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def train(boss="vorkath", n_frames=8, use_world_model=True, timesteps=2_000_000, n_envs=16):
    EnvClass = ENVS[boss]
    
    run_name = f"{boss}_unified_gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = f"models/{run_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*60)
    print(f"TRAINING: {boss.upper()} (Unified Architecture)")
    print("="*60)
    print(f"  Device: {DEVICE}")
    print(f"  Frames: {n_frames}")
    print(f"  World Model: {use_world_model}")
    print(f"  Envs: {n_envs}")
    print(f"  Steps: {timesteps:,}")
    print("="*60 + "\n")
    
    env = SubprocVecEnv([make_env(EnvClass, n_frames, i) for i in range(n_envs)])
    eval_env = SubprocVecEnv([make_env(EnvClass, n_frames, 100)])
    
    callbacks = [
        CheckpointCallback(save_freq=50000//n_envs, save_path=save_dir, name_prefix="model"),
        EvalCallback(eval_env, best_model_save_path=save_dir, eval_freq=25000//n_envs, n_eval_episodes=20),
    ]
    
    model = create_unified_ppo(
        env,
        n_frames=n_frames,
        use_world_model=use_world_model,
        imagination_horizon=3,
        device=DEVICE,
        tensorboard_log=f"{save_dir}/tb",
    )
    
    print(f"Parameters: {sum(p.numel() for p in model.policy.parameters()):,}\n")
    
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)
    model.save(f"{save_dir}/final_model")
    
    print(f"\n✓ Saved to {save_dir}")
    env.close()
    eval_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--boss", default="vorkath", choices=list(ENVS.keys()))
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--no-world-model", action="store_true")
    args = parser.parse_args()
    
    train(args.boss, args.frames, not args.no_world_model, args.timesteps, args.envs)
