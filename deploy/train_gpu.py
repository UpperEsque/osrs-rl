#!/usr/bin/env python3
"""
GPU-accelerated training for Vast.ai
Auto-detects CUDA and uses optimal settings
"""
import sys
sys.path.insert(0, 'python')

import torch
import os
from datetime import datetime
import argparse

# Check GPU
print("="*60)
print("HARDWARE CHECK")
print("="*60)
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    DEVICE = "cuda"
else:
    print("✗ No GPU found, using CPU")
    DEVICE = "cpu"
print()

from osrs_rl.bosses.vorkath import VorkathEnv
from osrs_rl.bosses.zulrah import ZulrahEnv
from osrs_rl.bosses.toa import WardensEnv, AkkhaEnv, KephriEnv, ZebakEnv, BaBaEnv
from osrs_rl.unified_policy import create_unified_ppo, AdvancedFrameStack
from osrs_rl.lstm_policy import FrameStackWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

ENVS = {
    "vorkath": VorkathEnv,
    "zulrah": ZulrahEnv,
    "wardens": WardensEnv,
    "akkha": AkkhaEnv,
    "kephri": KephriEnv,
    "zebak": ZebakEnv,
    "baba": BaBaEnv,
}


def make_env(EnvClass, n_frames, rank, seed=0):
    def _init():
        env = EnvClass()
        env = AdvancedFrameStack(env, n_frames=n_frames)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train(
    boss: str = "vorkath",
    mode: str = "unified",  # "unified", "frames", "simple"
    n_frames: int = 8,
    use_world_model: bool = True,
    timesteps: int = 2_000_000,
    n_envs: int = 16,  # More envs for GPU
):
    EnvClass = ENVS[boss]
    
    run_name = f"{boss}_{mode}_gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = f"models/{run_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    test_env = EnvClass()
    obs_dim = test_env.observation_space.shape[0]
    action_dim = test_env.action_space.n
    test_env.close()
    
    print("="*60)
    print(f"TRAINING: {boss.upper()} ({mode})")
    print("="*60)
    print(f"  Device: {DEVICE}")
    print(f"  Mode: {mode}")
    print(f"  Frames: {n_frames}")
    print(f"  World Model: {use_world_model}")
    print(f"  Envs: {n_envs}")
    print(f"  Steps: {timesteps:,}")
    print(f"  Save: {save_dir}")
    print("="*60 + "\n")
    
    # Create environments
    env = SubprocVecEnv([make_env(EnvClass, n_frames, i) for i in range(n_envs)])
    eval_env = SubprocVecEnv([make_env(EnvClass, n_frames, 100)])
    
    callbacks = [
        CheckpointCallback(save_freq=max(50000 // n_envs, 500), save_path=save_dir, name_prefix="model"),
        EvalCallback(eval_env, best_model_save_path=save_dir, eval_freq=max(25000 // n_envs, 250), n_eval_episodes=20),
    ]
    
    if mode == "unified":
        model = create_unified_ppo(
            env,
            n_frames=n_frames,
            use_world_model=use_world_model,
            imagination_horizon=3,
            device=DEVICE,
            tensorboard_log=f"{save_dir}/tb",
        )
    else:
        # Simple frame stacking with larger network
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.015,
            verbose=1,
            device=DEVICE,
            tensorboard_log=f"{save_dir}/tb",
            policy_kwargs={"net_arch": [512, 256, 256]},
        )
    
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Device: {next(model.policy.parameters()).device}\n")
    
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)
    model.save(f"{save_dir}/final_model")
    
    print(f"\n✓ Training complete! Saved to {save_dir}")
    
    env.close()
    eval_env.close()
    
    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--boss", type=str, default="vorkath", choices=list(ENVS.keys()))
    parser.add_argument("--mode", type=str, default="unified", choices=["unified", "frames", "simple"])
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--no-world-model", action="store_true")
    args = parser.parse_args()
    
    train(
        boss=args.boss,
        mode=args.mode,
        n_frames=args.frames,
        use_world_model=not args.no_world_model,
        timesteps=args.timesteps,
        n_envs=args.envs,
    )
