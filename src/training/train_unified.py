#!/usr/bin/env python3
"""
Train with Unified Architecture: Attention + GRU + World Model

This combines the best of all approaches:
1. Attention - Focus on relevant past frames
2. GRU - Maintain memory across episodes  
3. World Model - Imagine future states for planning

Usage:
    python train_unified.py --boss vorkath --timesteps 2000000
    python train_unified.py --boss zulrah --no-world-model  # Faster, still good
    python train_unified.py --boss akkha --frames 12  # More memory for Akkha
"""
import sys
sys.path.insert(0, 'python')

from osrs_rl.bosses.vorkath import VorkathEnv
from osrs_rl.bosses.zulrah import ZulrahEnv
from osrs_rl.bosses.toa import WardensEnv, AkkhaEnv, KephriEnv, ZebakEnv, BaBaEnv
from osrs_rl.unified_policy import create_unified_ppo, AdvancedFrameStack
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
from datetime import datetime
import argparse


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


def train_unified(
    boss: str = "vorkath",
    n_frames: int = 8,
    use_world_model: bool = True,
    imagination_horizon: int = 3,
    timesteps: int = 2_000_000,
    n_envs: int = 8,
):
    EnvClass = ENVS[boss]
    
    # Create descriptive run name
    components = ["unified"]
    components.append(f"f{n_frames}")
    if use_world_model:
        components.append(f"wm{imagination_horizon}")
    else:
        components.append("nowm")
    
    run_name = f"{boss}_{'_'.join(components)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = f"models/{run_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Get dimensions
    test_env = EnvClass()
    obs_dim = test_env.observation_space.shape[0]
    action_dim = test_env.action_space.n
    test_env.close()
    
    print("="*70)
    print("UNIFIED ARCHITECTURE: Attention + GRU + World Model")
    print("="*70)
    print(f"  Boss: {boss}")
    print(f"  Frames: {n_frames} (temporal window)")
    print(f"  Observation: {obs_dim} → {obs_dim * n_frames} (stacked)")
    print(f"  Actions: {action_dim}")
    print(f"  World Model: {'YES (horizon=' + str(imagination_horizon) + ')' if use_world_model else 'NO'}")
    print(f"  Envs: {n_envs} | Steps: {timesteps:,}")
    print(f"  Save: {save_dir}")
    print("="*70)
    print("\nArchitecture:")
    print("  [Frame Stack] → [Attention] → [GRU Memory]")
    if use_world_model:
        print("                       ↑")
        print("              [World Model Imagination]")
    print()
    
    # Create environments
    env = SubprocVecEnv([make_env(EnvClass, n_frames, i) for i in range(n_envs)])
    eval_env = SubprocVecEnv([make_env(EnvClass, n_frames, 100)])
    
    callbacks = [
        CheckpointCallback(
            save_freq=max(50000 // n_envs, 1000),
            save_path=save_dir,
            name_prefix="unified"
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=save_dir,
            eval_freq=max(25000 // n_envs, 500),
            n_eval_episodes=20
        ),
    ]
    
    # Create model
    model = create_unified_ppo(
        env,
        n_frames=n_frames,
        use_world_model=use_world_model,
        imagination_horizon=imagination_horizon,
        tensorboard_log=f"{save_dir}/tb",
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Total parameters: {total_params:,}")
    print()
    
    # Train
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)
    model.save(f"{save_dir}/final_model")
    
    print(f"\n✓ Saved to {save_dir}")
    
    env.close()
    eval_env.close()
    
    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with Unified Architecture")
    parser.add_argument("--boss", type=str, default="vorkath", choices=list(ENVS.keys()))
    parser.add_argument("--frames", type=int, default=8, help="Number of frames to stack")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--no-world-model", action="store_true", help="Disable world model")
    parser.add_argument("--imagination", type=int, default=3, help="World model imagination horizon")
    args = parser.parse_args()
    
    train_unified(
        boss=args.boss,
        n_frames=args.frames,
        use_world_model=not args.no_world_model,
        imagination_horizon=args.imagination,
        timesteps=args.timesteps,
        n_envs=args.envs,
    )
