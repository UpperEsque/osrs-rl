#!/usr/bin/env python3
"""Transfer learning - handles different observation/action spaces"""
import sys
sys.path.insert(0, 'python')

from osrs_rl.pve_env import OSRSPvEEnv, MONSTERS
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import glob
import os
from datetime import datetime
import argparse

def make_env(monster, rank, seed=0):
    def _init():
        env = OSRSPvEEnv(monster=monster)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def train_fresh(monster, timesteps=500_000, n_envs=8):
    """Train a fresh model on target monster (when transfer not possible)"""
    
    run_name = f"{monster}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = f"models/{run_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*60)
    print(f"FRESH TRAINING: {monster}")
    print("="*60)
    print(f"Envs: {n_envs} | Timesteps: {timesteps:,}")
    print(f"Save: {save_dir}\n")
    
    env = SubprocVecEnv([make_env(monster, i) for i in range(n_envs)])
    eval_env = SubprocVecEnv([make_env(monster, 100)])
    
    callbacks = [
        CheckpointCallback(save_freq=25000//n_envs, save_path=save_dir, name_prefix="ppo"),
        EvalCallback(eval_env, best_model_save_path=save_dir, eval_freq=10000//n_envs, n_eval_episodes=20),
    ]
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.02,
        verbose=1,
        tensorboard_log=f"{save_dir}/tb",
        policy_kwargs={"net_arch": [256, 256, 128]},
    )
    
    print(f"Parameters: {sum(p.numel() for p in model.policy.parameters()):,}\n")
    
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)
    model.save(f"{save_dir}/final_model")
    
    print(f"\n✓ Saved to {save_dir}")
    
    env.close()
    eval_env.close()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="zulrah", choices=list(MONSTERS.keys()))
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--envs", type=int, default=8)
    args = parser.parse_args()
    
    print(f"Note: Transfer learning requires matching observation spaces.")
    print(f"Training fresh model on {args.target}...\n")
    
    train_fresh(args.target, args.timesteps, args.envs)
