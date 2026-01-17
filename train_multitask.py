#!/usr/bin/env python3
"""Train on multiple bosses simultaneously for better generalization"""
import sys
sys.path.insert(0, 'python')

from osrs_rl.pve_env import OSRSPvEEnv, MONSTERS
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import numpy as np
import os
from datetime import datetime

def make_random_boss_env(bosses, rank, seed=0):
    """Create env that randomly picks a boss each episode"""
    def _init():
        # Randomly select boss for this env instance
        boss = bosses[rank % len(bosses)]
        env = OSRSPvEEnv(monster=boss)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def train_multitask(
    bosses: list = ["vorkath", "zulrah", "corporeal_beast"],
    timesteps: int = 2_000_000,
    n_envs: int = 12,  # More envs to cover all bosses
):
    """
    Train a single model on multiple bosses
    
    Benefits:
    - Better generalization
    - Learns common patterns (prayer, eating, positioning)
    - Can handle unseen bosses better
    """
    print("="*60)
    print("MULTI-TASK BOSS TRAINING")
    print(f"  Bosses: {bosses}")
    print(f"  Envs: {n_envs} | Steps: {timesteps:,}")
    print("="*60)
    
    # Create mixed environments (round-robin across bosses)
    env = SubprocVecEnv([
        make_random_boss_env(bosses, i) 
        for i in range(n_envs)
    ])
    
    # Eval on primary boss
    eval_env = SubprocVecEnv([make_random_boss_env([bosses[0]], 100)])
    
    run_name = f"multitask_{'_'.join(bosses)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = f"models/{run_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    callbacks = [
        CheckpointCallback(save_freq=50000//n_envs, save_path=save_dir, name_prefix="ppo"),
        EvalCallback(eval_env, best_model_save_path=save_dir, eval_freq=25000//n_envs, n_eval_episodes=20),
    ]
    
    # Larger network for multi-task
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.02,
        verbose=1,
        tensorboard_log=f"{save_dir}/tb",
        policy_kwargs={"net_arch": [512, 256, 256]},  # Bigger network
    )
    
    print(f"\nParameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)
    model.save(f"{save_dir}/final_model")
    
    print(f"\n✓ Saved to {save_dir}")
    
    env.close()
    eval_env.close()

if __name__ == "__main__":
    train_multitask(
        bosses=["vorkath", "zulrah", "abyssal_demon", "gargoyle"],
        timesteps=2_000_000,
        n_envs=12,
    )
