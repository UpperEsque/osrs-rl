#!/usr/bin/env python3
"""Transfer learning from Vorkath to other bosses"""
import sys
sys.path.insert(0, 'python')

from osrs_rl.pve_env import OSRSPvEEnv, MONSTERS
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import glob
import os
from datetime import datetime

def make_env(monster, rank, seed=0):
    def _init():
        env = OSRSPvEEnv(monster=monster)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def transfer_train(
    source_model_path: str,
    target_monster: str,
    timesteps: int = 500_000,
    n_envs: int = 8,
):
    """
    Transfer a trained model to a new monster/boss
    
    The model already knows:
    - Prayer switching patterns
    - When to eat food
    - Attack timing
    - Resource management
    
    It just needs to adapt to new:
    - Attack patterns
    - Special mechanics
    - Damage values
    """
    print("="*60)
    print(f"TRANSFER LEARNING")
    print(f"  Source: {source_model_path}")
    print(f"  Target: {target_monster}")
    print("="*60)
    
    # Load pre-trained model
    print(f"\nLoading pre-trained model...")
    model = PPO.load(source_model_path)
    
    # Create new environment
    print(f"Creating {target_monster} environment...")
    env = SubprocVecEnv([make_env(target_monster, i) for i in range(n_envs)])
    eval_env = SubprocVecEnv([make_env(target_monster, 100)])
    
    # Update model's environment
    model.set_env(env)
    
    # Optionally reduce learning rate for fine-tuning
    model.learning_rate = 1e-4  # Lower LR for fine-tuning
    
    # Setup save directory
    run_name = f"transfer_{target_monster}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = f"models/{run_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    callbacks = [
        CheckpointCallback(save_freq=25000//n_envs, save_path=save_dir, name_prefix="ppo"),
        EvalCallback(eval_env, best_model_save_path=save_dir, eval_freq=10000//n_envs, n_eval_episodes=20),
    ]
    
    # Continue training on new monster
    print(f"\nFine-tuning on {target_monster} for {timesteps:,} steps...")
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        reset_num_timesteps=False,  # Continue from previous training
        progress_bar=True,
    )
    
    model.save(f"{save_dir}/final_model")
    print(f"\n✓ Saved to {save_dir}")
    
    env.close()
    eval_env.close()
    
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, help="Path to source model")
    parser.add_argument("--target", type=str, default="zulrah", 
                       choices=list(MONSTERS.keys()))
    parser.add_argument("--timesteps", type=int, default=500000)
    args = parser.parse_args()
    
    # Find source model if not specified
    if args.source:
        source = args.source
    else:
        # Use best vorkath model
        paths = glob.glob("models/vorkath_*/best_model.zip")
        if not paths:
            print("No source model found!")
            exit(1)
        source = sorted(paths)[-1]
    
    transfer_train(source, args.target, args.timesteps)
