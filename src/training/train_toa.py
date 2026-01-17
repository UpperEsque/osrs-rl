#!/usr/bin/env python3
"""Train on ToA bosses with max invocations (600+)"""
import sys
sys.path.insert(0, 'python')

from osrs_rl.bosses.toa import (
    KephriEnv, ZebakEnv, AkkhaEnv, BaBaEnv, WardensEnv
)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
from datetime import datetime
import argparse

TOA_BOSSES = {
    "kephri": KephriEnv,
    "zebak": ZebakEnv,
    "akkha": AkkhaEnv,
    "baba": BaBaEnv,
    "wardens": WardensEnv,
}

def make_env(EnvClass, rank, seed=0):
    def _init():
        env = EnvClass()
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def train_toa(boss="wardens", timesteps=2_000_000, n_envs=8):
    EnvClass = TOA_BOSSES[boss]
    
    run_name = f"toa_{boss}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = f"models/{run_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*60)
    print(f"ToA Training - {boss.upper()} (600+ Invocation)")
    print("="*60)
    print(f"Envs: {n_envs} | Timesteps: {timesteps:,}")
    print(f"Save: {save_dir}\n")
    
    env = SubprocVecEnv([make_env(EnvClass, i) for i in range(n_envs)])
    eval_env = SubprocVecEnv([make_env(EnvClass, 100)])
    
    callbacks = [
        CheckpointCallback(save_freq=50000//n_envs, save_path=save_dir, name_prefix="ppo"),
        EvalCallback(eval_env, best_model_save_path=save_dir, eval_freq=25000//n_envs, n_eval_episodes=20),
    ]
    
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
        ent_coef=0.02,
        verbose=1,
        tensorboard_log=f"{save_dir}/tb",
        policy_kwargs={"net_arch": [512, 256, 256]},  # Bigger for complex mechanics
    )
    
    print(f"Parameters: {sum(p.numel() for p in model.policy.parameters()):,}\n")
    
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)
    model.save(f"{save_dir}/final_model")
    
    print(f"\n✓ Saved to {save_dir}")
    
    env.close()
    eval_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--boss", type=str, default="wardens", choices=list(TOA_BOSSES.keys()))
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--envs", type=int, default=8)
    args = parser.parse_args()
    
    train_toa(args.boss, args.timesteps, args.envs)
