#!/usr/bin/env python3
"""Curriculum learning: easy monsters → hard bosses"""
import sys
sys.path.insert(0, 'python')

from osrs_rl.pve_env import OSRSPvEEnv, MONSTERS
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import os
from datetime import datetime

def make_env(monster, rank, seed=0):
    def _init():
        env = OSRSPvEEnv(monster=monster)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def train_curriculum():
    """
    Progressive training:
    1. Sand crabs (easy, learn basics)
    2. Slayer monsters (medium, learn prayers)
    3. Easy bosses (hard, learn mechanics)
    4. Raid bosses (expert, master everything)
    """
    
    curriculum = [
        # (monster, steps, description)
        ("sand_crab", 100_000, "Learn basic combat"),
        ("ammonite_crab", 100_000, "Improve DPS patterns"),
        ("abyssal_demon", 200_000, "Learn prayer switching"),
        ("gargoyle", 200_000, "Learn positioning"),
        ("vorkath", 500_000, "Learn boss mechanics"),
        ("zulrah", 500_000, "Master phase transitions"),
    ]
    
    run_name = f"curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = f"models/{run_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    model = None
    n_envs = 8
    
    for stage, (monster, steps, desc) in enumerate(curriculum):
        print(f"\n{'='*60}")
        print(f"STAGE {stage+1}/{len(curriculum)}: {monster.upper()}")
        print(f"  {desc}")
        print(f"  Steps: {steps:,}")
        print(f"{'='*60}\n")
        
        env = SubprocVecEnv([make_env(monster, i) for i in range(n_envs)])
        eval_env = SubprocVecEnv([make_env(monster, 100)])
        
        if model is None:
            # First stage - create new model
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
        else:
            # Continue with existing model
            model.set_env(env)
            # Reduce LR slightly each stage
            model.learning_rate = max(1e-5, model.learning_rate * 0.8)
        
        callbacks = [
            EvalCallback(eval_env, best_model_save_path=f"{save_dir}/{monster}", 
                        eval_freq=25000//n_envs, n_eval_episodes=10),
        ]
        
        model.learn(
            total_timesteps=steps,
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=True,
        )
        
        # Save stage checkpoint
        model.save(f"{save_dir}/stage_{stage}_{monster}")
        
        env.close()
        eval_env.close()
    
    model.save(f"{save_dir}/final_curriculum_model")
    print(f"\n✓ Curriculum complete! Saved to {save_dir}")

if __name__ == "__main__":
    train_curriculum()
