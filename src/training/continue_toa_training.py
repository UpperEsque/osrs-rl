#!/usr/bin/env python3
"""
Continue ToA Wardens Training with Improved Hyperparameters

This script loads an existing model and continues training with:
1. Reduced learning rate (1e-4 instead of 3e-4)
2. Increased entropy coefficient (0.03 for more exploration)
3. Better logging and mechanic tracking

Usage:
    python continue_toa_training.py --model models/toa_wardens_*/best_model.zip --timesteps 1000000
"""
import os
import sys
import argparse
import glob
from datetime import datetime

sys.path.insert(0, 'python')

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# Try to import the new environment, fall back to old one
try:
    from osrs_rl.bosses.toa_wardens_v2 import WardensEnvV2 as WardensEnv
    print("Using improved WardensEnvV2")
except ImportError:
    from osrs_rl.bosses.toa import WardensEnv
    print("Using original WardensEnv")


class DetailedEvalCallback(BaseCallback):
    """Detailed evaluation with mechanic tracking."""
    
    def __init__(self, eval_env, eval_freq=10000, n_eval=20, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval = n_eval
        self.best_winrate = 0
        self.eval_count = 0
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._evaluate()
        return True
    
    def _evaluate(self):
        self.eval_count += 1
        
        wins = 0
        total_reward = 0
        total_damage = 0
        total_taken = 0
        phases_total = 0
        episode_lengths = []
        
        death_causes = {}
        
        for _ in range(self.n_eval):
            obs, _ = self.eval_env.reset()
            done = False
            ep_reward = 0
            ep_len = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = self.eval_env.step(action)
                done = term or trunc
                ep_reward += reward
                ep_len += 1
            
            total_reward += ep_reward
            episode_lengths.append(ep_len)
            total_damage += info.get("damage_dealt", 0)
            total_taken += info.get("damage_taken", 0)
            phases_total += info.get("phases_completed", 0)
            
            if info.get("result") == "kill":
                wins += 1
            else:
                cause = info.get("death_cause", "unknown")
                death_causes[cause] = death_causes.get(cause, 0) + 1
        
        winrate = wins / self.n_eval
        avg_reward = total_reward / self.n_eval
        avg_len = np.mean(episode_lengths)
        
        print(f"\n{'='*60}")
        print(f"EVAL #{self.eval_count} @ {self.num_timesteps:,} steps")
        print(f"{'='*60}")
        print(f"Win Rate: {wins}/{self.n_eval} ({winrate*100:.1f}%)")
        print(f"Avg Reward: {avg_reward:.1f}")
        print(f"Avg Episode Length: {avg_len:.1f} ticks")
        print(f"Avg Damage Dealt: {total_damage/self.n_eval:.0f}")
        print(f"Avg Damage Taken: {total_taken/self.n_eval:.0f}")
        print(f"Avg Phases: {phases_total/self.n_eval:.2f}")
        
        if death_causes:
            print(f"\nDeath Causes:")
            total_deaths = sum(death_causes.values())
            for cause, count in sorted(death_causes.items(), key=lambda x: -x[1]):
                print(f"  {cause}: {count} ({count/total_deaths*100:.1f}%)")
        
        print(f"{'='*60}\n")
        
        # Save best model
        if winrate > self.best_winrate:
            self.best_winrate = winrate
            self.model.save(f"{self.model.logger.dir}/best_model")
            print(f"New best model! Win rate: {winrate*100:.1f}%")


def find_latest_model(pattern="models/toa_wardens_*"):
    """Find the most recent model."""
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    
    latest_dir = sorted(dirs)[-1]
    
    # Look for best_model or latest checkpoint
    best = os.path.join(latest_dir, "best_model.zip")
    if os.path.exists(best):
        return best
    
    checkpoints = glob.glob(os.path.join(latest_dir, "ppo_*.zip"))
    if checkpoints:
        return sorted(checkpoints)[-1]
    
    return None


def make_env(rank=0, seed=0, invocation=300):
    """Create a single environment."""
    def _init():
        try:
            from osrs_rl.bosses.toa_wardens_v2 import WardensEnvV2
            env = WardensEnvV2(invocation=invocation)
        except ImportError:
            from osrs_rl.bosses.toa import WardensEnv
            env = WardensEnv()
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to model to continue")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Additional timesteps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--ent", type=float, default=0.03, help="Entropy coefficient")
    parser.add_argument("--envs", type=int, default=8, help="Number of environments")
    parser.add_argument("--invocation", type=int, default=300, help="ToA invocation level")
    
    args = parser.parse_args()
    
    # Find model
    model_path = args.model or find_latest_model()
    if not model_path:
        print("No model found! Please specify with --model")
        sys.exit(1)
    
    print(f"Loading model: {model_path}")
    print(f"New learning rate: {args.lr}")
    print(f"New entropy coef: {args.ent}")
    print(f"Invocation: {args.invocation}")
    print(f"Additional timesteps: {args.timesteps:,}")
    
    # Create environments
    env = SubprocVecEnv([make_env(i, invocation=args.invocation) for i in range(args.envs)])
    
    # Create eval environment
    try:
        from osrs_rl.bosses.toa_wardens_v2 import WardensEnvV2
        eval_env = WardensEnvV2(invocation=args.invocation)
    except ImportError:
        from osrs_rl.bosses.toa import WardensEnv
        eval_env = WardensEnv()
    
    # Load model
    model = PPO.load(model_path, env=env)
    
    # Update hyperparameters
    model.learning_rate = args.lr
    model.ent_coef = args.ent
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models/toa_continued_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Callbacks
    eval_callback = DetailedEvalCallback(
        eval_env=eval_env,
        eval_freq=25000,
        n_eval=20,
        verbose=1,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_dir,
        name_prefix="ppo",
    )
    
    # Continue training
    print(f"\nStarting training...")
    print(f"Save directory: {save_dir}")
    
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, checkpoint_callback],
        reset_num_timesteps=False,
        progress_bar=True,
    )
    
    # Save final model
    model.save(os.path.join(save_dir, "final_model"))
    print(f"\nTraining complete! Model saved to {save_dir}")
    
    # Final evaluation
    print("\nFinal Evaluation (100 episodes):")
    wins = 0
    for _ in range(100):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = eval_env.step(action)
            done = term or trunc
        if info.get("result") == "kill":
            wins += 1
    
    print(f"Final Win Rate: {wins}/100 ({wins}%)")
    
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
