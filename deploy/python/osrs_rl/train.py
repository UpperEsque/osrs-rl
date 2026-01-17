#!/usr/bin/env python3
"""
PPO Training for OSRS PvE Combat

Supports:
- Multiple monster environments
- Curriculum learning (easy → hard)
- Weights & Biases logging
- Model checkpointing
- Hyperparameter tuning
"""
import os
import argparse
import time
from datetime import datetime
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# Check for stable-baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import (
        BaseCallback, 
        CheckpointCallback, 
        EvalCallback
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("stable-baselines3 not found. Install with: pip install stable-baselines3")

# Check for wandb
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from osrs_rl.pve_env import OSRSPvEEnv, MONSTERS


# ============ CUSTOM NETWORK ============

class OSRSFeatureExtractor(nn.Module):
    """
    Custom feature extractor for OSRS observations.
    
    Separates:
    - Player stats (HP, prayer, spec, supplies)
    - Combat state (prayers, gear, cooldowns)
    - Monster state (HP, phase, attack style)
    - Temporal features (tick, attack patterns)
    """
    
    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__()
        
        obs_size = observation_space.shape[0]
        
        # Player branch (indices 0-10)
        self.player_net = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # Monster branch (indices 21-30)
        self.monster_net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # Combat state branch (remaining features)
        self.combat_net = nn.Sequential(
            nn.Linear(obs_size - 21, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # Combine
        self.combine = nn.Sequential(
            nn.Linear(96, features_dim),
            nn.ReLU(),
        )
        
        self.features_dim = features_dim
    
    def forward(self, obs):
        player_features = self.player_net(obs[:, :11])
        monster_features = self.monster_net(obs[:, 21:31])
        combat_features = self.combat_net(obs[:, 11:])
        
        combined = torch.cat([player_features, monster_features, combat_features], dim=1)
        return self.combine(combined)


# ============ CALLBACKS ============

class MetricsCallback(BaseCallback):
    """Log custom OSRS-specific metrics"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.kills = []
        self.deaths = []
        self.damage_dealt = []
        self.damage_taken = []
    
    def _on_step(self) -> bool:
        # Check for episode end
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals.get("infos", [{}])[i]
                
                self.kills.append(info.get("kills", 0))
                self.deaths.append(info.get("deaths", 0))
                self.damage_dealt.append(info.get("damage_dealt", 0))
                self.damage_taken.append(info.get("damage_taken", 0))
        
        return True
    
    def _on_rollout_end(self) -> None:
        if len(self.kills) > 0:
            self.logger.record("osrs/kills", np.mean(self.kills[-100:]))
            self.logger.record("osrs/deaths", np.mean(self.deaths[-100:]))
            self.logger.record("osrs/avg_damage_dealt", np.mean(self.damage_dealt[-100:]))
            self.logger.record("osrs/avg_damage_taken", np.mean(self.damage_taken[-100:]))
            
            # Kill/death ratio
            recent_kills = sum(self.kills[-100:])
            recent_deaths = sum(self.deaths[-100:])
            kd_ratio = recent_kills / max(recent_deaths, 1)
            self.logger.record("osrs/kd_ratio", kd_ratio)


class CurriculumCallback(BaseCallback):
    """
    Curriculum learning - start with easy monsters, progress to harder ones.
    """
    
    def __init__(
        self,
        curriculum: List[str],
        threshold_reward: float = 20.0,
        min_episodes: int = 100,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.curriculum = curriculum
        self.threshold_reward = threshold_reward
        self.min_episodes = min_episodes
        self.current_stage = 0
        self.episode_count = 0
        self.recent_rewards = []
    
    def _on_step(self) -> bool:
        for done in self.locals.get("dones", []):
            if done:
                self.episode_count += 1
                # Get episode reward from info
                for info in self.locals.get("infos", []):
                    if "episode" in info:
                        self.recent_rewards.append(info["episode"]["r"])
        
        # Check for stage advancement
        if (self.episode_count >= self.min_episodes and 
            len(self.recent_rewards) >= 50):
            
            avg_reward = np.mean(self.recent_rewards[-50:])
            
            if avg_reward >= self.threshold_reward:
                self._advance_stage()
        
        return True
    
    def _advance_stage(self):
        if self.current_stage < len(self.curriculum) - 1:
            self.current_stage += 1
            new_monster = self.curriculum[self.current_stage]
            
            if self.verbose:
                print(f"\n=== CURRICULUM: Advancing to {new_monster} ===\n")
            
            # Update environment
            # Note: This requires recreating the environment
            self.recent_rewards = []
            self.episode_count = 0


# ============ TRAINING FUNCTIONS ============

def make_env(monster: str, rank: int, seed: int = 0):
    """Create a single environment"""
    def _init():
        env = OSRSPvEEnv(monster=monster)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def create_vec_env(monster: str, n_envs: int = 4, seed: int = 42):
    """Create vectorized environment"""
    set_random_seed(seed)
    
    if n_envs == 1:
        return DummyVecEnv([make_env(monster, 0, seed)])
    else:
        return SubprocVecEnv([make_env(monster, i, seed) for i in range(n_envs)])


def train(
    monster: str = "vorkath",
    total_timesteps: int = 1_000_000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    seed: int = 42,
    save_dir: str = "models",
    use_wandb: bool = False,
    wandb_project: str = "osrs-rl",
    device: str = "auto",
):
    """
    Train a PPO agent on OSRS PvE
    
    Args:
        monster: Monster to train on (from MONSTERS dict)
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        learning_rate: Learning rate
        batch_size: Minibatch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        seed: Random seed
        save_dir: Directory for model saves
        use_wandb: Whether to log to W&B
        wandb_project: W&B project name
        device: Device (auto, cpu, cuda)
    """
    if not HAS_SB3:
        print("ERROR: stable-baselines3 required for training")
        return
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    run_name = f"{monster}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"=" * 60)
    print(f"OSRS RL Training - {MONSTERS[monster].name}")
    print(f"=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel envs: {n_envs}")
    print(f"Save directory: {run_dir}")
    print()
    
    # Initialize W&B
    if use_wandb and HAS_WANDB:
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "monster": monster,
                "total_timesteps": total_timesteps,
                "n_envs": n_envs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "clip_range": clip_range,
                "ent_coef": ent_coef,
                "seed": seed,
            },
            sync_tensorboard=True,
        )
    
    # Create environment
    print("Creating environments...")
    env = create_vec_env(monster, n_envs, seed)
    eval_env = create_vec_env(monster, 1, seed + 1000)
    
    # Create callbacks
    callbacks = [
        MetricsCallback(verbose=1),
        CheckpointCallback(
            save_freq=50000 // n_envs,
            save_path=run_dir,
            name_prefix="ppo_osrs",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=run_dir,
            log_path=run_dir,
            eval_freq=10000 // n_envs,
            n_eval_episodes=10,
            deterministic=True,
        ),
    ]
    
    if use_wandb and HAS_WANDB:
        callbacks.append(WandbCallback(
            verbose=2,
            model_save_path=run_dir,
            model_save_freq=100000 // n_envs,
        ))
    
    # Create model
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=1,
        tensorboard_log=os.path.join(run_dir, "tb"),
        device=device,
        seed=seed,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    print()
    
    # Train
    print("Starting training...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/3600:.2f} hours")
    
    # Save final model
    final_path = os.path.join(run_dir, "final_model")
    model.save(final_path)
    print(f"Saved final model to {final_path}")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    if use_wandb and HAS_WANDB:
        wandb.finish()
    
    return model


def evaluate(
    model_path: str,
    monster: str = "vorkath",
    n_episodes: int = 10,
    render: bool = True,
):
    """
    Evaluate a trained model
    """
    if not HAS_SB3:
        print("ERROR: stable-baselines3 required")
        return
    
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    env = OSRSPvEEnv(monster=monster, render_mode="human" if render else None)
    
    results = {
        "kills": [],
        "deaths": [],
        "damage_dealt": [],
        "damage_taken": [],
        "episode_rewards": [],
        "episode_lengths": [],
    }
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            if render:
                env.render()
        
        results["kills"].append(info.get("kills", 0))
        results["deaths"].append(info.get("deaths", 0))
        results["damage_dealt"].append(info.get("damage_dealt", 0))
        results["damage_taken"].append(info.get("damage_taken", 0))
        results["episode_rewards"].append(total_reward)
        results["episode_lengths"].append(steps)
        
        print(f"Episode {ep+1}: Reward={total_reward:.1f}, "
              f"Kills={info.get('kills', 0)}, "
              f"Damage={info.get('damage_dealt', 0)}")
    
    print("\n=== Evaluation Results ===")
    print(f"Episodes: {n_episodes}")
    print(f"Avg Reward: {np.mean(results['episode_rewards']):.2f}")
    print(f"Avg Kills: {np.mean(results['kills']):.2f}")
    print(f"Avg Deaths: {np.mean(results['deaths']):.2f}")
    print(f"Avg Damage Dealt: {np.mean(results['damage_dealt']):.0f}")
    print(f"Avg Damage Taken: {np.mean(results['damage_taken']):.0f}")
    print(f"K/D Ratio: {sum(results['kills'])/max(sum(results['deaths']), 1):.2f}")
    
    env.close()
    return results


# ============ CURRICULUM TRAINING ============

def train_curriculum(
    total_timesteps: int = 5_000_000,
    n_envs: int = 4,
    **kwargs
):
    """
    Train with curriculum learning: easy monsters → bosses
    """
    curriculum = [
        ("sand_crab", 100_000),
        ("ammonite_crab", 200_000),
        ("abyssal_demon", 500_000),
        ("gargoyle", 500_000),
        ("vorkath", 2_000_000),
        ("zulrah", 2_000_000),
    ]
    
    model = None
    
    for monster, steps in curriculum:
        print(f"\n{'='*60}")
        print(f"CURRICULUM STAGE: {monster}")
        print(f"{'='*60}\n")
        
        if model is None:
            # First stage - create new model
            model = train(monster=monster, total_timesteps=steps, n_envs=n_envs, **kwargs)
        else:
            # Continue training on new monster
            env = create_vec_env(monster, n_envs)
            model.set_env(env)
            model.learn(total_timesteps=steps, reset_num_timesteps=False)
    
    return model


# ============ CLI ============

def main():
    parser = argparse.ArgumentParser(description="OSRS RL Training")
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--monster", type=str, default="vorkath",
                             choices=list(MONSTERS.keys()),
                             help="Monster to train on")
    train_parser.add_argument("--timesteps", type=int, default=1_000_000,
                             help="Total training timesteps")
    train_parser.add_argument("--envs", type=int, default=4,
                             help="Number of parallel environments")
    train_parser.add_argument("--lr", type=float, default=3e-4,
                             help="Learning rate")
    train_parser.add_argument("--wandb", action="store_true",
                             help="Enable W&B logging")
    train_parser.add_argument("--device", type=str, default="auto",
                             help="Device (auto, cpu, cuda)")
    
    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a model")
    eval_parser.add_argument("model", type=str, help="Path to model")
    eval_parser.add_argument("--monster", type=str, default="vorkath",
                            choices=list(MONSTERS.keys()))
    eval_parser.add_argument("--episodes", type=int, default=10)
    eval_parser.add_argument("--no-render", action="store_true")
    
    # Curriculum command
    curr_parser = subparsers.add_parser("curriculum", help="Train with curriculum")
    curr_parser.add_argument("--timesteps", type=int, default=5_000_000)
    curr_parser.add_argument("--envs", type=int, default=4)
    curr_parser.add_argument("--wandb", action="store_true")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(
            monster=args.monster,
            total_timesteps=args.timesteps,
            n_envs=args.envs,
            learning_rate=args.lr,
            use_wandb=args.wandb,
            device=args.device,
        )
    elif args.command == "eval":
        evaluate(
            model_path=args.model,
            monster=args.monster,
            n_episodes=args.episodes,
            render=not args.no_render,
        )
    elif args.command == "curriculum":
        train_curriculum(
            total_timesteps=args.timesteps,
            n_envs=args.envs,
            use_wandb=args.wandb,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
