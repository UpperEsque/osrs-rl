#!/usr/bin/env python3
"""
ToA Wardens Curriculum Training V2

Best Practices Implemented:
1. Proper curriculum within ToA (0 → 150 → 300 → 450 → 600 invocation)
2. Learning rate decay schedule
3. Adaptive entropy coefficient
4. Dense reward shaping
5. Mechanic-specific tracking and logging
6. Automatic stage progression based on win rate
7. Model checkpointing with performance metrics

Usage:
    python train_toa_curriculum_v2.py --timesteps 5000000
    python train_toa_curriculum_v2.py --resume models/toa_curriculum_v2_*/latest.zip
    python train_toa_curriculum_v2.py --eval models/toa_curriculum_v2_*/best_model.zip
"""
import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional, Callable
from collections import deque

import numpy as np

# Add python path
sys.path.insert(0, 'python')

try:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install stable-baselines3 torch")
    sys.exit(1)

# Import our improved environment
from osrs_rl.bosses.toa_wardens_v2 import WardensEnvV2, make_wardens_env


# =============================================================================
# CURRICULUM CONFIGURATION
# =============================================================================

CURRICULUM_STAGES = [
    {
        "name": "Stage 1: Tutorial (0 invocation)",
        "invocation": 0,
        "target_winrate": 0.75,
        "min_steps": 300_000,
        "max_steps": 750_000,
        "learning_rate": 3e-4,
        "ent_coef": 0.02,
    },
    {
        "name": "Stage 2: Normal (150 invocation)",
        "invocation": 150,
        "target_winrate": 0.65,
        "min_steps": 400_000,
        "max_steps": 1_000_000,
        "learning_rate": 2e-4,
        "ent_coef": 0.015,
    },
    {
        "name": "Stage 3: Expert (300 invocation)",
        "invocation": 300,
        "target_winrate": 0.55,
        "min_steps": 500_000,
        "max_steps": 1_500_000,
        "learning_rate": 1.5e-4,
        "ent_coef": 0.01,
    },
    {
        "name": "Stage 4: Master (450 invocation)",
        "invocation": 450,
        "target_winrate": 0.45,
        "min_steps": 750_000,
        "max_steps": 2_000_000,
        "learning_rate": 1e-4,
        "ent_coef": 0.01,
    },
    {
        "name": "Stage 5: Grandmaster (600 invocation)",
        "invocation": 600,
        "target_winrate": 0.35,
        "min_steps": 1_000_000,
        "max_steps": 3_000_000,
        "learning_rate": 5e-5,
        "ent_coef": 0.005,
    },
]


# =============================================================================
# LEARNING RATE SCHEDULES
# =============================================================================

def linear_schedule(initial_value: float, final_value: float = 1e-5) -> Callable[[float], float]:
    """Linear learning rate decay."""
    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return func


def cosine_schedule(initial_value: float, final_value: float = 1e-5) -> Callable[[float], float]:
    """Cosine annealing learning rate decay."""
    def func(progress_remaining: float) -> float:
        return final_value + 0.5 * (initial_value - final_value) * (1 + np.cos(np.pi * (1 - progress_remaining)))
    return func


# =============================================================================
# CUSTOM CALLBACKS
# =============================================================================

class CurriculumCallback(BaseCallback):
    """
    Callback for curriculum learning with detailed tracking.
    
    Tracks:
    - Win rate over rolling window
    - Mechanic-specific success rates
    - Automatic stage progression
    """
    
    def __init__(
        self,
        stages: List[Dict],
        eval_env_fn: Callable,
        n_eval_episodes: int = 20,
        eval_freq: int = 10000,
        save_path: str = "models",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.stages = stages
        self.eval_env_fn = eval_env_fn
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.save_path = save_path
        
        self.current_stage = 0
        self.stage_steps = 0
        self.total_steps = 0
        
        # Rolling metrics
        self.recent_wins = deque(maxlen=100)
        self.recent_rewards = deque(maxlen=100)
        
        # Detailed tracking
        self.metrics_history = []
        self.best_winrate = 0.0
        
    def _on_training_start(self):
        """Initialize at training start."""
        os.makedirs(self.save_path, exist_ok=True)
        self._log_stage_start()
        
    def _on_step(self) -> bool:
        """Called at each step."""
        self.stage_steps += 1
        self.total_steps += 1
        
        # Check for episode completion in infos
        for info in self.locals.get("infos", []):
            if "result" in info:
                is_win = info["result"] == "kill"
                self.recent_wins.append(1 if is_win else 0)
                self.recent_rewards.append(info.get("episode", {}).get("r", 0))
        
        # Periodic evaluation
        if self.stage_steps % self.eval_freq == 0:
            self._evaluate_and_maybe_progress()
        
        return True
    
    def _evaluate_and_maybe_progress(self):
        """Evaluate current performance and check for stage progression."""
        stage = self.stages[self.current_stage]
        
        # Run evaluation
        eval_metrics = self._run_evaluation()
        
        # Log metrics
        self._log_metrics(eval_metrics)
        
        # Check for stage progression
        winrate = eval_metrics["winrate"]
        
        if winrate > self.best_winrate:
            self.best_winrate = winrate
            self._save_model("best_model")
        
        # Progress conditions
        should_progress = (
            self.stage_steps >= stage["min_steps"] and
            winrate >= stage["target_winrate"]
        ) or self.stage_steps >= stage["max_steps"]
        
        if should_progress and self.current_stage < len(self.stages) - 1:
            self._progress_to_next_stage()
    
    def _run_evaluation(self) -> Dict:
        """Run evaluation episodes and collect metrics."""
        stage = self.stages[self.current_stage]
        env = self.eval_env_fn(stage["invocation"])
        
        wins = 0
        total_reward = 0
        total_damage_dealt = 0
        total_damage_taken = 0
        
        # Mechanic tracking
        mechanics = {
            "slams_dodged": 0,
            "slams_hit": 0,
            "cores_killed": 0,
            "cores_exploded": 0,
            "lightning_dodged": 0,
            "lightning_hit": 0,
            "prayers_correct": 0,
            "prayers_wrong": 0,
            "phases_completed": 0,
        }
        
        death_causes = {}
        
        for _ in range(self.n_eval_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            # Track results
            if info.get("result") == "kill":
                wins += 1
            
            total_reward += episode_reward
            total_damage_dealt += info.get("damage_dealt", 0)
            total_damage_taken += info.get("damage_taken", 0)
            
            # Track mechanics
            for key in mechanics:
                mechanics[key] += info.get(key, 0)
            
            # Track death causes
            cause = info.get("death_cause", "unknown")
            if info.get("result") == "death":
                death_causes[cause] = death_causes.get(cause, 0) + 1
        
        env.close()
        
        return {
            "winrate": wins / self.n_eval_episodes,
            "wins": wins,
            "avg_reward": total_reward / self.n_eval_episodes,
            "avg_damage_dealt": total_damage_dealt / self.n_eval_episodes,
            "avg_damage_taken": total_damage_taken / self.n_eval_episodes,
            "mechanics": {k: v / self.n_eval_episodes for k, v in mechanics.items()},
            "death_causes": death_causes,
        }
    
    def _log_metrics(self, metrics: Dict):
        """Log detailed metrics."""
        stage = self.stages[self.current_stage]
        
        print("\n" + "=" * 70)
        print(f"EVALUATION @ {self.total_steps:,} steps (Stage {self.current_stage + 1})")
        print(f"Invocation: {stage['invocation']} | Target: {stage['target_winrate']*100:.0f}%")
        print("=" * 70)
        
        # Win rate
        winrate = metrics["winrate"]
        bar = "█" * int(winrate * 20) + "░" * (20 - int(winrate * 20))
        print(f"\nWin Rate: {metrics['wins']}/{self.n_eval_episodes} ({winrate*100:.1f}%) {bar}")
        print(f"Avg Reward: {metrics['avg_reward']:.1f}")
        print(f"Avg Damage Dealt: {metrics['avg_damage_dealt']:.0f}")
        print(f"Avg Damage Taken: {metrics['avg_damage_taken']:.0f}")
        
        # Mechanic breakdown
        m = metrics["mechanics"]
        print(f"\n--- Mechanics ---")
        
        slam_total = m["slams_dodged"] + m["slams_hit"]
        if slam_total > 0:
            print(f"Slam Dodge:    {m['slams_dodged']/slam_total*100:5.1f}% ({m['slams_dodged']:.1f}/{slam_total:.1f})")
        
        core_total = m["cores_killed"] + m["cores_exploded"]
        if core_total > 0:
            print(f"Core Kill:     {m['cores_killed']/core_total*100:5.1f}% ({m['cores_killed']:.1f}/{core_total:.1f})")
        
        lightning_total = m["lightning_dodged"] + m["lightning_hit"]
        if lightning_total > 0:
            print(f"Lightning:     {m['lightning_dodged']/lightning_total*100:5.1f}% ({m['lightning_dodged']:.1f}/{lightning_total:.1f})")
        
        prayer_total = m["prayers_correct"] + m["prayers_wrong"]
        if prayer_total > 0:
            print(f"Prayer Switch: {m['prayers_correct']/prayer_total*100:5.1f}% ({m['prayers_correct']:.1f}/{prayer_total:.1f})")
        
        print(f"Phases/Episode: {m['phases_completed']:.2f}")
        
        # Death causes
        if metrics["death_causes"]:
            print(f"\n--- Death Causes ---")
            total_deaths = sum(metrics["death_causes"].values())
            for cause, count in sorted(metrics["death_causes"].items(), key=lambda x: -x[1]):
                pct = count / total_deaths * 100
                print(f"  {cause:20s}: {count:2d} ({pct:5.1f}%)")
        
        print("=" * 70 + "\n")
        
        # Save metrics history
        self.metrics_history.append({
            "step": self.total_steps,
            "stage": self.current_stage,
            "invocation": stage["invocation"],
            **metrics,
        })
        
        # Save metrics to file
        metrics_file = os.path.join(self.save_path, "metrics_history.json")
        with open(metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
    
    def _progress_to_next_stage(self):
        """Progress to next curriculum stage."""
        # Save current stage model
        self._save_model(f"stage_{self.current_stage + 1}_final")
        
        self.current_stage += 1
        self.stage_steps = 0
        self.best_winrate = 0.0
        
        stage = self.stages[self.current_stage]
        
        # Update hyperparameters
        self.model.learning_rate = stage["learning_rate"]
        self.model.ent_coef = stage["ent_coef"]
        
        # Recreate environments with new invocation
        self._recreate_envs(stage["invocation"])
        
        self._log_stage_start()
    
    def _recreate_envs(self, invocation: int):
        """Recreate training environments with new invocation."""
        # This requires access to the training loop's env
        # We'll handle this by setting a flag that the main loop checks
        self.new_invocation = invocation
        self.needs_env_update = True
    
    def _log_stage_start(self):
        """Log stage start."""
        stage = self.stages[self.current_stage]
        print("\n" + "=" * 70)
        print(f"STARTING {stage['name'].upper()}")
        print(f"Target Win Rate: {stage['target_winrate']*100:.0f}%")
        print(f"Learning Rate: {stage['learning_rate']}")
        print(f"Entropy Coef: {stage['ent_coef']}")
        print(f"Min Steps: {stage['min_steps']:,} | Max Steps: {stage['max_steps']:,}")
        print("=" * 70 + "\n")
    
    def _save_model(self, name: str):
        """Save model checkpoint."""
        path = os.path.join(self.save_path, f"{name}.zip")
        self.model.save(path)
        if self.verbose:
            print(f"Saved model: {path}")


class MetricsCallback(BaseCallback):
    """Lightweight callback for tracking training metrics."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = 0
        self.losses = 0
        
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
            if info.get("result") == "kill":
                self.wins += 1
            elif info.get("result") == "death":
                self.losses += 1
        return True


# =============================================================================
# ENVIRONMENT CREATION
# =============================================================================

def make_train_env(invocation: int, n_envs: int = 8, seed: int = 0):
    """Create vectorized training environment."""
    def _make_env(rank: int):
        def _init():
            env = WardensEnvV2(invocation=invocation)
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env
        return _init
    
    if n_envs > 1:
        return SubprocVecEnv([_make_env(i) for i in range(n_envs)])
    else:
        return DummyVecEnv([_make_env(0)])


def make_eval_env(invocation: int):
    """Create single evaluation environment."""
    env = WardensEnvV2(invocation=invocation)
    return env


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_curriculum(
    total_timesteps: int = 5_000_000,
    n_envs: int = 8,
    resume_from: Optional[str] = None,
    save_dir: Optional[str] = None,
    seed: int = 42,
):
    """
    Run curriculum training for ToA Wardens.
    
    Args:
        total_timesteps: Total training steps across all stages
        n_envs: Number of parallel environments
        resume_from: Path to model to resume from
        save_dir: Directory to save models
        seed: Random seed
    """
    set_random_seed(seed)
    
    # Create save directory
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"models/toa_curriculum_v2_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save config
    config = {
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "seed": seed,
        "stages": CURRICULUM_STAGES,
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Determine starting stage
    start_stage = 0
    if resume_from:
        # Try to determine stage from filename or metrics
        print(f"Resuming from: {resume_from}")
        # Load metrics if available
        metrics_path = os.path.join(os.path.dirname(resume_from), "metrics_history.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                history = json.load(f)
                if history:
                    start_stage = history[-1].get("stage", 0)
    
    # Create initial environment
    initial_invocation = CURRICULUM_STAGES[start_stage]["invocation"]
    env = make_train_env(initial_invocation, n_envs, seed)
    
    # Create or load model
    if resume_from:
        model = PPO.load(resume_from, env=env)
        print(f"Loaded model from {resume_from}")
    else:
        stage = CURRICULUM_STAGES[start_stage]
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=linear_schedule(stage["learning_rate"]),
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=stage["ent_coef"],
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=os.path.join(save_dir, "tensorboard"),
            seed=seed,
            policy_kwargs={
                "net_arch": dict(pi=[256, 256], vf=[256, 256]),
            },
        )
    
    # Create callbacks
    curriculum_callback = CurriculumCallback(
        stages=CURRICULUM_STAGES,
        eval_env_fn=make_eval_env,
        n_eval_episodes=20,
        eval_freq=25000,
        save_path=save_dir,
        verbose=1,
    )
    curriculum_callback.current_stage = start_stage
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_dir,
        name_prefix="checkpoint",
    )
    
    # Training loop with environment updates
    steps_trained = 0
    while steps_trained < total_timesteps:
        # Train for a chunk
        chunk_steps = min(100000, total_timesteps - steps_trained)
        
        model.learn(
            total_timesteps=chunk_steps,
            callback=[curriculum_callback, checkpoint_callback],
            reset_num_timesteps=False,
            progress_bar=True,
        )
        
        steps_trained += chunk_steps
        
        # Check if we need to update environments
        if hasattr(curriculum_callback, 'needs_env_update') and curriculum_callback.needs_env_update:
            new_invocation = curriculum_callback.new_invocation
            print(f"\nUpdating environments to invocation {new_invocation}...")
            
            # Close old env and create new one
            env.close()
            env = make_train_env(new_invocation, n_envs, seed)
            model.set_env(env)
            
            curriculum_callback.needs_env_update = False
    
    # Save final model
    final_path = os.path.join(save_dir, "final_model.zip")
    model.save(final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ACROSS ALL INVOCATIONS")
    print("=" * 70)
    
    for stage in CURRICULUM_STAGES:
        inv = stage["invocation"]
        eval_env = make_eval_env(inv)
        
        wins = 0
        for _ in range(50):
            obs, _ = eval_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
            if info.get("result") == "kill":
                wins += 1
        
        print(f"Invocation {inv:3d}: {wins}/50 kills ({wins*2}%)")
        eval_env.close()
    
    env.close()
    return model, save_dir


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def evaluate_model(model_path: str, invocation: int = 600, n_episodes: int = 100):
    """
    Evaluate a trained model with detailed metrics.
    """
    model = PPO.load(model_path)
    env = WardensEnvV2(invocation=invocation)
    
    wins = 0
    total_reward = 0
    
    mechanics = {
        "slams_dodged": 0,
        "slams_hit": 0,
        "cores_killed": 0,
        "cores_exploded": 0,
        "lightning_dodged": 0,
        "lightning_hit": 0,
        "prayers_correct": 0,
        "prayers_wrong": 0,
        "phases_completed": 0,
    }
    
    death_causes = {}
    episode_lengths = []
    
    print(f"\nEvaluating {model_path}")
    print(f"Invocation: {invocation} | Episodes: {n_episodes}")
    print("-" * 50)
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_length += 1
        
        total_reward += ep_reward
        episode_lengths.append(ep_length)
        
        if info.get("result") == "kill":
            wins += 1
            status = "✓"
        else:
            status = "✗"
            cause = info.get("death_cause", "unknown")
            death_causes[cause] = death_causes.get(cause, 0) + 1
        
        for key in mechanics:
            mechanics[key] += info.get(key, 0)
        
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1:3d}: {status} | Reward: {ep_reward:6.1f} | Ticks: {ep_length:3d}")
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS - Invocation {invocation}")
    print("=" * 70)
    
    winrate = wins / n_episodes
    bar = "█" * int(winrate * 20) + "░" * (20 - int(winrate * 20))
    print(f"\nWin Rate: {wins}/{n_episodes} ({winrate*100:.1f}%) {bar}")
    print(f"Avg Reward: {total_reward / n_episodes:.1f}")
    print(f"Avg Episode Length: {np.mean(episode_lengths):.1f} ticks")
    
    print(f"\n--- Mechanics (per episode avg) ---")
    
    slam_total = mechanics["slams_dodged"] + mechanics["slams_hit"]
    if slam_total > 0:
        pct = mechanics["slams_dodged"] / slam_total * 100
        print(f"Slam Dodge:    {pct:5.1f}%")
    
    core_total = mechanics["cores_killed"] + mechanics["cores_exploded"]
    if core_total > 0:
        pct = mechanics["cores_killed"] / core_total * 100
        print(f"Core Kill:     {pct:5.1f}%")
    
    lightning_total = mechanics["lightning_dodged"] + mechanics["lightning_hit"]
    if lightning_total > 0:
        pct = mechanics["lightning_dodged"] / lightning_total * 100
        print(f"Lightning:     {pct:5.1f}%")
    
    prayer_total = mechanics["prayers_correct"] + mechanics["prayers_wrong"]
    if prayer_total > 0:
        pct = mechanics["prayers_correct"] / prayer_total * 100
        print(f"Prayer:        {pct:5.1f}%")
    
    print(f"Phases/Ep:     {mechanics['phases_completed'] / n_episodes:.2f}")
    
    if death_causes:
        print(f"\n--- Death Causes ---")
        total_deaths = sum(death_causes.values())
        for cause, count in sorted(death_causes.items(), key=lambda x: -x[1]):
            pct = count / total_deaths * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {cause:20s}: {count:3d} ({pct:5.1f}%) {bar}")
    
    print("=" * 70)
    
    return {
        "winrate": winrate,
        "avg_reward": total_reward / n_episodes,
        "mechanics": {k: v / n_episodes for k, v in mechanics.items()},
        "death_causes": death_causes,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ToA Wardens Curriculum Training V2")
    parser.add_argument("--timesteps", type=int, default=5_000_000, help="Total training timesteps")
    parser.add_argument("--envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--resume", type=str, default=None, help="Path to model to resume from")
    parser.add_argument("--eval", type=str, default=None, help="Path to model to evaluate")
    parser.add_argument("--invocation", type=int, default=600, help="Invocation for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    if args.eval:
        evaluate_model(args.eval, args.invocation)
    else:
        train_curriculum(
            total_timesteps=args.timesteps,
            n_envs=args.envs,
            resume_from=args.resume,
            seed=args.seed,
        )
