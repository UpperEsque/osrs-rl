"""
Main training script for OSRS RL
"""
import argparse
import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

import sys
sys.path.insert(0, '.')
from envs.walking_env import WalkingEnv


class PrintCallback(BaseCallback):
    """Callback to print training progress"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get('dones') is not None and self.locals['dones'][0]:
            info = self.locals.get('infos', [{}])[0]
            reward = self.locals.get('rewards', [0])[0]
            print(f"[Episode] Steps: {info.get('step', '?')}, Final Distance: {info.get('distance', '?'):.1f}, Reward: {reward:.2f}")
        return True


def make_env(task: str, port: int = 5555):
    """Create environment for task"""
    if task == "walking":
        env = WalkingEnv(
            port=port,
            target_distance=10,  # 10 tiles away
            max_episode_steps=50,  # Max 50 steps per episode
            reach_threshold=2,  # Within 2 tiles = success
        )
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return Monitor(env)


def train(args):
    """Main training loop"""
    print("=" * 60)
    print(f"OSRS RL Training")
    print(f"Task: {args.task}")
    print(f"Timesteps: {args.timesteps}")
    print("=" * 60)
    print("\nMake sure:")
    print("  1. RuneLite is running with OSRS-RL plugin")
    print("  2. You're logged into OSRS")
    print("  3. Your character is in a safe, open area")
    print()
    
    # Create environment
    env = make_env(args.task, args.port)
    
    # Create or load model
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=256,  # Smaller for real-time training
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Entropy for exploration
            verbose=1,
            tensorboard_log=f"./logs/{args.task}/",
        )
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_callback = CheckpointCallback(
        save_freq=500,  # Save every 500 steps
        save_path=f"./models/{args.task}/",
        name_prefix=f"{args.task}_{timestamp}"
    )
    
    print_callback = PrintCallback()
    
    # Train
    print("\nStarting training...")
    print("Press Ctrl+C to stop and save\n")
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, print_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    # Save final model
    final_path = f"./models/{args.task}/{args.task}_final.zip"
    model.save(final_path)
    print(f"\nModel saved to {final_path}")
    
    env.close()


def test(args):
    """Test a trained model"""
    print("=" * 60)
    print(f"OSRS RL Testing")
    print(f"Model: {args.model_path}")
    print("=" * 60)
    
    env = make_env(args.task, args.port)
    model = PPO.load(args.model_path)
    
    total_reward = 0
    episodes = 0
    
    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            episode_reward = 0
            steps = 0
            
            print(f"\n--- Episode {ep + 1} ---")
            
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                steps += 1
                
                print(f"  Step {steps}: action={action}, reward={reward:.2f}, pos={info['position']}, dist={info['distance']:.1f}")
                
                if terminated or truncated:
                    break
            
            total_reward += episode_reward
            episodes += 1
            print(f"Episode {ep + 1}: {steps} steps, {episode_reward:.2f} reward, {'SUCCESS' if terminated else 'TIMEOUT'}")
    
    except KeyboardInterrupt:
        print("\nTesting interrupted")
    
    if episodes > 0:
        print(f"\nAverage reward over {episodes} episodes: {total_reward / episodes:.2f}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OSRS RL Training")
    parser.add_argument("--task", type=str, default="walking",
                       choices=["walking", "woodcutting", "combat"],
                       help="Task to train")
    parser.add_argument("--timesteps", type=int, default=10000,
                       help="Total training timesteps")
    parser.add_argument("--port", type=int, default=5555,
                       help="Plugin port")
    parser.add_argument("--test", action="store_true",
                       help="Test mode (load and run model)")
    parser.add_argument("--model-path", type=str,
                       help="Path to model for testing")
    parser.add_argument("--resume", type=str,
                       help="Path to model to resume training from")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of test episodes")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(f"./models/{args.task}", exist_ok=True)
    os.makedirs(f"./logs/{args.task}", exist_ok=True)
    
    if args.test:
        if not args.model_path:
            print("Error: --model-path required for testing")
            exit(1)
        test(args)
    else:
        train(args)
