#!/usr/bin/env python3
"""
ToA Wardens Model Diagnostics

Analyzes a trained model to understand:
1. What's killing the agent
2. Which mechanics are being handled well/poorly
3. Action distribution
4. Phase progression
5. Resource management

Usage:
    python diagnose_toa_model.py --model models/toa_wardens_*/best_model.zip
"""
import os
import sys
import argparse
import glob
from collections import defaultdict

sys.path.insert(0, 'python')

import numpy as np
from stable_baselines3 import PPO

# Try to import environments
try:
    from osrs_rl.bosses.toa_wardens_v2 import WardensEnvV2 as WardensEnv, WardensAction
    ENV_VERSION = "V2"
except ImportError:
    from osrs_rl.bosses.toa import WardensEnv
    # Create dummy action enum if not available
    class WardensAction:
        pass
    ENV_VERSION = "V1"


def find_latest_model(pattern="models/toa_*"):
    """Find the most recent model."""
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    
    for d in sorted(dirs, reverse=True):
        best = os.path.join(d, "best_model.zip")
        if os.path.exists(best):
            return best
        checkpoints = glob.glob(os.path.join(d, "*.zip"))
        if checkpoints:
            return sorted(checkpoints)[-1]
    return None


def run_diagnostics(model_path: str, n_episodes: int = 100, invocation: int = 300):
    """Run comprehensive diagnostics on a model."""
    
    print("=" * 70)
    print("TOA WARDENS MODEL DIAGNOSTICS")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Environment: {ENV_VERSION}")
    print(f"Invocation: {invocation}")
    print(f"Episodes: {n_episodes}")
    print("=" * 70)
    
    # Load model
    model = PPO.load(model_path)
    
    if ENV_VERSION == "V2":
        env = WardensEnv(invocation=invocation)
    else:
        env = WardensEnv()
    
    # Tracking variables
    results = {"kill": 0, "death": 0, "timeout": 0}
    death_causes = defaultdict(int)
    action_counts = defaultdict(int)
    
    # Per-episode metrics
    episode_rewards = []
    episode_lengths = []
    damage_dealt_list = []
    damage_taken_list = []
    phases_reached = defaultdict(int)
    
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
    }
    
    # Resource tracking
    food_used = []
    brew_used = []
    restore_used = []
    
    # HP tracking at death
    hp_at_death = []
    prayer_at_death = []
    
    print("\nRunning episodes...")
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0
        ep_actions = defaultdict(int)
        
        initial_food = getattr(env, 'player', None)
        if initial_food:
            initial_food = initial_food.food_count
        else:
            initial_food = 12
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            ep_actions[int(action)] += 1
            action_counts[int(action)] += 1
            
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            ep_reward += reward
            ep_length += 1
        
        # Record results
        result = info.get("result", "unknown")
        results[result] = results.get(result, 0) + 1
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        damage_dealt_list.append(info.get("damage_dealt", 0))
        damage_taken_list.append(info.get("damage_taken", 0))
        
        # Track phase reached
        phase = info.get("phase", "P1_ELIDINIS")
        phases_reached[phase] += 1
        
        # Track mechanics
        for key in mechanics:
            mechanics[key] += info.get(key, 0)
        
        # Track death causes
        if result == "death":
            cause = info.get("death_cause", "unknown")
            death_causes[cause] += 1
            
            # Track state at death
            if hasattr(env, 'player'):
                hp_at_death.append(env.player.current_hp)
                prayer_at_death.append(env.player.current_prayer)
        
        # Track resources used
        if hasattr(env, 'player'):
            food_used.append(initial_food - env.player.food_count)
        
        # Progress indicator
        if (ep + 1) % 20 == 0:
            current_wr = results["kill"] / (ep + 1) * 100
            print(f"  Episode {ep+1:3d}/{n_episodes}: Win Rate so far: {current_wr:.1f}%")
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    # Win/Loss
    total = sum(results.values())
    print(f"\n--- Outcomes ---")
    for outcome, count in sorted(results.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {outcome:10s}: {count:3d} ({pct:5.1f}%) {bar}")
    
    winrate = results["kill"] / total
    print(f"\n  WIN RATE: {winrate*100:.1f}%")
    
    # Episode stats
    print(f"\n--- Episode Statistics ---")
    print(f"  Avg Reward:        {np.mean(episode_rewards):7.1f} ± {np.std(episode_rewards):.1f}")
    print(f"  Avg Length:        {np.mean(episode_lengths):7.1f} ± {np.std(episode_lengths):.1f} ticks")
    print(f"  Avg Damage Dealt:  {np.mean(damage_dealt_list):7.0f}")
    print(f"  Avg Damage Taken:  {np.mean(damage_taken_list):7.0f}")
    print(f"  Damage Ratio:      {np.mean(damage_dealt_list) / (np.mean(damage_taken_list) + 1):7.2f}x")
    
    # Phase progression
    print(f"\n--- Phase Progression ---")
    for phase, count in sorted(phases_reached.items()):
        pct = count / total * 100
        print(f"  Reached {phase}: {count} ({pct:.1f}%)")
    
    # Death causes
    if death_causes:
        print(f"\n--- Death Causes ---")
        total_deaths = sum(death_causes.values())
        for cause, count in sorted(death_causes.items(), key=lambda x: -x[1]):
            pct = count / total_deaths * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {cause:20s}: {count:3d} ({pct:5.1f}%) {bar}")
    
    # Mechanic success rates
    print(f"\n--- Mechanic Success Rates ---")
    
    slam_total = mechanics["slams_dodged"] + mechanics["slams_hit"]
    if slam_total > 0:
        pct = mechanics["slams_dodged"] / slam_total * 100
        status = "✓" if pct > 70 else "⚠" if pct > 40 else "✗"
        print(f"  {status} Slam Dodge:     {pct:5.1f}% ({mechanics['slams_dodged']}/{slam_total})")
    
    core_total = mechanics["cores_killed"] + mechanics["cores_exploded"]
    if core_total > 0:
        pct = mechanics["cores_killed"] / core_total * 100
        status = "✓" if pct > 70 else "⚠" if pct > 40 else "✗"
        print(f"  {status} Core Kill:      {pct:5.1f}% ({mechanics['cores_killed']}/{core_total})")
    
    lightning_total = mechanics["lightning_dodged"] + mechanics["lightning_hit"]
    if lightning_total > 0:
        pct = mechanics["lightning_dodged"] / lightning_total * 100
        status = "✓" if pct > 70 else "⚠" if pct > 40 else "✗"
        print(f"  {status} Lightning:      {pct:5.1f}% ({mechanics['lightning_dodged']}/{lightning_total})")
    
    prayer_total = mechanics["prayers_correct"] + mechanics["prayers_wrong"]
    if prayer_total > 0:
        pct = mechanics["prayers_correct"] / prayer_total * 100
        status = "✓" if pct > 70 else "⚠" if pct > 40 else "✗"
        print(f"  {status} Prayer Switch:  {pct:5.1f}% ({mechanics['prayers_correct']}/{prayer_total})")
    
    # Action distribution
    print(f"\n--- Action Distribution (Top 15) ---")
    total_actions = sum(action_counts.values())
    
    # Try to map action IDs to names
    try:
        action_names = {a.value: a.name for a in WardensAction}
    except:
        action_names = {}
    
    for action_id, count in sorted(action_counts.items(), key=lambda x: -x[1])[:15]:
        name = action_names.get(action_id, f"Action_{action_id}")
        pct = count / total_actions * 100
        bar = "█" * int(pct / 2) + "░" * (25 - int(pct / 2))
        print(f"  {name:25s}: {pct:5.1f}% {bar}")
    
    # Resource usage
    if food_used:
        print(f"\n--- Resource Usage (per episode) ---")
        print(f"  Avg Food Used: {np.mean(food_used):.1f}")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    recommendations = []
    
    if winrate < 0.3:
        recommendations.append("• Win rate is low - consider starting with lower invocation")
    
    if slam_total > 0 and mechanics["slams_dodged"] / slam_total < 0.5:
        recommendations.append("• Slam dodge rate is low - increase penalty for getting hit by slams")
    
    if core_total > 0 and mechanics["cores_killed"] / core_total < 0.5:
        recommendations.append("• Core kill rate is low - increase reward for killing cores")
    
    if prayer_total > 0 and mechanics["prayers_correct"] / prayer_total < 0.6:
        recommendations.append("• Prayer switching needs improvement - add prayer prediction to observations")
    
    if death_causes.get("auto_attack", 0) > total_deaths * 0.3 if death_causes else False:
        recommendations.append("• Many deaths from auto attacks - improve DPS and resource management")
    
    if np.mean(episode_lengths) < 60:
        recommendations.append("• Episodes are short - agent dying early, needs better survival skills")
    
    if not recommendations:
        recommendations.append("• Model is performing well! Consider increasing invocation.")
    
    for rec in recommendations:
        print(rec)
    
    print("\n" + "=" * 70)
    
    return {
        "winrate": winrate,
        "avg_reward": np.mean(episode_rewards),
        "avg_length": np.mean(episode_lengths),
        "death_causes": dict(death_causes),
        "mechanics": mechanics,
        "action_distribution": dict(action_counts),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to model")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--invocation", type=int, default=300, help="ToA invocation")
    
    args = parser.parse_args()
    
    model_path = args.model or find_latest_model()
    if not model_path:
        print("No model found! Please specify with --model")
        sys.exit(1)
    
    run_diagnostics(model_path, args.episodes, args.invocation)


if __name__ == "__main__":
    main()
