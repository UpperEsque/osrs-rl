#!/usr/bin/env python3
"""Test trained models"""
import sys
from osrs_rl.bosses.vorkath import VorkathEnv, VorkathAction, VorkathPhase
from osrs_rl.bosses.zulrah import ZulrahEnv
from osrs_rl.lstm_policy import FrameStackWrapper
from stable_baselines3 import PPO
import glob

def test_vorkath(model_path, n_frames=4, episodes=50):
    model = PPO.load(model_path)
    env = FrameStackWrapper(VorkathEnv(), n_frames=n_frames)
    
    kills = 0
    deaths = {"spawn": 0, "acid": 0, "fireball": 0, "low_hp": 0, "normal": 0}
    mechanics = {"fireball_dodge": 0, "fireball_total": 0,
                 "spawn_kill": 0, "spawn_total": 0,
                 "acid_walk": 0, "acid_total": 0}
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        last_phase = VorkathPhase.NORMAL
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            m = env.env.mechanics
            
            if m.vorkath.fireball_incoming:
                mechanics["fireball_total"] += 1
                if action in [7, 8, 9, 10]:  # Move actions
                    mechanics["fireball_dodge"] += 1
            
            if m.vorkath.zombified_spawn_active:
                mechanics["spawn_total"] += 1
                if action == 15:  # Crumble undead
                    mechanics["spawn_kill"] += 1
            
            if m.vorkath.phase == VorkathPhase.ACID:
                mechanics["acid_total"] += 1
                if action == 16:  # Walk around
                    mechanics["acid_walk"] += 1
            
            last_phase = m.vorkath.phase
            last_hp = m.player.current_hp
            
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
        
        if env.env.mechanics.player.current_hp > 0:
            kills += 1
        else:
            if last_phase == VorkathPhase.ZOMBIFIED_SPAWN:
                deaths["spawn"] += 1
            elif last_phase == VorkathPhase.ACID:
                deaths["acid"] += 1
            elif env.env.mechanics.vorkath.fireball_incoming:
                deaths["fireball"] += 1
            elif last_hp < 30:
                deaths["low_hp"] += 1
            else:
                deaths["normal"] += 1
    
    return kills, episodes, deaths, mechanics

if __name__ == "__main__":
    print("="*60)
    print("MODEL TESTING")
    print("="*60)
    
    # Find latest model
    paths = glob.glob("models/*/best_model.zip")
    if not paths:
        print("No models found!")
        exit()
    
    model_path = sorted(paths)[-1]
    print(f"\nTesting: {model_path}\n")
    
    kills, total, deaths, mechanics = test_vorkath(model_path, n_frames=4, episodes=100)
    
    print(f"Kill Rate: {kills}/{total} ({kills}%)")
    
    print(f"\n=== DEATH CAUSES ===")
    for cause, count in deaths.items():
        if count > 0:
            print(f"  {cause}: {count}")
    
    print(f"\n=== MECHANICS ===")
    if mechanics["fireball_total"]:
        print(f"  Fireball dodge: {mechanics['fireball_dodge']}/{mechanics['fireball_total']} ({100*mechanics['fireball_dodge']//mechanics['fireball_total']}%)")
    if mechanics["spawn_total"]:
        print(f"  Spawn kill: {mechanics['spawn_kill']}/{mechanics['spawn_total']} ({100*mechanics['spawn_kill']//mechanics['spawn_total']}%)")
    if mechanics["acid_total"]:
        print(f"  Acid walk: {mechanics['acid_walk']}/{mechanics['acid_total']} ({100*mechanics['acid_walk']//mechanics['acid_total']}%)")
