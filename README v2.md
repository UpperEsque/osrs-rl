# OSRS RL Improvements Package

This package contains improved training scripts and environments for ToA Wardens based on analysis of your current training progress.

## Quick Start

### 1. Copy Files to Your Project

```bash
# Copy all files to your osrs-rl directory
cp python/osrs_rl/bosses/toa_wardens_v2.py ~/osrs-rl/python/osrs_rl/bosses/
cp *.py ~/osrs-rl/
```

### 2. Option A: Continue Current Training with Better Hyperparameters

If you want to continue your current model with improved settings:

```bash
cd ~/osrs-rl

# This will:
# - Load your latest ToA model
# - Reduce learning rate to 1e-4
# - Increase entropy to 0.03 for more exploration
# - Continue training with better logging

python continue_toa_training.py --timesteps 1000000
```

### 3. Option B: Start Fresh with Proper Curriculum

For best results, start fresh with invocation-based curriculum:

```bash
cd ~/osrs-rl

# This will train from 0 → 150 → 300 → 450 → 600 invocation
python train_toa_curriculum_v2.py --timesteps 5000000 --envs 8
```

### 4. Diagnose Current Model

Understand what's killing your agent:

```bash
python diagnose_toa_model.py --model models/toa_wardens_xxx/best_model.zip --episodes 100
```

## Key Improvements

### 1. Learning Rate Schedule

| Stage | Invocation | Learning Rate |
|-------|------------|---------------|
| 1 | 0 | 3e-4 |
| 2 | 150 | 2e-4 |
| 3 | 300 | 1.5e-4 |
| 4 | 450 | 1e-4 |
| 5 | 600 | 5e-5 |

### 2. Dense Reward Shaping

```python
reward += damage_dealt * 0.1    # DPS reward
reward += 3.0                   # dodged slam
reward += 15.0                  # killed core
reward += 2.0                   # correct prayer
reward += 30.0                  # phase completed
reward += 5/10/15               # survival milestones
```

### 3. Mechanic Tracking

- `slams_dodged` / `slams_hit`
- `cores_killed` / `cores_exploded`
- `lightning_dodged` / `lightning_hit`
- `prayers_correct` / `prayers_wrong`
- `death_cause` tracking

## Expected Results

| Invocation | Target Win Rate | Steps Needed |
|------------|-----------------|--------------|
| 0 | 75% | 300k-750k |
| 150 | 65% | 400k-1M |
| 300 | 55% | 500k-1.5M |
| 450 | 45% | 750k-2M |
| 600 | 35% | 1M-3M |

## Command Reference

```bash
# Diagnose current model
python diagnose_toa_model.py --model models/xxx/best_model.zip

# Continue with better hyperparams
python continue_toa_training.py --model models/xxx/best_model.zip --lr 1e-4 --ent 0.03

# Fresh curriculum training
python train_toa_curriculum_v2.py --timesteps 5000000

# Evaluate at specific invocation
python train_toa_curriculum_v2.py --eval models/xxx/best_model.zip --invocation 600
```
