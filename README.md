# OSRS Reinforcement Learning

Train RL agents to play Old School RuneScape bosses (Vorkath, ToA Wardens, Zulrah).

## Project Structure
```
osrs-rl/
├── src/
│   ├── envs/          # Gym environments (simulators)
│   ├── training/      # Training scripts
│   ├── inference/     # Run trained models
│   └── plugin/        # RuneLite plugin (Java)
├── models/            # Trained models (not in git)
└── README.md
```

## Setup

### 1. Install Python dependencies
```bash
pip install stable-baselines3 gymnasium numpy
```

### 2. Build RuneLite plugin
```bash
cd src/plugin
gradle build
cp build/libs/*.jar ~/.runelite/plugins/
```

## Training
```bash
# Train Vorkath (simple)
python src/training/train_vorkath_detailed.py

# Train with curriculum (recommended)
python src/training/train_toa_curriculum_v2.py
```

## Run Live
```bash
# Start RuneLite with plugin first, then:
python src/inference/run_vorkath.py
```

## Current Status

- [x] Vorkath simulation environment
- [x] ToA Wardens simulation environment  
- [x] RuneLite plugin for game state extraction
- [ ] Working live inference (sim-to-real gap)
- [ ] Imitation learning from human play
