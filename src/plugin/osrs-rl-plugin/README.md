# OSRS RL Plugin

RuneLite plugin for testing trained RL models in-game.

## Setup

### 1. Copy plugin to your machine

```bash
# Extract or copy the plugin folder
cp -r osrs-rl-plugin ~/osrs-rl/plugin
cd ~/osrs-rl/plugin
```

### 2. Build the plugin

```bash
gradle build
```

### 3. Run RuneLite with the plugin

**Terminal 1:**
```bash
cd ~/osrs-rl/plugin
gradle run
```

Wait for RuneLite to load and log into the game.

### 4. Enable the plugin

1. In RuneLite, click the **wrench icon** (Configuration)
2. Search for **"OSRS RL"**
3. Toggle it **ON**
4. You should see in the console: `[OSRS-RL] Plugin started on port 5050`

### 5. Run the Python client

**Terminal 2:**
```bash
cd ~/osrs-rl
python plugin/vorkath_client.py --model models/vorkath_detailed_xxx/best_model.zip
```

Or let it auto-find the best model:
```bash
python plugin/vorkath_client.py
```

## Usage

1. Go to Vorkath with proper gear
2. Start the fight
3. The RL model will take over

## Action Mapping

| ID | Action | Description |
|----|--------|-------------|
| 0 | WAIT | Do nothing |
| 1 | ATTACK_RANGED | Attack Vorkath with ranged |
| 2 | ATTACK_SPEC | Use special attack |
| 3 | PRAY_MAGE | Protect from Magic |
| 4 | PRAY_RANGE | Protect from Missiles |
| 5 | PRAY_OFF | Turn off protection prayers |
| 6 | TOGGLE_RIGOUR | Toggle Rigour prayer |
| 7 | EAT_FOOD | Eat food |
| 8 | DRINK_PRAYER | Drink prayer potion |
| 9 | DRINK_ANTIFIRE | Drink antifire |
| 10 | DRINK_ANTIVENOM | Drink antivenom |
| 11-14 | MOVE_* | Movement (N/S/E/W) |
| 15 | WALK_AROUND | Acid phase walking |
| 16 | CAST_CRUMBLE_UNDEAD | Cast on zombified spawn |

## Observation Vector (35 values)

- [0-4]: Player stats (HP%, Prayer%, Spec%, cooldowns)
- [5-9]: Active prayers
- [10-14]: Resources (food, pots)
- [15-19]: Vorkath stats (HP%, phase, distance)
- [20-24]: Active mechanics (fireball, acid, spawn)
- [25-29]: Attack type one-hot
- [30-34]: Position info

## Troubleshooting

### Plugin not showing?
- Make sure you're running with `gradle run` (not regular RuneLite)
- Check RuneLite console for errors
- Search for "OSRS RL" in plugin config

### Connection refused?
- Make sure RuneLite is running first
- Check plugin is enabled (port 5050)
- Check firewall settings

### Model not working?
- Verify model path is correct
- Check observation vector size matches (35)
- Look at console output for errors
