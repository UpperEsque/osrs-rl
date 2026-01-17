"""
OSRS Reinforcement Learning Framework

Environments:
- OSRSPvEEnv: PvE combat environment (monsters, bosses, raids)

Training:
- train(): Train PPO on a specific monster
- evaluate(): Evaluate a trained model
- train_curriculum(): Progressive difficulty training

Game Data:
- get_game_data(): Access items, NPCs, locations, skills, prayers
"""

from osrs_rl.pve_env import OSRSPvEEnv, MONSTERS, Action, Prayer, AttackStyle
from osrs_rl.game_data import get_game_data, GameData

__version__ = "0.1.0"
__all__ = [
    "OSRSPvEEnv",
    "MONSTERS", 
    "Action",
    "Prayer",
    "AttackStyle",
    "get_game_data",
    "GameData",
]

# Import detailed boss environments
try:
    from osrs_rl.bosses.vorkath import VorkathEnv, VorkathAction
    __all__.extend(["VorkathEnv", "VorkathAction"])
except ImportError:
    pass
