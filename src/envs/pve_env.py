"""
OSRS PvE Combat Environment for Reinforcement Learning

Supports:
- Regular monsters (Slayer, training)
- Bosses (Zulrah, Vorkath, etc.)
- Raid bosses (CoX, ToB, ToA)
- Multi-phase fights
- Prayer flicking
- Gear switching
- Eating/potions
- Special attacks
- Movement/positioning
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from enum import IntEnum
import json
import os


class AttackStyle(IntEnum):
    MELEE = 0
    RANGED = 1
    MAGIC = 2


class Prayer(IntEnum):
    NONE = 0
    PROTECT_MELEE = 1
    PROTECT_RANGED = 2
    PROTECT_MAGIC = 3
    PIETY = 4          # +20% att, +23% str, +25% def
    RIGOUR = 5         # +20% ranged, +23% ranged str
    AUGURY = 6         # +25% magic, +25% def


class Action(IntEnum):
    # No action
    WAIT = 0
    
    # Attack styles
    ATTACK_MELEE = 1
    ATTACK_RANGED = 2
    ATTACK_MAGIC = 3
    
    # Prayers - Protection
    PRAY_MELEE = 4
    PRAY_RANGED = 5
    PRAY_MAGIC = 6
    PRAY_OFF = 7
    
    # Prayers - Offensive
    PRAY_PIETY = 8
    PRAY_RIGOUR = 9
    PRAY_AUGURY = 10
    
    # Consumables
    EAT_FOOD = 11
    DRINK_PRAYER = 12
    DRINK_RESTORE = 13  # Super restore
    DRINK_BREW = 14     # Saradomin brew
    
    # Special attack
    SPEC_ATTACK = 15
    
    # Movement
    MOVE_AWAY = 16      # Kite/dodge
    MOVE_CLOSER = 17    # Get in melee range
    MOVE_SAFE_SPOT = 18 # Move to safe tile
    
    # Gear switches
    SWITCH_MELEE_GEAR = 19
    SWITCH_RANGED_GEAR = 20
    SWITCH_MAGIC_GEAR = 21


@dataclass
class PlayerStats:
    """Player combat stats and equipment"""
    # Levels
    hitpoints: int = 99
    attack: int = 99
    strength: int = 99
    defence: int = 99
    ranged: int = 99
    magic: int = 99
    prayer: int = 99
    
    # Current gear set bonuses
    melee_attack: int = 120
    melee_strength: int = 118
    ranged_attack: int = 100
    ranged_strength: int = 95
    magic_attack: int = 65
    magic_damage: float = 0.24
    
    # Defence bonuses
    stab_def: int = 200
    slash_def: int = 210
    crush_def: int = 195
    ranged_def: int = 220
    magic_def: int = 50
    
    # Weapon speeds
    melee_speed: int = 4
    ranged_speed: int = 3
    magic_speed: int = 5
    
    # Max hits (calculated based on gear)
    melee_max: int = 50
    ranged_max: int = 40
    magic_max: int = 42


@dataclass  
class MonsterConfig:
    """Configuration for a monster/boss"""
    name: str
    combat_level: int = 100
    hitpoints: int = 100
    max_hit: int = 20
    attack_style: AttackStyle = AttackStyle.MELEE
    attack_speed: int = 4
    
    # Defence levels/bonuses
    defence_level: int = 100
    stab_def: int = 0
    slash_def: int = 0
    crush_def: int = 0
    ranged_def: int = 0
    magic_def: int = 0
    
    # Special mechanics
    phases: List[Dict] = field(default_factory=list)
    special_attacks: List[Dict] = field(default_factory=list)
    immune_to: List[AttackStyle] = field(default_factory=list)
    
    # Respawn
    respawn_ticks: int = 0  # 0 = no respawn (boss)
    
    # Drops (for reward shaping)
    avg_kill_value: int = 0


@dataclass
class PlayerState:
    """Runtime player state"""
    current_hp: int = 99
    max_hp: int = 99
    current_prayer: int = 99  
    max_prayer: int = 99
    special_attack: int = 100
    
    # Current combat mode
    attack_style: AttackStyle = AttackStyle.MELEE
    active_prayers: set = field(default_factory=set)
    
    # Gear state
    current_gear: str = "melee"  # melee, ranged, magic
    
    # Cooldowns (in ticks)
    attack_cooldown: int = 0
    eat_cooldown: int = 0
    
    # Inventory
    food_count: int = 12
    food_heal: int = 22
    prayer_pots: int = 4
    prayer_restore: int = 32  # Per dose
    super_restores: int = 4
    brews: int = 6
    brew_heal: int = 16
    
    # Position (simplified)
    distance_to_target: int = 1  # 1 = melee range
    in_safe_spot: bool = False
    
    # Status effects
    poisoned: bool = False
    poison_damage: int = 0
    venomed: bool = False
    venom_damage: int = 0
    frozen: bool = False
    freeze_ticks: int = 0
    
    # Stats boosted/drained
    attack_boost: int = 0
    strength_boost: int = 0
    defence_boost: int = 0
    ranged_boost: int = 0
    magic_boost: int = 0
    
    def reset(self, stats: PlayerStats):
        self.current_hp = stats.hitpoints
        self.max_hp = stats.hitpoints
        self.current_prayer = stats.prayer
        self.max_prayer = stats.prayer
        self.special_attack = 100
        self.attack_style = AttackStyle.MELEE
        self.active_prayers = set()
        self.current_gear = "melee"
        self.attack_cooldown = 0
        self.eat_cooldown = 0
        self.food_count = 12
        self.prayer_pots = 4
        self.super_restores = 4
        self.brews = 6
        self.distance_to_target = 1
        self.in_safe_spot = False
        self.poisoned = False
        self.poison_damage = 0
        self.venomed = False
        self.venom_damage = 0
        self.frozen = False
        self.freeze_ticks = 0
        self.attack_boost = 0
        self.strength_boost = 0
        self.defence_boost = 0
        self.ranged_boost = 0
        self.magic_boost = 0


@dataclass
class MonsterState:
    """Runtime monster state"""
    current_hp: int = 100
    max_hp: int = 100
    current_phase: int = 0
    attack_cooldown: int = 0
    special_cooldown: int = 0
    current_attack_style: AttackStyle = AttackStyle.MELEE
    next_attack_style: Optional[AttackStyle] = None  # For telegraphed attacks
    is_attackable: bool = True
    ticks_until_attackable: int = 0
    
    def reset(self, config: MonsterConfig):
        self.current_hp = config.hitpoints
        self.max_hp = config.hitpoints
        self.current_phase = 0
        self.attack_cooldown = 0
        self.special_cooldown = 0
        self.current_attack_style = config.attack_style
        self.next_attack_style = None
        self.is_attackable = True
        self.ticks_until_attackable = 0


# ============ PRESET MONSTER CONFIGS ============

MONSTERS = {
    # Training monsters
    "sand_crab": MonsterConfig(
        name="Sand Crab",
        combat_level=15,
        hitpoints=60,
        max_hit=1,
        attack_style=AttackStyle.MELEE,
        attack_speed=4,
        defence_level=1,
        respawn_ticks=50,
    ),
    "ammonite_crab": MonsterConfig(
        name="Ammonite Crab", 
        combat_level=25,
        hitpoints=100,
        max_hit=1,
        attack_style=AttackStyle.MELEE,
        attack_speed=4,
        defence_level=1,
        respawn_ticks=50,
    ),
    
    # Slayer monsters
    "abyssal_demon": MonsterConfig(
        name="Abyssal Demon",
        combat_level=124,
        hitpoints=150,
        max_hit=8,
        attack_style=AttackStyle.MELEE,
        attack_speed=4,
        defence_level=135,
        slash_def=20,
        avg_kill_value=5000,
    ),
    "gargoyle": MonsterConfig(
        name="Gargoyle",
        combat_level=111,
        hitpoints=105,
        max_hit=11,
        attack_style=AttackStyle.MELEE,
        attack_speed=4,
        defence_level=107,
        crush_def=-10,
        avg_kill_value=8000,
    ),
    
    # Bosses
    "vorkath": MonsterConfig(
        name="Vorkath",
        combat_level=732,
        hitpoints=750,
        max_hit=32,
        attack_style=AttackStyle.MAGIC,
        attack_speed=5,
        defence_level=214,
        stab_def=26,
        slash_def=108,
        crush_def=108,
        ranged_def=26,
        magic_def=240,
        special_attacks=[
            {"name": "acid_pool", "frequency": 7, "damage": 0, "effect": "move_required"},
            {"name": "freeze_breath", "frequency": 7, "damage": 30, "effect": "freeze"},
            {"name": "fireball", "frequency": 6, "damage": 121, "effect": "dodge_required"},
            {"name": "zombified_spawn", "frequency": 7, "damage": 0, "effect": "kill_spawn"},
        ],
        avg_kill_value=150000,
    ),
    "zulrah": MonsterConfig(
        name="Zulrah",
        combat_level=725,
        hitpoints=500,
        max_hit=41,
        attack_style=AttackStyle.RANGED,
        attack_speed=4,
        defence_level=300,
        stab_def=0,
        slash_def=0,
        crush_def=0,
        ranged_def=300,
        magic_def=0,
        phases=[
            {"style": AttackStyle.RANGED, "hp_threshold": 1.0, "immune_to": [AttackStyle.RANGED]},
            {"style": AttackStyle.MELEE, "hp_threshold": 0.75, "immune_to": [AttackStyle.MELEE, AttackStyle.RANGED]},
            {"style": AttackStyle.MAGIC, "hp_threshold": 0.5, "immune_to": [AttackStyle.MAGIC]},
            {"style": AttackStyle.RANGED, "hp_threshold": 0.25, "immune_to": [AttackStyle.RANGED]},
        ],
        special_attacks=[
            {"name": "venom", "frequency": 4, "damage": 0, "effect": "venom"},
            {"name": "snakeling", "frequency": 5, "damage": 15, "effect": "spawn"},
        ],
        avg_kill_value=130000,
    ),
    "corporeal_beast": MonsterConfig(
        name="Corporeal Beast",
        combat_level=785,
        hitpoints=2000,
        max_hit=51,
        attack_style=AttackStyle.MAGIC,
        attack_speed=4,
        defence_level=310,
        stab_def=100,  # Only weak to spears
        slash_def=200,
        crush_def=200,
        ranged_def=230,
        magic_def=200,
        special_attacks=[
            {"name": "dark_core", "frequency": 10, "damage": 0, "effect": "drain_prayer"},
            {"name": "stomp", "frequency": 0, "damage": 51, "effect": "under_boss"},
        ],
        avg_kill_value=400000,
    ),
    
    # Raid bosses
    "olm_head": MonsterConfig(
        name="Great Olm (Head)",
        combat_level=1043,
        hitpoints=800,  # Scales with party size
        max_hit=26,
        attack_style=AttackStyle.MAGIC,
        attack_speed=4,
        defence_level=175,
        magic_def=50,
        special_attacks=[
            {"name": "crystal_burst", "frequency": 4, "damage": 20, "effect": "move_required"},
            {"name": "lightning", "frequency": 5, "damage": 50, "effect": "targeted_move"},
            {"name": "fire_wall", "frequency": 6, "damage": 15, "effect": "run_through"},
            {"name": "falling_crystals", "frequency": 4, "damage": 25, "effect": "random_tiles"},
            {"name": "teleport", "frequency": 8, "damage": 0, "effect": "teleport_player"},
        ],
        phases=[
            {"style": AttackStyle.MAGIC, "hp_threshold": 1.0},
            {"style": AttackStyle.RANGED, "hp_threshold": 0.5},
        ],
        avg_kill_value=500000,
    ),
    "verzik_p3": MonsterConfig(
        name="Verzik Vitur (P3)",
        combat_level=1040,
        hitpoints=2500,
        max_hit=78,
        attack_style=AttackStyle.MELEE,
        attack_speed=4,
        defence_level=200,
        special_attacks=[
            {"name": "web", "frequency": 6, "damage": 0, "effect": "stun"},
            {"name": "yellow_pool", "frequency": 4, "damage": 100, "effect": "stack"},
            {"name": "green_ball", "frequency": 5, "damage": 75, "effect": "bounce"},
            {"name": "purple_tornado", "frequency": 8, "damage": 30, "effect": "chase"},
        ],
        avg_kill_value=800000,
    ),
}


class OSRSPvEEnv(gym.Env):
    """
    OSRS PvE Combat Environment
    
    Features:
    - Multiple monster/boss configurations
    - Phase-based fights
    - Special attack mechanics
    - Prayer switching
    - Gear switching
    - Positioning
    - Resource management (food, prayer, specs)
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 2}
    
    def __init__(
        self,
        monster: str = "vorkath",
        player_stats: Optional[PlayerStats] = None,
        max_ticks: int = 500,
        render_mode: Optional[str] = None,
        reward_shaping: bool = True,
    ):
        super().__init__()
        
        # Get monster config
        if monster in MONSTERS:
            self.monster_config = MONSTERS[monster]
        else:
            raise ValueError(f"Unknown monster: {monster}. Available: {list(MONSTERS.keys())}")
        
        self.player_stats = player_stats or PlayerStats()
        self.max_ticks = max_ticks
        self.render_mode = render_mode
        self.reward_shaping = reward_shaping
        
        # Initialize states
        self.player = PlayerState()
        self.monster = MonsterState()
        
        # Tracking
        self.tick = 0
        self.total_damage_dealt = 0
        self.total_damage_taken = 0
        self.kills = 0
        self.deaths = 0
        
        # Observation space: 32 continuous values
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(32,), dtype=np.float32
        )
        
        # Action space: 22 discrete actions
        self.action_space = spaces.Discrete(22)
        
        # History for pattern recognition
        self.monster_attack_history: List[AttackStyle] = []
        self.max_history = 10
    
    def _get_obs(self) -> np.ndarray:
        """Build observation vector"""
        obs = np.zeros(32, dtype=np.float32)
        
        # Player stats (0-5)
        obs[0] = self.player.current_hp / self.player.max_hp
        obs[1] = self.player.current_prayer / self.player.max_prayer
        obs[2] = self.player.special_attack / 100.0
        obs[3] = self.player.food_count / 12.0
        obs[4] = (self.player.prayer_pots + self.player.super_restores) / 8.0
        obs[5] = self.player.brews / 6.0
        
        # Player cooldowns (6-7)
        obs[6] = min(self.player.attack_cooldown / 6.0, 1.0)
        obs[7] = min(self.player.eat_cooldown / 3.0, 1.0)
        
        # Player combat state one-hot (8-10)
        obs[8 + int(self.player.attack_style)] = 1.0
        
        # Active protection prayer one-hot (11-14)
        if Prayer.PROTECT_MELEE in self.player.active_prayers:
            obs[12] = 1.0
        elif Prayer.PROTECT_RANGED in self.player.active_prayers:
            obs[13] = 1.0
        elif Prayer.PROTECT_MAGIC in self.player.active_prayers:
            obs[14] = 1.0
        else:
            obs[11] = 1.0  # No protection
        
        # Offensive prayer active (15)
        obs[15] = 1.0 if any(p in self.player.active_prayers for p in [Prayer.PIETY, Prayer.RIGOUR, Prayer.AUGURY]) else 0.0
        
        # Player position (16-17)
        obs[16] = min(self.player.distance_to_target / 10.0, 1.0)
        obs[17] = 1.0 if self.player.in_safe_spot else 0.0
        
        # Player status effects (18-20)
        obs[18] = 1.0 if self.player.poisoned or self.player.venomed else 0.0
        obs[19] = 1.0 if self.player.frozen else 0.0
        obs[20] = min(self.player.freeze_ticks / 10.0, 1.0)
        
        # Monster stats (21-24)
        obs[21] = self.monster.current_hp / self.monster.max_hp
        obs[22] = self.monster.current_phase / max(len(self.monster_config.phases), 1)
        obs[23] = 1.0 if self.monster.is_attackable else 0.0
        obs[24] = min(self.monster.attack_cooldown / 6.0, 1.0)
        
        # Monster current attack style one-hot (25-27)
        obs[25 + int(self.monster.current_attack_style)] = 1.0
        
        # Monster telegraphed next attack (28-30)
        if self.monster.next_attack_style is not None:
            obs[28 + int(self.monster.next_attack_style)] = 1.0
        
        # Tick in fight (31)
        obs[31] = min(self.tick / self.max_ticks, 1.0)
        
        return obs
    
    def _get_info(self) -> Dict:
        """Get info dict"""
        return {
            "tick": self.tick,
            "player_hp": self.player.current_hp,
            "monster_hp": self.monster.current_hp,
            "damage_dealt": self.total_damage_dealt,
            "damage_taken": self.total_damage_taken,
            "kills": self.kills,
            "deaths": self.deaths,
            "monster_phase": self.monster.current_phase,
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player.reset(self.player_stats)
        self.monster.reset(self.monster_config)
        
        self.tick = 0
        self.total_damage_dealt = 0
        self.total_damage_taken = 0
        self.monster_attack_history = []
        
        return self._get_obs(), self._get_info()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one game tick"""
        self.tick += 1
        reward = 0.0
        terminated = False
        truncated = False
        
        # 1. Process player action
        reward += self._process_player_action(Action(action))
        
        # 2. Process monster action
        reward += self._process_monster_action()
        
        # 3. Update cooldowns and status effects
        self._update_tick()
        
        # 4. Check phase transitions
        self._check_phase_transition()
        
        # 5. Check termination conditions
        if self.player.current_hp <= 0:
            terminated = True
            self.deaths += 1
            reward -= 100.0  # Death penalty
        
        if self.monster.current_hp <= 0:
            self.kills += 1
            reward += 50.0  # Kill reward
            if self.reward_shaping:
                reward += self.monster_config.avg_kill_value / 10000.0  # GP bonus
            # Reset monster for next kill (if farming)
            if self.monster_config.respawn_ticks > 0:
                self.monster.reset(self.monster_config)
            else:
                terminated = True  # Boss kill ends episode
        
        if self.tick >= self.max_ticks:
            truncated = True
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _process_player_action(self, action: Action) -> float:
        """Process player's action, return reward"""
        reward = 0.0
        
        # Decrement cooldowns
        if self.player.attack_cooldown > 0:
            self.player.attack_cooldown -= 1
        if self.player.eat_cooldown > 0:
            self.player.eat_cooldown -= 1
        
        # Handle action
        if action == Action.WAIT:
            pass
        
        # Attack actions
        elif action in [Action.ATTACK_MELEE, Action.ATTACK_RANGED, Action.ATTACK_MAGIC]:
            style = AttackStyle(action.value - 1)
            reward += self._player_attack(style)
        
        # Protection prayers
        elif action == Action.PRAY_MELEE:
            self._set_protection_prayer(Prayer.PROTECT_MELEE)
        elif action == Action.PRAY_RANGED:
            self._set_protection_prayer(Prayer.PROTECT_RANGED)
        elif action == Action.PRAY_MAGIC:
            self._set_protection_prayer(Prayer.PROTECT_MAGIC)
        elif action == Action.PRAY_OFF:
            self.player.active_prayers.discard(Prayer.PROTECT_MELEE)
            self.player.active_prayers.discard(Prayer.PROTECT_RANGED)
            self.player.active_prayers.discard(Prayer.PROTECT_MAGIC)
        
        # Offensive prayers
        elif action == Action.PRAY_PIETY:
            self._set_offensive_prayer(Prayer.PIETY)
        elif action == Action.PRAY_RIGOUR:
            self._set_offensive_prayer(Prayer.RIGOUR)
        elif action == Action.PRAY_AUGURY:
            self._set_offensive_prayer(Prayer.AUGURY)
        
        # Consumables
        elif action == Action.EAT_FOOD:
            reward += self._eat_food()
        elif action == Action.DRINK_PRAYER:
            self._drink_prayer_pot()
        elif action == Action.DRINK_RESTORE:
            self._drink_restore()
        elif action == Action.DRINK_BREW:
            self._drink_brew()
        
        # Special attack
        elif action == Action.SPEC_ATTACK:
            reward += self._special_attack()
        
        # Movement
        elif action == Action.MOVE_AWAY:
            self.player.distance_to_target = min(10, self.player.distance_to_target + 2)
        elif action == Action.MOVE_CLOSER:
            self.player.distance_to_target = max(1, self.player.distance_to_target - 2)
        elif action == Action.MOVE_SAFE_SPOT:
            self.player.in_safe_spot = True
        
        # Gear switches
        elif action == Action.SWITCH_MELEE_GEAR:
            self.player.current_gear = "melee"
            self.player.attack_style = AttackStyle.MELEE
        elif action == Action.SWITCH_RANGED_GEAR:
            self.player.current_gear = "ranged"
            self.player.attack_style = AttackStyle.RANGED
        elif action == Action.SWITCH_MAGIC_GEAR:
            self.player.current_gear = "magic"
            self.player.attack_style = AttackStyle.MAGIC
        
        return reward
    
    def _player_attack(self, style: AttackStyle) -> float:
        """Execute player attack, return reward"""
        reward = 0.0
        
        # Check cooldown
        if self.player.attack_cooldown > 0:
            return -0.1  # Penalty for wasted action
        
        # Check if monster is attackable
        if not self.monster.is_attackable:
            return -0.1
        
        # Check range
        if style == AttackStyle.MELEE and self.player.distance_to_target > 1:
            return -0.1  # Out of melee range
        
        # Check immunity
        if style in self.monster_config.immune_to:
            return -0.5  # Big penalty for attacking immune phase
        
        # Check current phase immunity
        if self.monster_config.phases:
            phase = self.monster_config.phases[self.monster.current_phase]
            if style in phase.get("immune_to", []):
                return -0.5
        
        # Calculate hit
        damage = self._calculate_player_damage(style)
        self.monster.current_hp -= damage
        self.total_damage_dealt += damage
        
        # Set cooldown based on weapon speed
        if style == AttackStyle.MELEE:
            self.player.attack_cooldown = self.player_stats.melee_speed
        elif style == AttackStyle.RANGED:
            self.player.attack_cooldown = self.player_stats.ranged_speed
        else:
            self.player.attack_cooldown = self.player_stats.magic_speed
        
        # Reward for damage
        if self.reward_shaping:
            reward += damage / 10.0
        
        return reward
    
    def _calculate_player_damage(self, style: AttackStyle) -> int:
        """Calculate player damage (simplified)"""
        # Get max hit based on style
        if style == AttackStyle.MELEE:
            max_hit = self.player_stats.melee_max
        elif style == AttackStyle.RANGED:
            max_hit = self.player_stats.ranged_max
        else:
            max_hit = self.player_stats.magic_max
        
        # Apply offensive prayer bonus
        if Prayer.PIETY in self.player.active_prayers and style == AttackStyle.MELEE:
            max_hit = int(max_hit * 1.23)
        elif Prayer.RIGOUR in self.player.active_prayers and style == AttackStyle.RANGED:
            max_hit = int(max_hit * 1.23)
        elif Prayer.AUGURY in self.player.active_prayers and style == AttackStyle.MAGIC:
            max_hit = int(max_hit * 1.04)  # Augury doesn't boost magic damage much
        
        # Roll damage (simplified - uniform distribution)
        if self.np_random.random() < 0.85:  # 85% hit chance (simplified)
            return self.np_random.integers(0, max_hit + 1)
        return 0
    
    def _special_attack(self) -> float:
        """Execute special attack"""
        if self.player.special_attack < 50:
            return -0.1  # Not enough spec
        
        if self.player.attack_cooldown > 0:
            return -0.1
        
        if not self.monster.is_attackable:
            return -0.1
        
        self.player.special_attack -= 50
        
        # High damage spec (like AGS)
        max_hit = int(self.player_stats.melee_max * 1.375)  # AGS modifier
        
        if self.np_random.random() < 0.90:  # Higher accuracy on spec
            damage = self.np_random.integers(0, max_hit + 1)
        else:
            damage = 0
        
        self.monster.current_hp -= damage
        self.total_damage_dealt += damage
        self.player.attack_cooldown = 6  # Spec recovery
        
        return damage / 5.0 if self.reward_shaping else 0.0
    
    def _process_monster_action(self) -> float:
        """Process monster's action, return reward (negative for damage taken)"""
        reward = 0.0
        
        # Decrement cooldowns
        if self.monster.attack_cooldown > 0:
            self.monster.attack_cooldown -= 1
            return reward
        
        # Check for special attacks
        for spec in self.monster_config.special_attacks:
            if self.np_random.random() < 1.0 / spec["frequency"]:
                reward += self._execute_monster_special(spec)
                return reward
        
        # Normal attack
        if self.player.in_safe_spot:
            return reward  # Safe spotted, no damage
        
        # Determine attack style (may change per phase)
        attack_style = self.monster.current_attack_style
        
        # Calculate damage
        damage = self._calculate_monster_damage(attack_style)
        
        # Check protection prayer
        protection_map = {
            AttackStyle.MELEE: Prayer.PROTECT_MELEE,
            AttackStyle.RANGED: Prayer.PROTECT_RANGED,
            AttackStyle.MAGIC: Prayer.PROTECT_MAGIC,
        }
        
        correct_prayer = protection_map[attack_style]
        if correct_prayer in self.player.active_prayers:
            damage = 0  # Full protection in PvE
            if self.reward_shaping:
                reward += 0.5  # Reward for correct prayer
        else:
            if self.reward_shaping and damage > 0:
                reward -= 0.3  # Penalty for wrong prayer
        
        self.player.current_hp -= damage
        self.total_damage_taken += damage
        
        # Record attack for pattern recognition
        self.monster_attack_history.append(attack_style)
        if len(self.monster_attack_history) > self.max_history:
            self.monster_attack_history.pop(0)
        
        # Set monster attack cooldown
        self.monster.attack_cooldown = self.monster_config.attack_speed
        
        # Negative reward for damage taken
        if self.reward_shaping:
            reward -= damage / 20.0
        
        return reward
    
    def _calculate_monster_damage(self, style: AttackStyle) -> int:
        """Calculate monster damage"""
        max_hit = self.monster_config.max_hit
        
        if self.np_random.random() < 0.80:  # 80% hit chance
            return self.np_random.integers(0, max_hit + 1)
        return 0
    
    def _execute_monster_special(self, spec: Dict) -> float:
        """Execute monster special attack"""
        reward = 0.0
        effect = spec.get("effect", "")
        damage = spec.get("damage", 0)
        
        if effect == "move_required":
            # Player needs to move or take damage
            if not self.player.in_safe_spot:
                self.player.current_hp -= damage
                self.total_damage_taken += damage
                reward -= damage / 20.0
        
        elif effect == "freeze":
            self.player.frozen = True
            self.player.freeze_ticks = 5
            self.player.current_hp -= damage
            self.total_damage_taken += damage
        
        elif effect == "venom":
            self.player.venomed = True
            self.player.venom_damage = 6
        
        elif effect == "drain_prayer":
            drain = self.np_random.integers(5, 20)
            self.player.current_prayer = max(0, self.player.current_prayer - drain)
        
        elif effect == "dodge_required":
            # Telegraphed attack - can be dodged
            self.monster.next_attack_style = AttackStyle.MAGIC
            # Damage applied next tick if player doesn't move
        
        return reward
    
    def _set_protection_prayer(self, prayer: Prayer):
        """Set protection prayer (only one at a time)"""
        self.player.active_prayers.discard(Prayer.PROTECT_MELEE)
        self.player.active_prayers.discard(Prayer.PROTECT_RANGED)
        self.player.active_prayers.discard(Prayer.PROTECT_MAGIC)
        self.player.active_prayers.add(prayer)
    
    def _set_offensive_prayer(self, prayer: Prayer):
        """Set offensive prayer (only one at a time)"""
        self.player.active_prayers.discard(Prayer.PIETY)
        self.player.active_prayers.discard(Prayer.RIGOUR)
        self.player.active_prayers.discard(Prayer.AUGURY)
        self.player.active_prayers.add(prayer)
    
    def _eat_food(self) -> float:
        """Eat food"""
        if self.player.food_count <= 0:
            return -0.1
        if self.player.eat_cooldown > 0:
            return -0.1
        
        heal = min(self.player.food_heal, self.player.max_hp - self.player.current_hp)
        self.player.current_hp += heal
        self.player.food_count -= 1
        self.player.eat_cooldown = 3
        
        return 0.0
    
    def _drink_prayer_pot(self):
        """Drink prayer potion"""
        if self.player.prayer_pots <= 0:
            return
        
        restore = min(self.player.prayer_restore, self.player.max_prayer - self.player.current_prayer)
        self.player.current_prayer += restore
        self.player.prayer_pots -= 1
    
    def _drink_restore(self):
        """Drink super restore"""
        if self.player.super_restores <= 0:
            return
        
        # Restores prayer and stats
        restore = min(32, self.player.max_prayer - self.player.current_prayer)
        self.player.current_prayer += restore
        self.player.super_restores -= 1
    
    def _drink_brew(self):
        """Drink saradomin brew"""
        if self.player.brews <= 0:
            return
        
        # Heals but drains stats
        heal = min(self.player.brew_heal, self.player.max_hp - self.player.current_hp)
        self.player.current_hp += heal
        
        # Also boosts defence and drains attack/strength/magic/ranged
        self.player.brews -= 1
    
    def _update_tick(self):
        """Update per-tick effects"""
        # Prayer drain
        drain = 0
        for prayer in self.player.active_prayers:
            if prayer in [Prayer.PROTECT_MELEE, Prayer.PROTECT_RANGED, Prayer.PROTECT_MAGIC]:
                drain += 1
            elif prayer in [Prayer.PIETY, Prayer.RIGOUR, Prayer.AUGURY]:
                drain += 2
        
        self.player.current_prayer = max(0, self.player.current_prayer - drain // 3)
        
        # Disable prayers if out of points
        if self.player.current_prayer <= 0:
            self.player.active_prayers.clear()
        
        # Poison/venom damage
        if self.player.venomed:
            self.player.current_hp -= self.player.venom_damage
            self.total_damage_taken += self.player.venom_damage
            self.player.venom_damage = min(20, self.player.venom_damage + 2)
        elif self.player.poisoned:
            self.player.current_hp -= self.player.poison_damage
            self.total_damage_taken += self.player.poison_damage
        
        # Freeze duration
        if self.player.frozen:
            self.player.freeze_ticks -= 1
            if self.player.freeze_ticks <= 0:
                self.player.frozen = False
        
        # Spec regeneration (1% every 30 ticks)
        if self.tick % 30 == 0:
            self.player.special_attack = min(100, self.player.special_attack + 10)
    
    def _check_phase_transition(self):
        """Check if monster should transition to new phase"""
        if not self.monster_config.phases:
            return
        
        hp_ratio = self.monster.current_hp / self.monster.max_hp
        
        for i, phase in enumerate(self.monster_config.phases):
            if hp_ratio <= phase.get("hp_threshold", 1.0):
                if i > self.monster.current_phase:
                    self.monster.current_phase = i
                    self.monster.current_attack_style = phase.get("style", AttackStyle.MAGIC)
                    # Brief invulnerability during phase transition
                    self.monster.is_attackable = False
                    self.monster.ticks_until_attackable = 3
    
    def render(self):
        """Render environment state"""
        if self.render_mode == "human" or self.render_mode == "ansi":
            print(f"\n=== Tick {self.tick} ===")
            print(f"Player: {self.player.current_hp}/{self.player.max_hp} HP | "
                  f"{self.player.current_prayer}/{self.player.max_prayer} Prayer | "
                  f"{self.player.special_attack}% Spec")
            print(f"  Food: {self.player.food_count} | Brews: {self.player.brews} | Restores: {self.player.super_restores}")
            print(f"  Prayer: {[p.name for p in self.player.active_prayers]}")
            print(f"  Distance: {self.player.distance_to_target} | Safe: {self.player.in_safe_spot}")
            print(f"Monster: {self.monster.current_hp}/{self.monster.max_hp} HP | "
                  f"Phase {self.monster.current_phase} | "
                  f"Style: {self.monster.current_attack_style.name}")
            print(f"Damage dealt: {self.total_damage_dealt} | Taken: {self.total_damage_taken}")


# Register environments
def register_envs():
    """Register all monster environments"""
    from gymnasium.envs.registration import register
    
    for monster_name in MONSTERS.keys():
        env_id = f"OSRS-{monster_name.replace('_', '-').title()}-v0"
        try:
            register(
                id=env_id,
                entry_point="osrs_rl.pve_env:OSRSPvEEnv",
                kwargs={"monster": monster_name},
            )
        except:
            pass  # Already registered


if __name__ == "__main__":
    # Test the environment
    env = OSRSPvEEnv(monster="vorkath", render_mode="human")
    obs, info = env.reset()
    
    print("Testing Vorkath environment...")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    total_reward = 0
    for _ in range(50):
        # Random action for testing
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    env.render()
    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Final info: {info}")
