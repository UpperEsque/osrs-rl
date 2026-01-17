"""
ToA Wardens Environment V2 - Improved Reward Shaping & Mechanic Tracking

Changes from V1:
- Dense rewards for mechanic handling
- Per-phase completion bonuses
- Survival milestones
- Detailed death cause tracking
- Invocation scaling support for curriculum
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import IntEnum
import random


class WardensPhase(IntEnum):
    P1_ELIDINIS = 0
    P2_TUMEKEN = 1  
    P3_ENRAGED = 2


class WardensAction(IntEnum):
    # Movement
    WAIT = 0
    MOVE_NORTH = 1
    MOVE_SOUTH = 2
    MOVE_EAST = 3
    MOVE_WEST = 4
    MOVE_TO_SAFE = 5  # Move to safe tile
    
    # Combat
    ATTACK_RANGED = 6
    ATTACK_MAGE = 7
    ATTACK_SPEC = 8
    
    # Prayer
    PRAY_MAGE = 9
    PRAY_RANGE = 10
    PRAY_MELEE = 11
    PRAY_OFF = 12
    TOGGLE_RIGOUR = 13
    TOGGLE_AUGURY = 14
    
    # Consumables
    EAT_FOOD = 15
    DRINK_BREW = 16
    DRINK_RESTORE = 17
    
    # Mechanics
    ATTACK_CORE = 18  # Attack the core when it spawns
    DODGE_SLAM = 19   # Explicit dodge action


@dataclass
class WardensState:
    """State of the Wardens boss"""
    current_hp: int = 880  # Base HP, scales with invocation
    max_hp: int = 880
    phase: WardensPhase = WardensPhase.P1_ELIDINIS
    
    # Attack patterns
    attack_tick: int = 0
    next_attack: str = "auto"  # auto, slam, lightning, divine
    
    # Core mechanic
    core_active: bool = False
    core_hp: int = 0
    core_max_hp: int = 60
    core_ticks_remaining: int = 0
    
    # Lightning/special mechanics
    lightning_active: bool = False
    lightning_tiles: List[Tuple[int, int]] = field(default_factory=list)
    slam_incoming: bool = False
    slam_ticks: int = 0
    divine_projectile: str = "none"  # none, mage, range
    
    # Phase thresholds (% of max HP)
    p2_threshold: float = 0.66
    p3_threshold: float = 0.33
    
    def reset(self, invocation: int = 0):
        """Reset with invocation scaling"""
        # HP scales with invocation
        hp_multiplier = 1.0 + (invocation / 600) * 0.25  # +25% at 600
        self.max_hp = int(880 * hp_multiplier)
        self.current_hp = self.max_hp
        self.phase = WardensPhase.P1_ELIDINIS
        self.attack_tick = 0
        self.next_attack = "auto"
        self.core_active = False
        self.core_hp = 0
        self.core_ticks_remaining = 0
        self.lightning_active = False
        self.lightning_tiles = []
        self.slam_incoming = False
        self.slam_ticks = 0
        self.divine_projectile = "none"


@dataclass 
class PlayerState:
    """Player state in ToA"""
    current_hp: int = 99
    max_hp: int = 99
    current_prayer: int = 99
    max_prayer: int = 99
    special_attack: int = 100
    
    # Position (simplified 5x5 grid)
    position: Tuple[int, int] = (2, 2)
    
    # Prayer state
    protect_mage: bool = False
    protect_range: bool = False
    protect_melee: bool = False
    rigour: bool = False
    augury: bool = False
    
    # Cooldowns
    attack_cooldown: int = 0
    eat_cooldown: int = 0
    
    # Resources
    food_count: int = 12
    brew_doses: int = 8
    restore_doses: int = 8
    
    # Combat stats
    ranged_max_hit: int = 55
    mage_max_hit: int = 50
    
    def reset(self, invocation: int = 0):
        """Reset with invocation penalties"""
        self.current_hp = self.max_hp
        self.current_prayer = self.max_prayer
        self.special_attack = 100
        self.position = (2, 2)
        self.protect_mage = False
        self.protect_range = False
        self.protect_melee = False
        self.rigour = False
        self.augury = False
        self.attack_cooldown = 0
        self.eat_cooldown = 0
        
        # Resources scale down with invocation
        resource_mult = 1.0 - (invocation / 600) * 0.3  # -30% at 600
        self.food_count = int(12 * resource_mult)
        self.brew_doses = int(8 * resource_mult)
        self.restore_doses = int(8 * resource_mult)


class WardensEnvV2(gym.Env):
    """
    ToA Wardens Environment V2
    
    Improvements:
    - Dense reward shaping
    - Mechanic-specific tracking
    - Invocation scaling
    - Survival milestones
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 2}
    
    # Damage constants (scale with invocation)
    BASE_AUTO_DAMAGE = 25
    BASE_SLAM_DAMAGE = 60
    BASE_LIGHTNING_DAMAGE = 40
    BASE_CORE_EXPLOSION = 80
    DIVINE_DAMAGE = 35  # If wrong prayer
    
    # Mechanic timings
    SLAM_WINDUP_TICKS = 3
    CORE_DURATION_TICKS = 12
    LIGHTNING_DURATION = 4
    
    def __init__(
        self,
        invocation: int = 0,
        max_ticks: int = 300,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.invocation = invocation
        self.max_ticks = max_ticks
        self.render_mode = render_mode
        
        # Calculate invocation modifiers
        self.damage_mult = 1.0 + (invocation / 600) * 0.3  # +30% damage at 600
        self.prayer_drain_mult = 1.0 + (invocation / 600) * 2.0  # 3x drain at 600
        self.heal_reduction = (invocation / 600) * 0.25  # -25% healing at 600
        
        # Initialize states
        self.wardens = WardensState()
        self.player = PlayerState()
        self.rng = np.random.default_rng()
        
        # Tick counter
        self.tick = 0
        
        # Tracking metrics
        self.metrics = {
            "damage_dealt": 0,
            "damage_taken": 0,
            "slams_dodged": 0,
            "slams_hit": 0,
            "cores_killed": 0,
            "cores_exploded": 0,
            "lightning_dodged": 0,
            "lightning_hit": 0,
            "prayers_correct": 0,
            "prayers_wrong": 0,
            "phases_completed": 0,
            "death_cause": None,
        }
        
        # Observation: 40 values
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(40,), dtype=np.float32
        )
        
        # Actions
        self.action_space = spaces.Discrete(len(WardensAction))
    
    def _get_obs(self) -> np.ndarray:
        """Build observation vector"""
        obs = np.zeros(40, dtype=np.float32)
        
        # Player stats (0-7)
        obs[0] = self.player.current_hp / self.player.max_hp
        obs[1] = self.player.current_prayer / self.player.max_prayer
        obs[2] = self.player.special_attack / 100.0
        obs[3] = self.player.attack_cooldown / 5.0
        obs[4] = self.player.food_count / 12.0
        obs[5] = self.player.brew_doses / 8.0
        obs[6] = self.player.restore_doses / 8.0
        obs[7] = self.player.eat_cooldown / 3.0
        
        # Player prayer state (8-12)
        obs[8] = float(self.player.protect_mage)
        obs[9] = float(self.player.protect_range)
        obs[10] = float(self.player.protect_melee)
        obs[11] = float(self.player.rigour)
        obs[12] = float(self.player.augury)
        
        # Player position (13-14)
        obs[13] = self.player.position[0] / 4.0
        obs[14] = self.player.position[1] / 4.0
        
        # Wardens state (15-22)
        obs[15] = self.wardens.current_hp / self.wardens.max_hp
        obs[16] = float(self.wardens.phase) / 2.0
        obs[17] = float(self.wardens.core_active)
        obs[18] = self.wardens.core_hp / self.wardens.core_max_hp if self.wardens.core_active else 0
        obs[19] = self.wardens.core_ticks_remaining / self.CORE_DURATION_TICKS if self.wardens.core_active else 0
        obs[20] = float(self.wardens.slam_incoming)
        obs[21] = self.wardens.slam_ticks / self.SLAM_WINDUP_TICKS if self.wardens.slam_incoming else 0
        obs[22] = float(self.wardens.lightning_active)
        
        # Divine projectile encoding (23-25)
        obs[23] = float(self.wardens.divine_projectile == "mage")
        obs[24] = float(self.wardens.divine_projectile == "range")
        obs[25] = float(self.wardens.divine_projectile == "none")
        
        # Lightning tile danger (26-30) - simplified
        if self.wardens.lightning_active:
            for i, (lx, ly) in enumerate(self.wardens.lightning_tiles[:5]):
                px, py = self.player.position
                dist = abs(lx - px) + abs(ly - py)
                obs[26 + i] = max(0, 1.0 - dist / 4.0)  # Closer = higher danger
        
        # Attack pattern prediction (31-35)
        attack_types = ["auto", "slam", "lightning", "divine", "core"]
        for i, atype in enumerate(attack_types):
            obs[31 + i] = float(self.wardens.next_attack == atype)
        
        # Invocation/difficulty (36)
        obs[36] = self.invocation / 600.0
        
        # Tick progress (37)
        obs[37] = self.tick / self.max_ticks
        
        # Safe tile indicator (38)
        obs[38] = float(self._is_safe_position(self.player.position))
        
        # Phase progress within current phase (39)
        if self.wardens.phase == WardensPhase.P1_ELIDINIS:
            phase_hp_start = self.wardens.max_hp
            phase_hp_end = self.wardens.max_hp * self.wardens.p2_threshold
        elif self.wardens.phase == WardensPhase.P2_TUMEKEN:
            phase_hp_start = self.wardens.max_hp * self.wardens.p2_threshold
            phase_hp_end = self.wardens.max_hp * self.wardens.p3_threshold
        else:
            phase_hp_start = self.wardens.max_hp * self.wardens.p3_threshold
            phase_hp_end = 0
        
        phase_progress = (phase_hp_start - self.wardens.current_hp) / (phase_hp_start - phase_hp_end + 1)
        obs[39] = min(1.0, max(0.0, phase_progress))
        
        return obs
    
    def _is_safe_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is safe from lightning"""
        if not self.wardens.lightning_active:
            return True
        return pos not in self.wardens.lightning_tiles
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.wardens.reset(self.invocation)
        self.player.reset(self.invocation)
        self.tick = 0
        
        # Reset metrics
        self.metrics = {
            "damage_dealt": 0,
            "damage_taken": 0,
            "slams_dodged": 0,
            "slams_hit": 0,
            "cores_killed": 0,
            "cores_exploded": 0,
            "lightning_dodged": 0,
            "lightning_hit": 0,
            "prayers_correct": 0,
            "prayers_wrong": 0,
            "phases_completed": 0,
            "death_cause": None,
        }
        
        return self._get_obs(), {}
    
    def step(self, action: int):
        action = WardensAction(action)
        self.tick += 1
        
        reward = 0.0
        info = dict(self.metrics)
        
        # Process player action
        action_reward = self._process_player_action(action)
        reward += action_reward
        
        # Process wardens turn
        wardens_reward = self._process_wardens_turn()
        reward += wardens_reward
        
        # Drain prayer (scales with invocation)
        prayer_drain = int(1 * self.prayer_drain_mult)
        if self.player.rigour:
            prayer_drain += int(2 * self.prayer_drain_mult)
        if self.player.augury:
            prayer_drain += int(2 * self.prayer_drain_mult)
        if self.player.protect_mage or self.player.protect_range or self.player.protect_melee:
            prayer_drain += int(1 * self.prayer_drain_mult)
        
        self.player.current_prayer = max(0, self.player.current_prayer - prayer_drain)
        
        # Disable prayers if no prayer points
        if self.player.current_prayer <= 0:
            self.player.protect_mage = False
            self.player.protect_range = False
            self.player.protect_melee = False
            self.player.rigour = False
            self.player.augury = False
        
        # Check phase transitions
        old_phase = self.wardens.phase
        self._check_phase_transition()
        if self.wardens.phase != old_phase:
            reward += 30.0  # Phase completion bonus
            self.metrics["phases_completed"] += 1
        
        # Survival milestones
        if self.tick == 50:
            reward += 5.0
        elif self.tick == 100:
            reward += 10.0
        elif self.tick == 150:
            reward += 15.0
        
        # Check termination
        done = False
        truncated = False
        
        if self.player.current_hp <= 0:
            done = True
            reward -= 100.0
            info["result"] = "death"
            info["death_cause"] = self.metrics["death_cause"]
        elif self.wardens.current_hp <= 0:
            done = True
            reward += 150.0
            info["result"] = "kill"
        elif self.tick >= self.max_ticks:
            truncated = True
            reward -= 50.0  # Timeout penalty
            info["result"] = "timeout"
        
        # Update info
        info.update(self.metrics)
        info["tick"] = self.tick
        info["phase"] = self.wardens.phase.name
        info["invocation"] = self.invocation
        
        return self._get_obs(), reward, done, truncated, info
    
    def _process_player_action(self, action: WardensAction) -> float:
        """Process player action and return reward"""
        reward = 0.0
        
        # Reduce cooldowns
        if self.player.attack_cooldown > 0:
            self.player.attack_cooldown -= 1
        if self.player.eat_cooldown > 0:
            self.player.eat_cooldown -= 1
        
        # Movement actions
        if action in [WardensAction.MOVE_NORTH, WardensAction.MOVE_SOUTH,
                      WardensAction.MOVE_EAST, WardensAction.MOVE_WEST,
                      WardensAction.MOVE_TO_SAFE]:
            reward += self._handle_movement(action)
        
        # Attack actions
        elif action == WardensAction.ATTACK_RANGED:
            reward += self._handle_attack("ranged")
        elif action == WardensAction.ATTACK_MAGE:
            reward += self._handle_attack("mage")
        elif action == WardensAction.ATTACK_SPEC:
            reward += self._handle_spec_attack()
        
        # Attack core
        elif action == WardensAction.ATTACK_CORE:
            reward += self._handle_core_attack()
        
        # Dodge slam (explicit)
        elif action == WardensAction.DODGE_SLAM:
            reward += self._handle_dodge()
        
        # Prayer actions
        elif action in [WardensAction.PRAY_MAGE, WardensAction.PRAY_RANGE,
                        WardensAction.PRAY_MELEE, WardensAction.PRAY_OFF,
                        WardensAction.TOGGLE_RIGOUR, WardensAction.TOGGLE_AUGURY]:
            reward += self._handle_prayer(action)
        
        # Consumables
        elif action == WardensAction.EAT_FOOD:
            reward += self._handle_eat()
        elif action == WardensAction.DRINK_BREW:
            reward += self._handle_brew()
        elif action == WardensAction.DRINK_RESTORE:
            reward += self._handle_restore()
        
        return reward
    
    def _handle_movement(self, action: WardensAction) -> float:
        """Handle movement and return reward"""
        reward = 0.0
        x, y = self.player.position
        
        if action == WardensAction.MOVE_NORTH:
            y = min(4, y + 1)
        elif action == WardensAction.MOVE_SOUTH:
            y = max(0, y - 1)
        elif action == WardensAction.MOVE_EAST:
            x = min(4, x + 1)
        elif action == WardensAction.MOVE_WEST:
            x = max(0, x - 1)
        elif action == WardensAction.MOVE_TO_SAFE:
            # Find nearest safe tile
            if self.wardens.lightning_active:
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx <= 4 and 0 <= ny <= 4:
                            if (nx, ny) not in self.wardens.lightning_tiles:
                                x, y = nx, ny
                                break
        
        old_safe = self._is_safe_position(self.player.position)
        self.player.position = (x, y)
        new_safe = self._is_safe_position(self.player.position)
        
        # Reward for moving to safety
        if not old_safe and new_safe:
            reward += 2.0
            self.metrics["lightning_dodged"] += 1
        
        # Reward for dodging slam by moving
        if self.wardens.slam_incoming and self.wardens.slam_ticks == 1:
            reward += 3.0  # Moved during slam windup
            self.metrics["slams_dodged"] += 1
        
        return reward
    
    def _handle_attack(self, style: str) -> float:
        """Handle attack and return reward"""
        if self.player.attack_cooldown > 0:
            return -0.1  # Small penalty for invalid action
        
        if self.wardens.core_active:
            return -0.5  # Should be attacking core
        
        reward = 0.0
        
        # Calculate damage
        if style == "ranged":
            max_hit = self.player.ranged_max_hit
            if self.player.rigour:
                max_hit = int(max_hit * 1.23)
            self.player.attack_cooldown = 3
        else:  # mage
            max_hit = self.player.mage_max_hit
            if self.player.augury:
                max_hit = int(max_hit * 1.04)
            self.player.attack_cooldown = 4
        
        # Hit chance (simplified)
        if self.rng.random() < 0.85:
            damage = self.rng.integers(0, max_hit + 1)
            self.wardens.current_hp -= damage
            self.metrics["damage_dealt"] += damage
            reward += damage * 0.1  # Reward per damage
        
        return reward
    
    def _handle_spec_attack(self) -> float:
        """Handle special attack"""
        if self.player.special_attack < 50:
            return -0.1
        if self.player.attack_cooldown > 0:
            return -0.1
        
        self.player.special_attack -= 50
        self.player.attack_cooldown = 5
        
        # High damage spec
        if self.rng.random() < 0.80:
            damage = self.rng.integers(40, 75)
            self.wardens.current_hp -= damage
            self.metrics["damage_dealt"] += damage
            return damage * 0.15
        
        return 0.0
    
    def _handle_core_attack(self) -> float:
        """Handle attacking the core"""
        if not self.wardens.core_active:
            return -0.5  # No core to attack
        
        if self.player.attack_cooldown > 0:
            return -0.1
        
        self.player.attack_cooldown = 2  # Faster attacks on core
        
        # Always hits core
        damage = self.rng.integers(15, 30)
        self.wardens.core_hp -= damage
        
        reward = damage * 0.3  # Higher reward for core damage
        
        # Core killed
        if self.wardens.core_hp <= 0:
            self.wardens.core_active = False
            self.wardens.core_ticks_remaining = 0
            self.metrics["cores_killed"] += 1
            reward += 15.0  # Big bonus for killing core
        
        return reward
    
    def _handle_dodge(self) -> float:
        """Handle explicit dodge action"""
        if self.wardens.slam_incoming:
            # Move to random adjacent safe tile
            x, y = self.player.position
            moves = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            valid = [(nx, ny) for nx, ny in moves if 0 <= nx <= 4 and 0 <= ny <= 4]
            if valid:
                self.player.position = self.rng.choice(valid)
            return 2.0
        return -0.1  # Dodge when no slam incoming
    
    def _handle_prayer(self, action: WardensAction) -> float:
        """Handle prayer switching"""
        reward = 0.0
        
        if action == WardensAction.PRAY_MAGE:
            self.player.protect_mage = True
            self.player.protect_range = False
            self.player.protect_melee = False
            # Reward for correct prayer against divine
            if self.wardens.divine_projectile == "mage":
                reward += 1.5
        elif action == WardensAction.PRAY_RANGE:
            self.player.protect_range = True
            self.player.protect_mage = False
            self.player.protect_melee = False
            if self.wardens.divine_projectile == "range":
                reward += 1.5
        elif action == WardensAction.PRAY_MELEE:
            self.player.protect_melee = True
            self.player.protect_mage = False
            self.player.protect_range = False
        elif action == WardensAction.PRAY_OFF:
            self.player.protect_mage = False
            self.player.protect_range = False
            self.player.protect_melee = False
        elif action == WardensAction.TOGGLE_RIGOUR:
            self.player.rigour = not self.player.rigour
            self.player.augury = False
        elif action == WardensAction.TOGGLE_AUGURY:
            self.player.augury = not self.player.augury
            self.player.rigour = False
        
        return reward
    
    def _handle_eat(self) -> float:
        """Handle eating food"""
        if self.player.food_count <= 0:
            return -0.1
        if self.player.eat_cooldown > 0:
            return -0.1
        
        self.player.food_count -= 1
        self.player.eat_cooldown = 3
        
        heal = int(22 * (1 - self.heal_reduction))
        self.player.current_hp = min(self.player.max_hp, self.player.current_hp + heal)
        
        # Reward based on how needed the heal was
        hp_percent = self.player.current_hp / self.player.max_hp
        if hp_percent < 0.3:
            return 1.0  # Needed heal
        elif hp_percent < 0.5:
            return 0.5
        else:
            return -0.3  # Wasted food
    
    def _handle_brew(self) -> float:
        """Handle drinking brew"""
        if self.player.brew_doses <= 0:
            return -0.1
        if self.player.eat_cooldown > 0:
            return -0.1
        
        self.player.brew_doses -= 1
        self.player.eat_cooldown = 3
        
        heal = int(16 * (1 - self.heal_reduction))
        self.player.current_hp = min(self.player.max_hp, self.player.current_hp + heal)
        
        hp_percent = self.player.current_hp / self.player.max_hp
        if hp_percent < 0.4:
            return 0.8
        return -0.2
    
    def _handle_restore(self) -> float:
        """Handle drinking restore"""
        if self.player.restore_doses <= 0:
            return -0.1
        
        self.player.restore_doses -= 1
        restore = int(32 * (1 - self.heal_reduction * 0.5))  # Less affected
        self.player.current_prayer = min(self.player.max_prayer, self.player.current_prayer + restore)
        
        prayer_percent = self.player.current_prayer / self.player.max_prayer
        if prayer_percent < 0.3:
            return 1.0
        elif prayer_percent < 0.5:
            return 0.5
        return -0.2
    
    def _process_wardens_turn(self) -> float:
        """Process wardens attack and mechanics"""
        reward = 0.0
        
        # Determine next attack based on tick
        self.wardens.attack_tick += 1
        
        # Attack pattern (simplified)
        if self.wardens.attack_tick % 12 == 0:
            # Special attack cycle
            specials = ["slam", "lightning", "divine", "core"]
            weights = [0.25, 0.25, 0.3, 0.2]
            if self.wardens.phase == WardensPhase.P3_ENRAGED:
                weights = [0.3, 0.3, 0.25, 0.15]  # More dangerous in P3
            self.wardens.next_attack = self.rng.choice(specials, p=weights)
        
        # Process current mechanic
        if self.wardens.slam_incoming:
            self.wardens.slam_ticks -= 1
            if self.wardens.slam_ticks <= 0:
                # Slam lands
                damage = int(self.BASE_SLAM_DAMAGE * self.damage_mult)
                self.player.current_hp -= damage
                self.metrics["damage_taken"] += damage
                self.metrics["slams_hit"] += 1
                self.wardens.slam_incoming = False
                self.metrics["death_cause"] = "slam"
                reward -= 5.0  # Penalty for getting hit by slam
        
        elif self.wardens.lightning_active:
            # Check if player in lightning
            if self.player.position in self.wardens.lightning_tiles:
                damage = int(self.BASE_LIGHTNING_DAMAGE * self.damage_mult)
                self.player.current_hp -= damage
                self.metrics["damage_taken"] += damage
                self.metrics["lightning_hit"] += 1
                self.metrics["death_cause"] = "lightning"
                reward -= 3.0
            
            self.wardens.lightning_tiles = []  # Clear for next tick
            self.wardens.lightning_active = False
        
        elif self.wardens.core_active:
            self.wardens.core_ticks_remaining -= 1
            if self.wardens.core_ticks_remaining <= 0:
                # Core explodes
                damage = int(self.BASE_CORE_EXPLOSION * self.damage_mult)
                self.player.current_hp -= damage
                self.metrics["damage_taken"] += damage
                self.metrics["cores_exploded"] += 1
                self.wardens.core_active = False
                self.metrics["death_cause"] = "core_explosion"
                reward -= 8.0
        
        # Start new mechanic
        elif self.wardens.next_attack == "slam":
            self.wardens.slam_incoming = True
            self.wardens.slam_ticks = self.SLAM_WINDUP_TICKS
            self.wardens.next_attack = "auto"
        
        elif self.wardens.next_attack == "lightning":
            self.wardens.lightning_active = True
            # Random lightning tiles
            num_tiles = 3 if self.wardens.phase == WardensPhase.P3_ENRAGED else 2
            self.wardens.lightning_tiles = [
                (self.rng.integers(0, 5), self.rng.integers(0, 5))
                for _ in range(num_tiles)
            ]
            self.wardens.next_attack = "auto"
        
        elif self.wardens.next_attack == "divine":
            self.wardens.divine_projectile = self.rng.choice(["mage", "range"])
            # Check prayer
            correct_prayer = (
                (self.wardens.divine_projectile == "mage" and self.player.protect_mage) or
                (self.wardens.divine_projectile == "range" and self.player.protect_range)
            )
            if correct_prayer:
                self.metrics["prayers_correct"] += 1
                reward += 2.0
            else:
                damage = int(self.DIVINE_DAMAGE * self.damage_mult)
                self.player.current_hp -= damage
                self.metrics["damage_taken"] += damage
                self.metrics["prayers_wrong"] += 1
                self.metrics["death_cause"] = "divine_projectile"
                reward -= 2.0
            self.wardens.divine_projectile = "none"
            self.wardens.next_attack = "auto"
        
        elif self.wardens.next_attack == "core":
            self.wardens.core_active = True
            self.wardens.core_hp = self.wardens.core_max_hp
            self.wardens.core_ticks_remaining = self.CORE_DURATION_TICKS
            self.wardens.next_attack = "auto"
        
        else:
            # Auto attack
            if self.wardens.attack_tick % 4 == 0:
                damage = int(self.BASE_AUTO_DAMAGE * self.damage_mult)
                # Prayer can reduce
                if self.player.protect_mage or self.player.protect_range:
                    damage = int(damage * 0.4)  # 60% reduction
                self.player.current_hp -= damage
                self.metrics["damage_taken"] += damage
                if damage > 0:
                    self.metrics["death_cause"] = "auto_attack"
        
        return reward
    
    def _check_phase_transition(self):
        """Check and handle phase transitions"""
        hp_percent = self.wardens.current_hp / self.wardens.max_hp
        
        if self.wardens.phase == WardensPhase.P1_ELIDINIS and hp_percent <= self.wardens.p2_threshold:
            self.wardens.phase = WardensPhase.P2_TUMEKEN
            # Clear any active mechanics
            self.wardens.slam_incoming = False
            self.wardens.lightning_active = False
            self.wardens.core_active = False
        
        elif self.wardens.phase == WardensPhase.P2_TUMEKEN and hp_percent <= self.wardens.p3_threshold:
            self.wardens.phase = WardensPhase.P3_ENRAGED
            self.wardens.slam_incoming = False
            self.wardens.lightning_active = False
            self.wardens.core_active = False


# Convenience function to create env with specific invocation
def make_wardens_env(invocation: int = 0):
    def _init():
        return WardensEnvV2(invocation=invocation)
    return _init
