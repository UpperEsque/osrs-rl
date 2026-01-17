"""
Tombs of Amascut (ToA) - All Bosses with Max Invocations (600+)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import IntEnum
import gymnasium as gym
from gymnasium import spaces


# ============ INVOCATION SYSTEM ============

@dataclass
class InvocationSettings:
    """ToA Invocation raid modifiers for max invocations (600+)"""
    walk_the_path: bool = True
    persistence: bool = True
    overclocked: bool = True
    overclocked_2: bool = True
    insanity: bool = True
    on_a_diet: bool = True
    dehydration: bool = True
    blood_thinners: bool = True
    quiet_prayers: bool = True
    lively: bool = True
    upset_stomach: bool = True
    softcore: bool = True
    hardcore: bool = True
    walk_for_it: bool = True
    jungle_japes: bool = True
    shaking_things_up: bool = True
    boulderdash: bool = True
    
    def get_invocation_level(self) -> int:
        level = 0
        if self.walk_the_path: level += 50
        if self.persistence: level += 25
        if self.overclocked: level += 20
        if self.overclocked_2: level += 30
        if self.insanity: level += 50
        if self.on_a_diet: level += 30
        if self.dehydration: level += 15
        if self.blood_thinners: level += 25
        if self.quiet_prayers: level += 35
        if self.lively: level += 20
        if self.upset_stomach: level += 25
        if self.softcore: level += 25
        if self.hardcore: level += 50
        if self.walk_for_it: level += 30
        if self.jungle_japes: level += 20
        if self.shaking_things_up: level += 20
        if self.boulderdash: level += 25
        return level
    
    def get_hp_multiplier(self) -> float:
        return 1.25 if self.persistence else 1.0
    
    def get_attack_speed_multiplier(self) -> float:
        mult = 1.0
        if self.overclocked: mult -= 0.10
        if self.overclocked_2: mult -= 0.10
        return max(0.6, mult)
    
    def get_healing_multiplier(self) -> float:
        mult = 1.0
        if self.on_a_diet: mult -= 0.33
        if self.dehydration: mult -= 0.20
        return max(0.3, mult)
    
    def get_prayer_drain_multiplier(self) -> float:
        return 3.0 if self.insanity else 1.0


# ============ COMMON ACTION SPACE ============

class ToAAction(IntEnum):
    WAIT = 0
    ATTACK_MELEE = 1
    ATTACK_RANGED = 2
    ATTACK_MAGIC = 3
    ATTACK_SPEC = 4
    PRAY_MELEE = 5
    PRAY_RANGED = 6
    PRAY_MAGIC = 7
    PRAY_OFF = 8
    TOGGLE_PIETY = 9
    TOGGLE_RIGOUR = 10
    TOGGLE_AUGURY = 11
    MOVE_NORTH = 12
    MOVE_SOUTH = 13
    MOVE_EAST = 14
    MOVE_WEST = 15
    DODGE = 16
    EAT_FOOD = 17
    DRINK_RESTORE = 18
    USE_SPEC_RESTORE = 19
    KILL_ADD = 20


# ============ PLAYER STATE ============

@dataclass
class ToAPlayerState:
    current_hp: int = 99
    max_hp: int = 99
    current_prayer: int = 99
    max_prayer: int = 99
    special_attack: int = 100
    position: Tuple[int, int] = (5, 5)
    protect_melee: bool = False
    protect_ranged: bool = False
    protect_magic: bool = False
    piety: bool = False
    rigour: bool = False
    augury: bool = False
    attack_cooldown: int = 0
    eat_cooldown: int = 0
    food_count: int = 6
    restores: int = 4
    melee_max: int = 55
    ranged_max: int = 50
    magic_max: int = 48
    
    def reset(self):
        self.current_hp = self.max_hp
        self.current_prayer = self.max_prayer
        self.special_attack = 100
        self.position = (5, 5)
        self.protect_melee = False
        self.protect_ranged = False
        self.protect_magic = False
        self.attack_cooldown = 0
        self.eat_cooldown = 0
        self.food_count = 6
        self.restores = 4


# ============ WARDENS (FINAL BOSS) ============

class WardenPhase(IntEnum):
    P1 = 0
    P2 = 1
    P3 = 2


@dataclass
class WardenState:
    current_hp: int = 880
    max_hp: int = 880
    phase: WardenPhase = WardenPhase.P1
    attack_cooldown: int = 0
    attacks_until_special: int = 5
    slam_incoming: bool = False
    slam_ticks: int = 0
    core_active: bool = False
    core_hp: int = 0
    lightning_tiles: List[Tuple[int, int]] = field(default_factory=list)
    enraged: bool = False
    
    def reset(self, invocations: InvocationSettings):
        hp_mult = invocations.get_hp_multiplier()
        self.current_hp = int(880 * hp_mult)
        self.max_hp = int(880 * hp_mult)
        self.phase = WardenPhase.P1
        self.attack_cooldown = 0
        self.attacks_until_special = 5
        self.slam_incoming = False
        self.core_active = False
        self.lightning_tiles = []
        self.enraged = False


class WardenMechanics:
    ARENA_SIZE = 12
    SLAM_DAMAGE = 70
    DIVINE_DAMAGE = 55
    LIGHTNING_DAMAGE = 40
    CORE_EXPLODE_DAMAGE = 80
    
    def __init__(self, invocations=None, rng=None):
        self.invocations = invocations or InvocationSettings()
        self.rng = rng or np.random.default_rng()
        self.boss = WardenState()
        self.player = ToAPlayerState()
        self.tick = 0
    
    def reset(self):
        self.boss.reset(self.invocations)
        self.player.reset()
        self.tick = 0
        return self.boss, self.player
    
    def step(self, action: ToAAction):
        self.tick += 1
        reward = 0.0
        info = {"damage_dealt": 0, "damage_taken": 0}
        
        # Player action
        reward += self._process_player(action, info)
        
        # Boss action
        reward += self._process_boss(info)
        
        # Update status
        self._update_status()
        
        # Check done
        done = False
        if self.player.current_hp <= 0:
            done = True
            reward -= 100
            info["result"] = "death"
        elif self.boss.current_hp <= 0:
            done = True
            reward += 150
            info["result"] = "kill"
        
        return reward, done, info
    
    def _process_player(self, action, info):
        reward = 0.0
        if self.player.attack_cooldown > 0:
            self.player.attack_cooldown -= 1
        if self.player.eat_cooldown > 0:
            self.player.eat_cooldown -= 1
        
        if action == ToAAction.ATTACK_RANGED:
            if self.player.attack_cooldown <= 0:
                if self.rng.random() < 0.85:
                    dmg = self.rng.integers(0, self.player.ranged_max + 1)
                    if self.player.rigour: dmg = int(dmg * 1.23)
                    self.boss.current_hp -= dmg
                    info["damage_dealt"] = dmg
                    reward += dmg / 10
                self.player.attack_cooldown = 4
        
        elif action == ToAAction.ATTACK_SPEC:
            if self.player.special_attack >= 50 and self.player.attack_cooldown <= 0:
                self.player.special_attack -= 50
                dmg = self.rng.integers(40, 80)
                self.boss.current_hp -= dmg
                info["damage_dealt"] = dmg
                reward += dmg / 5
                self.player.attack_cooldown = 5
        
        elif action == ToAAction.KILL_ADD:
            if self.boss.core_active:
                self.boss.core_hp -= self.rng.integers(20, 40)
                if self.boss.core_hp <= 0:
                    self.boss.core_active = False
                    reward += 10
        
        elif action == ToAAction.PRAY_MELEE:
            self.player.protect_melee = True
            self.player.protect_ranged = False
            self.player.protect_magic = False
        elif action == ToAAction.PRAY_RANGED:
            self.player.protect_ranged = True
            self.player.protect_melee = False
            self.player.protect_magic = False
        elif action == ToAAction.PRAY_MAGIC:
            self.player.protect_magic = True
            self.player.protect_melee = False
            self.player.protect_ranged = False
        elif action == ToAAction.TOGGLE_RIGOUR:
            self.player.rigour = not self.player.rigour
        
        elif action in [ToAAction.MOVE_NORTH, ToAAction.MOVE_SOUTH, ToAAction.MOVE_EAST, ToAAction.MOVE_WEST]:
            dx, dy = {
                ToAAction.MOVE_NORTH: (0, -1),
                ToAAction.MOVE_SOUTH: (0, 1),
                ToAAction.MOVE_EAST: (1, 0),
                ToAAction.MOVE_WEST: (-1, 0),
            }[action]
            x, y = self.player.position
            self.player.position = (
                max(0, min(self.ARENA_SIZE-1, x + dx)),
                max(0, min(self.ARENA_SIZE-1, y + dy))
            )
            if self.boss.slam_incoming:
                reward += 3
                self.boss.slam_incoming = False
        
        elif action == ToAAction.DODGE:
            # Move away from danger
            if self.boss.lightning_tiles:
                for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                    new_pos = (self.player.position[0]+dx, self.player.position[1]+dy)
                    if new_pos not in self.boss.lightning_tiles:
                        self.player.position = new_pos
                        break
            if self.boss.slam_incoming:
                reward += 2
        
        elif action == ToAAction.EAT_FOOD:
            if self.player.food_count > 0 and self.player.eat_cooldown <= 0:
                heal = int(22 * self.invocations.get_healing_multiplier())
                self.player.current_hp = min(self.player.max_hp, self.player.current_hp + heal)
                self.player.food_count -= 1
                self.player.eat_cooldown = 3
        
        elif action == ToAAction.DRINK_RESTORE:
            if self.player.restores > 0:
                self.player.current_prayer = min(self.player.max_prayer, self.player.current_prayer + 32)
                self.player.restores -= 1
        
        return reward
    
    def _process_boss(self, info):
        reward = 0.0
        damage = 0
        
        if self.boss.attack_cooldown > 0:
            self.boss.attack_cooldown -= 1
            return reward
        
        self.boss.attacks_until_special -= 1
        
        # Specials
        if self.boss.attacks_until_special <= 0:
            special = self.rng.choice(["slam", "lightning", "core"])
            if special == "slam":
                self.boss.slam_incoming = True
                self.boss.slam_ticks = 2
            elif special == "lightning":
                self.boss.lightning_tiles = [
                    (self.rng.integers(0, self.ARENA_SIZE), self.rng.integers(0, self.ARENA_SIZE))
                    for _ in range(10)
                ]
            elif special == "core":
                self.boss.core_active = True
                self.boss.core_hp = 50
            self.boss.attacks_until_special = 4 if self.boss.enraged else 6
        else:
            # Normal attack
            style = self.rng.choice(["magic", "ranged"])
            if style == "magic" and not self.player.protect_magic:
                damage = self.rng.integers(20, self.DIVINE_DAMAGE)
            elif style == "ranged" and not self.player.protect_ranged:
                damage = self.rng.integers(20, self.DIVINE_DAMAGE)
            else:
                reward += 0.5
        
        # Slam
        if self.boss.slam_incoming:
            self.boss.slam_ticks -= 1
            if self.boss.slam_ticks <= 0 and self.boss.slam_incoming:
                damage += self.SLAM_DAMAGE
                reward -= 5
                self.boss.slam_incoming = False
        
        # Lightning
        if self.player.position in self.boss.lightning_tiles:
            damage += self.LIGHTNING_DAMAGE
            reward -= 2
        
        # Clear lightning
        if self.rng.random() < 0.3:
            self.boss.lightning_tiles = []
        
        # Core explosion
        if self.boss.core_active and self.tick % 20 == 0:
            if self.boss.core_hp > 0:
                damage += self.CORE_EXPLODE_DAMAGE
                reward -= 8
                self.boss.core_active = False
        
        self.player.current_hp -= damage
        info["damage_taken"] += damage
        
        speed = self.invocations.get_attack_speed_multiplier()
        self.boss.attack_cooldown = int(4 * speed)
        
        # Enrage
        if self.boss.current_hp < self.boss.max_hp * 0.2:
            self.boss.enraged = True
        
        return reward
    
    def _update_status(self):
        drain = 0
        if self.player.protect_melee or self.player.protect_ranged or self.player.protect_magic:
            drain += 1
        if self.player.rigour:
            drain += 2
        drain = int(drain * self.invocations.get_prayer_drain_multiplier())
        self.player.current_prayer = max(0, self.player.current_prayer - drain // 2)
        
        if self.player.current_prayer <= 0:
            self.player.protect_melee = False
            self.player.protect_ranged = False
            self.player.protect_magic = False
            self.player.rigour = False
    
    def get_observation(self):
        obs = np.zeros(40, dtype=np.float32)
        obs[0] = self.player.current_hp / self.player.max_hp
        obs[1] = self.player.current_prayer / self.player.max_prayer
        obs[2] = self.player.special_attack / 100
        obs[3] = self.player.food_count / 6
        obs[4] = self.player.position[0] / self.ARENA_SIZE
        obs[5] = self.player.position[1] / self.ARENA_SIZE
        obs[6] = 1.0 if self.player.protect_melee else 0.0
        obs[7] = 1.0 if self.player.protect_ranged else 0.0
        obs[8] = 1.0 if self.player.protect_magic else 0.0
        obs[9] = 1.0 if self.player.rigour else 0.0
        obs[10] = self.player.attack_cooldown / 6
        
        obs[16] = self.boss.current_hp / self.boss.max_hp
        obs[17] = float(self.boss.phase) / 2
        obs[18] = 1.0 if self.boss.slam_incoming else 0.0
        obs[19] = self.boss.slam_ticks / 3 if self.boss.slam_incoming else 0.0
        obs[20] = 1.0 if self.boss.core_active else 0.0
        obs[21] = self.boss.core_hp / 50 if self.boss.core_active else 0.0
        obs[22] = len(self.boss.lightning_tiles) / 15
        obs[23] = 1.0 if self.player.position in self.boss.lightning_tiles else 0.0
        obs[24] = 1.0 if self.boss.enraged else 0.0
        obs[25] = self.tick / 1000
        
        return obs


class WardensEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(self, render_mode=None, max_ticks=800):
        super().__init__()
        self.render_mode = render_mode
        self.max_ticks = max_ticks
        self.invocations = InvocationSettings()
        self.mechanics = WardenMechanics(self.invocations)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(40,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ToAAction))
        self.total_damage_dealt = 0
        self.total_damage_taken = 0
        self.kills = 0
        self.deaths = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed: self.mechanics.rng = np.random.default_rng(seed)
        self.mechanics.reset()
        self.total_damage_dealt = 0
        self.total_damage_taken = 0
        return self.mechanics.get_observation(), self._get_info()
    
    def step(self, action):
        reward, done, info = self.mechanics.step(ToAAction(action))
        self.total_damage_dealt += info.get("damage_dealt", 0)
        self.total_damage_taken += info.get("damage_taken", 0)
        if info.get("result") == "kill": self.kills += 1
        elif info.get("result") == "death": self.deaths += 1
        truncated = self.mechanics.tick >= self.max_ticks
        return self.mechanics.get_observation(), reward, done, truncated, self._get_info()
    
    def _get_info(self):
        return {
            "tick": self.mechanics.tick,
            "boss_hp": self.mechanics.boss.current_hp,
            "player_hp": self.mechanics.player.current_hp,
            "phase": self.mechanics.boss.phase.name,
            "invocation": self.invocations.get_invocation_level(),
            "damage_dealt": self.total_damage_dealt,
            "damage_taken": self.total_damage_taken,
            "kills": self.kills,
            "deaths": self.deaths,
        }
    
    def render(self):
        if self.render_mode in ["human", "ansi"]:
            b = self.mechanics.boss
            p = self.mechanics.player
            print(f"\n{'='*55}")
            print(f"WARDENS (600 Invo) | Tick {self.mechanics.tick} | Phase: {b.phase.name}")
            print(f"{'='*55}")
            hp_bar = '█' * int(b.current_hp/b.max_hp*20) + '░' * (20-int(b.current_hp/b.max_hp*20))
            print(f"Warden: [{hp_bar}] {b.current_hp}/{b.max_hp}")
            if b.slam_incoming: print(f"  ⚠️ SLAM ({b.slam_ticks} ticks)")
            if b.core_active: print(f"  💀 CORE ({b.core_hp} HP)")
            if b.lightning_tiles: print(f"  ⚡ LIGHTNING ({len(b.lightning_tiles)} tiles)")
            if b.enraged: print(f"  🔥 ENRAGED")
            hp_bar = '█' * int(p.current_hp/p.max_hp*20) + '░' * (20-int(p.current_hp/p.max_hp*20))
            print(f"\nPlayer: [{hp_bar}] {p.current_hp}/{p.max_hp}")
            prayers = []
            if p.protect_magic: prayers.append("🛡️Mage")
            if p.protect_ranged: prayers.append("🛡️Range")
            if p.rigour: prayers.append("⚔️Rigour")
            print(f"Prayer: {p.current_prayer} | Active: {' '.join(prayers) if prayers else 'None'}")
            print(f"Food: {p.food_count} | Spec: {p.special_attack}%")


# ============ OTHER BOSSES ============
# Simplified versions for faster training

class KephriEnv(gym.Env):
    """Kephri - Scarab boss with swarms and shields"""
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, render_mode=None, max_ticks=500):
        super().__init__()
        self.render_mode = render_mode
        self.max_ticks = max_ticks
        self.rng = np.random.default_rng()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(40,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ToAAction))
        self.reset()
    
    def reset(self, seed=None, options=None):
        if seed: self.rng = np.random.default_rng(seed)
        self.tick = 0
        self.boss_hp = 560
        self.boss_max_hp = 560
        self.player_hp = 99
        self.player_prayer = 99
        self.swarms = 0
        self.shield = False
        self.attack_cd = 0
        self.total_dmg_dealt = 0
        self.total_dmg_taken = 0
        return self._get_obs(), {}
    
    def step(self, action):
        self.tick += 1
        reward = 0
        dmg_dealt = 0
        dmg_taken = 0
        
        # Player action
        if action == ToAAction.ATTACK_RANGED and self.attack_cd <= 0 and not self.shield:
            dmg_dealt = self.rng.integers(0, 50)
            self.boss_hp -= dmg_dealt
            reward += dmg_dealt / 10
            self.attack_cd = 4
        elif action == ToAAction.ATTACK_SPEC and not self.shield:
            # Keris spec - 33% chance of triple
            if self.rng.random() < 0.33:
                dmg_dealt = self.rng.integers(80, 130)
            else:
                dmg_dealt = self.rng.integers(20, 50)
            self.boss_hp -= dmg_dealt
            reward += dmg_dealt / 5
            self.attack_cd = 5
        elif action == ToAAction.KILL_ADD:
            if self.swarms > 0:
                self.swarms -= 1
                reward += 3
            elif self.shield:
                self.shield = False
                reward += 5
        
        if self.attack_cd > 0:
            self.attack_cd -= 1
        
        # Boss action
        if self.rng.random() < 0.15:
            self.swarms = 3
        elif self.rng.random() < 0.1:
            self.shield = True
        else:
            dmg_taken = self.rng.integers(10, 30)
            self.player_hp -= dmg_taken
        
        # Swarm damage
        if self.swarms > 0 and self.tick % 6 == 0:
            swarm_dmg = self.swarms * 15
            self.player_hp -= swarm_dmg
            dmg_taken += swarm_dmg
            reward -= 2
        
        self.total_dmg_dealt += dmg_dealt
        self.total_dmg_taken += dmg_taken
        
        done = False
        if self.player_hp <= 0:
            done = True
            reward -= 100
        elif self.boss_hp <= 0:
            done = True
            reward += 100
        
        return self._get_obs(), reward, done, self.tick >= self.max_ticks, {
            "damage_dealt": self.total_dmg_dealt,
            "damage_taken": self.total_dmg_taken,
            "kills": 1 if self.boss_hp <= 0 else 0,
            "deaths": 1 if self.player_hp <= 0 else 0,
        }
    
    def _get_obs(self):
        obs = np.zeros(40, dtype=np.float32)
        obs[0] = self.player_hp / 99
        obs[1] = self.player_prayer / 99
        obs[16] = self.boss_hp / self.boss_max_hp
        obs[17] = self.swarms / 4
        obs[18] = 1.0 if self.shield else 0.0
        return obs
    
    def render(self):
        print(f"Kephri: {self.boss_hp}/{self.boss_max_hp} | Player: {self.player_hp} | Swarms: {self.swarms} | Shield: {self.shield}")


class ZebakEnv(gym.Env):
    """Zebak - Crocodile boss with waves"""
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, render_mode=None, max_ticks=500):
        super().__init__()
        self.render_mode = render_mode
        self.max_ticks = max_ticks
        self.rng = np.random.default_rng()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(40,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ToAAction))
        self.reset()
    
    def reset(self, seed=None, options=None):
        if seed: self.rng = np.random.default_rng(seed)
        self.tick = 0
        self.boss_hp = 725
        self.boss_max_hp = 725
        self.player_hp = 99
        self.player_prayer = 99
        self.protect_mage = False
        self.protect_range = False
        self.wave_active = False
        self.player_y = 5
        self.wave_y = -1
        self.blood_spawns = 0
        self.enraged = False
        self.last_style = 0
        self.attack_cd = 0
        self.total_dmg_dealt = 0
        self.total_dmg_taken = 0
        return self._get_obs(), {}
    
    def step(self, action):
        self.tick += 1
        reward = 0
        dmg_dealt = 0
        dmg_taken = 0
        
        # Player
        if action == ToAAction.ATTACK_MAGIC and self.attack_cd <= 0:
            dmg_dealt = self.rng.integers(0, 48)
            self.boss_hp -= dmg_dealt
            reward += dmg_dealt / 10
            self.attack_cd = 5
        elif action == ToAAction.PRAY_MAGIC:
            self.protect_mage = True
            self.protect_range = False
        elif action == ToAAction.PRAY_RANGED:
            self.protect_range = True
            self.protect_mage = False
        elif action == ToAAction.MOVE_NORTH:
            self.player_y = max(0, self.player_y - 2)
        elif action == ToAAction.MOVE_SOUTH:
            self.player_y = min(10, self.player_y + 2)
        elif action == ToAAction.KILL_ADD and self.blood_spawns > 0:
            self.blood_spawns -= 1
            reward += 3
        
        if self.attack_cd > 0:
            self.attack_cd -= 1
        
        # Boss
        if self.rng.random() < 0.12:
            self.wave_active = True
            self.wave_y = 0
        elif self.rng.random() < 0.1:
            self.blood_spawns = 3
        else:
            # Alternate mage/range when enraged
            if self.enraged:
                self.last_style = 1 - self.last_style
            else:
                self.last_style = self.rng.integers(0, 2)
            
            if self.last_style == 0:  # Mage
                if not self.protect_mage:
                    dmg_taken = self.rng.integers(15, 40)
                else:
                    reward += 0.5
            else:  # Range
                if not self.protect_range:
                    dmg_taken = self.rng.integers(15, 40)
                else:
                    reward += 0.5
        
        # Wave damage
        if self.wave_active:
            self.wave_y += 1
            if abs(self.player_y - self.wave_y) <= 1:
                dmg_taken += 45
                reward -= 3
            if self.wave_y > 10:
                self.wave_active = False
        
        # Blood heals boss
        if self.blood_spawns > 0 and self.tick % 8 == 0:
            self.boss_hp = min(self.boss_max_hp, self.boss_hp + self.blood_spawns * 20)
            reward -= 2
        
        self.player_hp -= dmg_taken
        
        # Enrage
        if self.boss_hp < self.boss_max_hp * 0.25:
            self.enraged = True
        
        self.total_dmg_dealt += dmg_dealt
        self.total_dmg_taken += dmg_taken
        
        done = False
        if self.player_hp <= 0:
            done = True
            reward -= 100
        elif self.boss_hp <= 0:
            done = True
            reward += 100
        
        return self._get_obs(), reward, done, self.tick >= self.max_ticks, {
            "damage_dealt": self.total_dmg_dealt,
            "damage_taken": self.total_dmg_taken,
            "kills": 1 if self.boss_hp <= 0 else 0,
            "deaths": 1 if self.player_hp <= 0 else 0,
        }
    
    def _get_obs(self):
        obs = np.zeros(40, dtype=np.float32)
        obs[0] = self.player_hp / 99
        obs[1] = self.player_prayer / 99
        obs[4] = self.player_y / 10
        obs[7] = 1.0 if self.protect_range else 0.0
        obs[8] = 1.0 if self.protect_mage else 0.0
        obs[16] = self.boss_hp / self.boss_max_hp
        obs[17] = 1.0 if self.wave_active else 0.0
        obs[18] = abs(self.player_y - self.wave_y) / 10 if self.wave_active else 1.0
        obs[19] = self.blood_spawns / 3
        obs[20] = float(self.last_style)
        obs[21] = 1.0 if self.enraged else 0.0
        return obs
    
    def render(self):
        print(f"Zebak: {self.boss_hp}/{self.boss_max_hp} | Player: {self.player_hp} y={self.player_y} | Wave: {self.wave_y if self.wave_active else 'None'}")


class AkkhaEnv(gym.Env):
    """Akkha - Shadow boss with memory mechanic"""
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, render_mode=None, max_ticks=600):
        super().__init__()
        self.render_mode = render_mode
        self.max_ticks = max_ticks
        self.rng = np.random.default_rng()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(50,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ToAAction))
        self.reset()
    
    def reset(self, seed=None, options=None):
        if seed: self.rng = np.random.default_rng(seed)
        self.tick = 0
        self.boss_hp = 600
        self.boss_max_hp = 600
        self.player_hp = 99
        self.player_prayer = 99
        self.protect = [False, False, False]  # melee, range, mage
        self.current_style = 0
        self.attack_sequence = []
        self.memory_active = False
        self.memory_index = 0
        self.shadow_style = -1
        self.attack_cd = 0
        self.total_dmg_dealt = 0
        self.total_dmg_taken = 0
        return self._get_obs(), {}
    
    def step(self, action):
        self.tick += 1
        reward = 0
        dmg_dealt = 0
        dmg_taken = 0
        
        # Player
        if action == ToAAction.ATTACK_MELEE and self.attack_cd <= 0:
            dmg_dealt = self.rng.integers(0, 55)
            self.boss_hp -= dmg_dealt
            reward += dmg_dealt / 10
            self.attack_cd = 4
        elif action == ToAAction.PRAY_MELEE:
            self.protect = [True, False, False]
            reward += self._check_memory(0)
        elif action == ToAAction.PRAY_RANGED:
            self.protect = [False, True, False]
            reward += self._check_memory(1)
        elif action == ToAAction.PRAY_MAGIC:
            self.protect = [False, False, True]
            reward += self._check_memory(2)
        
        if self.attack_cd > 0:
            self.attack_cd -= 1
        
        # Boss
        if not self.memory_active and self.rng.random() < 0.12 and len(self.attack_sequence) >= 4:
            self.memory_active = True
            self.memory_index = 0
        elif self.rng.random() < 0.08:
            self.shadow_style = (self.current_style + 1) % 3
        else:
            # Rotate style
            self.current_style = (self.current_style + 1) % 3
            self.attack_sequence.append(self.current_style)
            if len(self.attack_sequence) > 8:
                self.attack_sequence.pop(0)
            
            if not self.protect[self.current_style]:
                dmg_taken = self.rng.integers(15, 35)
            else:
                reward += 0.5
        
        # Shadow attack
        if self.shadow_style >= 0:
            if not self.protect[self.shadow_style]:
                dmg_taken += self.rng.integers(10, 25)
            self.shadow_style = -1
        
        self.player_hp -= dmg_taken
        self.total_dmg_dealt += dmg_dealt
        self.total_dmg_taken += dmg_taken
        
        done = False
        if self.player_hp <= 0:
            done = True
            reward -= 100
        elif self.boss_hp <= 0:
            done = True
            reward += 100
        
        return self._get_obs(), reward, done, self.tick >= self.max_ticks, {
            "damage_dealt": self.total_dmg_dealt,
            "damage_taken": self.total_dmg_taken,
            "kills": 1 if self.boss_hp <= 0 else 0,
            "deaths": 1 if self.player_hp <= 0 else 0,
        }
    
    def _check_memory(self, style):
        if not self.memory_active:
            return 0
        if self.memory_index < len(self.attack_sequence):
            expected = self.attack_sequence[-(4-self.memory_index)]
            if style == expected:
                self.memory_index += 1
                if self.memory_index >= 4:
                    self.memory_active = False
                    return 5
                return 1
            else:
                self.player_hp -= 40
                return -3
        return 0
    
    def _get_obs(self):
        obs = np.zeros(50, dtype=np.float32)
        obs[0] = self.player_hp / 99
        obs[1] = self.player_prayer / 99
        obs[6] = 1.0 if self.protect[0] else 0.0
        obs[7] = 1.0 if self.protect[1] else 0.0
        obs[8] = 1.0 if self.protect[2] else 0.0
        obs[16] = self.boss_hp / self.boss_max_hp
        obs[17] = float(self.current_style) / 2
        obs[18] = 1.0 if self.memory_active else 0.0
        obs[19] = self.memory_index / 4
        obs[20] = float(self.shadow_style + 1) / 3
        # Memory sequence
        for i, s in enumerate(self.attack_sequence[-4:]):
            obs[30+i] = float(s) / 2
        return obs
    
    def render(self):
        styles = ['M','R','G']
        print(f"Akkha: {self.boss_hp}/{self.boss_max_hp} | Player: {self.player_hp} | Style: {styles[self.current_style]} | Memory: {self.memory_active}")


class BaBaEnv(gym.Env):
    """Ba-Ba - Baboon boss with boulders"""
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, render_mode=None, max_ticks=500):
        super().__init__()
        self.render_mode = render_mode
        self.max_ticks = max_ticks
        self.rng = np.random.default_rng()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(40,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ToAAction))
        self.reset()
    
    def reset(self, seed=None, options=None):
        if seed: self.rng = np.random.default_rng(seed)
        self.tick = 0
        self.boss_hp = 480
        self.boss_max_hp = 480
        self.player_hp = 99
        self.player_x = 5
        self.protect_melee = False
        self.boulder_lanes = []
        self.slam_incoming = False
        self.safe_x = 5
        self.baboons = 0
        self.attack_cd = 0
        self.total_dmg_dealt = 0
        self.total_dmg_taken = 0
        return self._get_obs(), {}
    
    def step(self, action):
        self.tick += 1
        reward = 0
        dmg_dealt = 0
        dmg_taken = 0
        
        # Player
        if action == ToAAction.ATTACK_MELEE and self.attack_cd <= 0:
            dmg_dealt = self.rng.integers(0, 55)
            self.boss_hp -= dmg_dealt
            reward += dmg_dealt / 10
            self.attack_cd = 4
        elif action == ToAAction.PRAY_MELEE:
            self.protect_melee = True
        elif action == ToAAction.MOVE_EAST:
            self.player_x = min(10, self.player_x + 2)
        elif action == ToAAction.MOVE_WEST:
            self.player_x = max(0, self.player_x - 2)
        elif action == ToAAction.KILL_ADD and self.baboons > 0:
            self.baboons -= 1
            reward += 2
        
        if self.attack_cd > 0:
            self.attack_cd -= 1
        
        # Boss
        if self.rng.random() < 0.15:
            self.boulder_lanes = [self.rng.integers(0, 11) for _ in range(3)]
        elif self.rng.random() < 0.1:
            self.slam_incoming = True
            self.safe_x = self.rng.integers(0, 11)
        elif self.rng.random() < 0.1:
            self.baboons = 3
        else:
            if not self.protect_melee:
                dmg_taken = self.rng.integers(15, 40)
            else:
                dmg_taken = self.rng.integers(0, 10)
                reward += 0.3
        
        # Boulder damage
        if self.player_x in self.boulder_lanes:
            dmg_taken += 50
            reward -= 3
        self.boulder_lanes = []
        
        # Slam damage
        if self.slam_incoming:
            if abs(self.player_x - self.safe_x) > 1:
                dmg_taken += 55
                reward -= 3
            else:
                reward += 2
            self.slam_incoming = False
        
        self.player_hp -= dmg_taken
        self.total_dmg_dealt += dmg_dealt
        self.total_dmg_taken += dmg_taken
        
        done = False
        if self.player_hp <= 0:
            done = True
            reward -= 100
        elif self.boss_hp <= 0:
            done = True
            reward += 100
        
        return self._get_obs(), reward, done, self.tick >= self.max_ticks, {
            "damage_dealt": self.total_dmg_dealt,
            "damage_taken": self.total_dmg_taken,
            "kills": 1 if self.boss_hp <= 0 else 0,
            "deaths": 1 if self.player_hp <= 0 else 0,
        }
    
    def _get_obs(self):
        obs = np.zeros(40, dtype=np.float32)
        obs[0] = self.player_hp / 99
        obs[4] = self.player_x / 10
        obs[6] = 1.0 if self.protect_melee else 0.0
        obs[16] = self.boss_hp / self.boss_max_hp
        obs[17] = 1.0 if self.boulder_lanes else 0.0
        obs[18] = 1.0 if self.player_x in self.boulder_lanes else 0.0
        obs[19] = 1.0 if self.slam_incoming else 0.0
        obs[20] = self.safe_x / 10 if self.slam_incoming else 0.5
        obs[21] = self.baboons / 3
        return obs
    
    def render(self):
        print(f"Ba-Ba: {self.boss_hp}/{self.boss_max_hp} | Player: {self.player_hp} x={self.player_x} | Boulders: {self.boulder_lanes}")


# Test
if __name__ == "__main__":
    print("Testing ToA bosses...")
    
    for name, EnvClass in [("Wardens", WardensEnv), ("Kephri", KephriEnv), 
                           ("Zebak", ZebakEnv), ("Akkha", AkkhaEnv), ("Ba-Ba", BaBaEnv)]:
        print(f"\n--- {name} ---")
        env = EnvClass(render_mode="human")
        obs, _ = env.reset()
        print(f"Obs: {obs.shape} | Actions: {env.action_space.n}")
        
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, done, trunc, info = env.step(action)
            if done: break
        
        env.render()
        print(f"✓ {name} working!")
    
    print("\n✓ All ToA bosses ready!")
