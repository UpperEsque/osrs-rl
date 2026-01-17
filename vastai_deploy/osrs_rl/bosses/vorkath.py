"""
Vorkath Boss - Full Mechanics Implementation

Phases:
1. Normal Phase - Ranged/Magic attacks, acid pools, fireballs
2. Acid Phase - Walks around spitting acid, player must dodge
3. Zombified Spawn Phase - Spawns a zombie that must be killed (Crumble Undead)

Attack Pattern:
- 6 normal attacks
- Then either Acid Phase OR Zombified Spawn (alternates)
- Repeat

Special Attacks:
- Fireball (purple) - One-shot if not moved (121 damage)
- Acid pools - Rapid damage if standing in them
- Freeze breath - Freezes player, dragonfire damage
- Prayer disable - Disables prayers briefly
- Zombified spawn - Must kill with Crumble Undead or it heals Vorkath
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import IntEnum
import gymnasium as gym
from gymnasium import spaces


class VorkathPhase(IntEnum):
    NORMAL = 0
    ACID = 1
    ZOMBIFIED_SPAWN = 2


class VorkathAttack(IntEnum):
    MAGIC_DRAGONFIRE = 0
    RANGED_DRAGONFIRE = 1
    MELEE_DRAGONFIRE = 2
    PINK_DRAGONFIRE = 3
    FIREBALL = 4
    FREEZE_BREATH = 5
    ACID_RAPID_FIRE = 6
    ACID_PHASE_START = 7
    ZOMBIFIED_SPAWN = 8


class VorkathAction(IntEnum):
    WAIT = 0
    ATTACK_RANGED = 1
    ATTACK_SPEC = 2
    CAST_CRUMBLE_UNDEAD = 3
    PRAY_MAGE = 4
    PRAY_RANGE = 5
    PRAY_OFF = 6
    TOGGLE_RIGOUR = 7
    MOVE_NORTH = 8
    MOVE_SOUTH = 9
    MOVE_EAST = 10
    MOVE_WEST = 11
    WALK_AROUND = 12
    EAT_FOOD = 13
    DRINK_PRAYER = 14
    DRINK_ANTIFIRE = 15
    DRINK_ANTIVENOM = 16


@dataclass
class VorkathState:
    current_hp: int = 750
    max_hp: int = 750
    phase: VorkathPhase = VorkathPhase.NORMAL
    attacks_until_special: int = 6
    next_special_is_acid: bool = True
    attack_cooldown: int = 0
    current_attack: Optional[VorkathAttack] = None
    fireball_incoming: bool = False
    fireball_ticks: int = 0
    freeze_active: bool = False
    freeze_ticks: int = 0
    acid_pools: List[Tuple[int, int]] = field(default_factory=list)
    acid_phase_ticks: int = 0
    zombified_spawn_hp: int = 0
    zombified_spawn_active: bool = False
    spawn_ticks_until_heal: int = 0
    position: Tuple[int, int] = (5, 5)
    is_attackable: bool = True
    
    def reset(self):
        self.current_hp = self.max_hp
        self.phase = VorkathPhase.NORMAL
        self.attacks_until_special = 6
        self.next_special_is_acid = True
        self.attack_cooldown = 0
        self.current_attack = None
        self.fireball_incoming = False
        self.fireball_ticks = 0
        self.freeze_active = False
        self.freeze_ticks = 0
        self.acid_pools = []
        self.acid_phase_ticks = 0
        self.zombified_spawn_hp = 0
        self.zombified_spawn_active = False
        self.spawn_ticks_until_heal = 0
        self.position = (5, 5)
        self.is_attackable = True


@dataclass
class VorkathPlayerState:
    current_hp: int = 99
    max_hp: int = 99
    current_prayer: int = 99
    max_prayer: int = 99
    special_attack: int = 100
    position: Tuple[int, int] = (5, 8)
    protect_magic: bool = False
    protect_ranged: bool = False
    rigour: bool = False
    attack_cooldown: int = 0
    eat_cooldown: int = 0
    food_count: int = 8
    prayer_pots: int = 4
    antifire_active: bool = True
    antivenom_active: bool = True
    frozen: bool = False
    freeze_ticks: int = 0
    venomed: bool = False
    crumble_undead_casts: int = 10
    ranged_level: int = 99
    ranged_max_hit: int = 48
    
    def reset(self):
        self.current_hp = self.max_hp
        self.current_prayer = self.max_prayer
        self.special_attack = 100
        self.position = (5, 8)
        self.protect_magic = False
        self.protect_ranged = False
        self.rigour = False
        self.attack_cooldown = 0
        self.eat_cooldown = 0
        self.food_count = 8
        self.prayer_pots = 4
        self.antifire_active = True
        self.antivenom_active = True
        self.frozen = False
        self.freeze_ticks = 0
        self.venomed = False
        self.crumble_undead_casts = 10


class VorkathMechanics:
    ARENA_SIZE = 10
    MAGIC_MAX_HIT = 30
    RANGED_MAX_HIT = 32
    PINK_DRAGONFIRE_MAX = 30
    FIREBALL_DAMAGE = 121
    ACID_DAMAGE_PER_TICK = 8
    FREEZE_BREATH_DAMAGE = 25
    ZOMBIFIED_SPAWN_HEAL = 50
    
    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng()
        self.vorkath = VorkathState()
        self.player = VorkathPlayerState()
        self.tick = 0
    
    def reset(self) -> Tuple[VorkathState, VorkathPlayerState]:
        self.vorkath.reset()
        self.player.reset()
        self.tick = 0
        return self.vorkath, self.player
    
    def step(self, action: VorkathAction) -> Tuple[float, bool, dict]:
        self.tick += 1
        reward = 0.0
        info = {"damage_dealt": 0, "damage_taken": 0}
        
        action_reward, action_info = self._process_player_action(action)
        reward += action_reward
        info.update(action_info)
        
        self._update_player_status()
        
        vorkath_reward, vorkath_info = self._process_vorkath_turn()
        reward += vorkath_reward
        info["damage_taken"] += vorkath_info.get("damage", 0)
        
        if self._player_in_acid():
            damage = self.ACID_DAMAGE_PER_TICK
            self.player.current_hp -= damage
            info["damage_taken"] += damage
            reward -= 0.5
        
        if self.vorkath.zombified_spawn_active:
            # Spawn damages player every tick until killed with Crumble Undead
            spawn_damage = self.rng.integers(5, 12)
            self.player.current_hp -= spawn_damage
            reward -= 0.3  # Pressure to kill spawn quickly
            
            self.vorkath.spawn_ticks_until_heal -= 1
            if self.vorkath.spawn_ticks_until_heal <= 0:
                # Player failed to kill spawn - Vorkath heals!
                heal = min(self.ZOMBIFIED_SPAWN_HEAL, self.vorkath.max_hp - self.vorkath.current_hp)
                self.vorkath.current_hp += heal
                self.vorkath.zombified_spawn_active = False
                self.vorkath.zombified_spawn_hp = 0
                self.vorkath.phase = VorkathPhase.NORMAL
                self.vorkath.is_attackable = True
                reward -= 15.0  # Big penalty

        done = False
        if self.player.current_hp <= 0:
            done = True
            reward -= 50.0
            info["result"] = "death"
        elif self.vorkath.current_hp <= 0:
            done = True
            reward += 100.0
            info["result"] = "kill"
        
        return reward, done, info
    
    def _process_player_action(self, action: VorkathAction) -> Tuple[float, dict]:
        reward = 0.0
        info = {"damage_dealt": 0}
        
        if self.player.attack_cooldown > 0:
            self.player.attack_cooldown -= 1
        if self.player.eat_cooldown > 0:
            self.player.eat_cooldown -= 1
        
        if self.player.frozen and action not in [
            VorkathAction.PRAY_MAGE, VorkathAction.PRAY_RANGE, 
            VorkathAction.PRAY_OFF, VorkathAction.TOGGLE_RIGOUR,
            VorkathAction.EAT_FOOD, VorkathAction.DRINK_PRAYER
        ]:
            return -0.1, info
        
        if action == VorkathAction.WAIT:
            pass
        
        elif action == VorkathAction.ATTACK_RANGED:
            if self.player.attack_cooldown > 0:
                return -0.1, info
            if not self.vorkath.is_attackable:
                return -0.1, info
            if self.vorkath.zombified_spawn_active:
                return -0.2, info
            
            if self.rng.random() < 0.92:
                damage = self.rng.integers(0, self.player.ranged_max_hit + 1)
                if self.player.rigour:
                    damage = int(damage * 1.23)
                self.vorkath.current_hp -= damage
                info["damage_dealt"] = damage
                reward += damage / 10.0
            
            self.player.attack_cooldown = 5
        
        elif action == VorkathAction.ATTACK_SPEC:
            if self.player.special_attack < 50:
                return -0.1, info
            if self.player.attack_cooldown > 0:
                return -0.1, info
            
            self.player.special_attack -= 50
            if self.rng.random() < 0.75:
                damage = self.rng.integers(30, 60)
                self.vorkath.current_hp -= damage
                info["damage_dealt"] = damage
                reward += damage / 5.0
            
            self.player.attack_cooldown = 6
        
        elif action == VorkathAction.CAST_CRUMBLE_UNDEAD:
            if not self.vorkath.zombified_spawn_active:
                return -0.2, info
            if self.player.crumble_undead_casts <= 0:
                return -0.1, info
            
            self.player.crumble_undead_casts -= 1
            self.vorkath.zombified_spawn_active = False
            self.vorkath.zombified_spawn_hp = 0
            self.vorkath.is_attackable = True
            self.vorkath.phase = VorkathPhase.NORMAL
            reward += 5.0
        
        elif action == VorkathAction.PRAY_MAGE:
            self.player.protect_magic = True
            self.player.protect_ranged = False
        
        elif action == VorkathAction.PRAY_RANGE:
            self.player.protect_ranged = True
            self.player.protect_magic = False
        
        elif action == VorkathAction.PRAY_OFF:
            self.player.protect_magic = False
            self.player.protect_ranged = False
        
        elif action == VorkathAction.TOGGLE_RIGOUR:
            self.player.rigour = not self.player.rigour
        
        elif action in [VorkathAction.MOVE_NORTH, VorkathAction.MOVE_SOUTH,
                        VorkathAction.MOVE_EAST, VorkathAction.MOVE_WEST]:
            dx, dy = {
                VorkathAction.MOVE_NORTH: (0, -1),
                VorkathAction.MOVE_SOUTH: (0, 1),
                VorkathAction.MOVE_EAST: (1, 0),
                VorkathAction.MOVE_WEST: (-1, 0),
            }[action]
            
            new_x = max(0, min(self.ARENA_SIZE - 1, self.player.position[0] + dx))
            new_y = max(0, min(self.ARENA_SIZE - 1, self.player.position[1] + dy))
            self.player.position = (new_x, new_y)
            
            if self.vorkath.fireball_incoming:
                reward += 2.0
                self.vorkath.fireball_incoming = False
        
        elif action == VorkathAction.WALK_AROUND:
            if self.vorkath.phase == VorkathPhase.ACID:
                self._move_to_safe_tile()
                reward += 0.5
        
        elif action == VorkathAction.EAT_FOOD:
            if self.player.food_count <= 0:
                return -0.1, info
            if self.player.eat_cooldown > 0:
                return -0.1, info
            
            heal = min(22, self.player.max_hp - self.player.current_hp)
            self.player.current_hp += heal
            self.player.food_count -= 1
            self.player.eat_cooldown = 3
        
        elif action == VorkathAction.DRINK_PRAYER:
            if self.player.prayer_pots <= 0:
                return -0.1, info
            
            restore = min(32, self.player.max_prayer - self.player.current_prayer)
            self.player.current_prayer += restore
            self.player.prayer_pots -= 1
        
        elif action == VorkathAction.DRINK_ANTIFIRE:
            self.player.antifire_active = True
        
        elif action == VorkathAction.DRINK_ANTIVENOM:
            self.player.antivenom_active = True
            self.player.venomed = False
        
        return reward, info
    
    def _process_vorkath_turn(self) -> Tuple[float, dict]:
        reward = 0.0
        info = {"damage": 0}
        
        if self.vorkath.attack_cooldown > 0:
            self.vorkath.attack_cooldown -= 1
            return reward, info
        
        if self.vorkath.phase == VorkathPhase.ACID:
            return self._process_acid_phase()
        elif self.vorkath.phase == VorkathPhase.ZOMBIFIED_SPAWN:
            return reward, info
        
        if self.vorkath.attacks_until_special <= 0:
            return self._start_special_phase()
        
        self.vorkath.attacks_until_special -= 1
        attack_roll = self.rng.random()
        damage = 0
        
        if attack_roll < 0.35:
            damage = self._calculate_attack_damage(
                VorkathAttack.MAGIC_DRAGONFIRE,
                self.MAGIC_MAX_HIT,
                self.player.protect_magic
            )
            if damage > 0:
                reward -= 0.3
        
        elif attack_roll < 0.65:
            damage = self._calculate_attack_damage(
                VorkathAttack.RANGED_DRAGONFIRE,
                self.RANGED_MAX_HIT,
                self.player.protect_ranged
            )
            if damage > 0:
                reward -= 0.3
        
        elif attack_roll < 0.75:
            damage = self._calculate_attack_damage(
                VorkathAttack.PINK_DRAGONFIRE,
                self.PINK_DRAGONFIRE_MAX,
                self.player.protect_magic
            )
            self.player.current_prayer = max(0, self.player.current_prayer - 30)
        
        elif attack_roll < 0.85:
            self.vorkath.fireball_incoming = True
            self.vorkath.fireball_ticks = 3
            damage = 0
        
        elif attack_roll < 0.95:
            damage = self.FREEZE_BREATH_DAMAGE if not self.player.antifire_active else 10
            self.player.frozen = True
            self.player.freeze_ticks = 4
        
        else:
            self._spawn_acid_pools(3)
            damage = self._calculate_attack_damage(
                VorkathAttack.RANGED_DRAGONFIRE,
                self.RANGED_MAX_HIT,
                self.player.protect_ranged
            )
        
        self.player.current_hp -= damage
        info["damage"] = damage
        
        if damage == 0 and attack_roll < 0.65:
            reward += 0.5
        
        self.vorkath.attack_cooldown = 5
        
        if self.vorkath.fireball_incoming:
            self.vorkath.fireball_ticks -= 1
            if self.vorkath.fireball_ticks <= 0:
                if self.vorkath.fireball_incoming:
                    fb_damage = self.FIREBALL_DAMAGE
                    self.player.current_hp -= fb_damage
                    info["damage"] += fb_damage
                    reward -= 10.0
                self.vorkath.fireball_incoming = False
        
        return reward, info
    
    def _start_special_phase(self) -> Tuple[float, dict]:
        if self.vorkath.next_special_is_acid:
            self.vorkath.phase = VorkathPhase.ACID
            self.vorkath.is_attackable = False
            self.vorkath.acid_phase_ticks = 25
            self._spawn_acid_pools(30)
        else:
            self.vorkath.phase = VorkathPhase.ZOMBIFIED_SPAWN
            self.vorkath.zombified_spawn_active = True
            self.vorkath.zombified_spawn_hp = 50
            self.vorkath.spawn_ticks_until_heal = 15
        
        self.vorkath.next_special_is_acid = not self.vorkath.next_special_is_acid
        self.vorkath.attacks_until_special = 6
        
        return 0.0, {"damage": 0}
    
    def _process_acid_phase(self) -> Tuple[float, dict]:
        self.vorkath.acid_phase_ticks -= 1
        
        if self.vorkath.acid_phase_ticks <= 0:
            self.vorkath.phase = VorkathPhase.NORMAL
            self.vorkath.is_attackable = True
            self.vorkath.acid_pools = []
        
        return 0.0, {"damage": 0}
    
    def _calculate_attack_damage(self, attack: VorkathAttack, max_hit: int, protected: bool) -> int:
        if protected:
            return 0
        
        if not self.player.antifire_active:
            max_hit = int(max_hit * 1.5)
        
        return self.rng.integers(0, max_hit + 1)
    
    def _spawn_acid_pools(self, count: int):
        self.vorkath.acid_pools = []
        for _ in range(count):
            x = self.rng.integers(0, self.ARENA_SIZE)
            y = self.rng.integers(0, self.ARENA_SIZE)
            self.vorkath.acid_pools.append((x, y))
    
    def _player_in_acid(self) -> bool:
        return self.player.position in self.vorkath.acid_pools
    
    def _move_to_safe_tile(self):
        x, y = self.player.position
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_pos = (x + dx, y + dy)
            if (0 <= new_pos[0] < self.ARENA_SIZE and 
                0 <= new_pos[1] < self.ARENA_SIZE and
                new_pos not in self.vorkath.acid_pools):
                self.player.position = new_pos
                return
    
    def _update_player_status(self):
        drain = 0
        if self.player.protect_magic or self.player.protect_ranged:
            drain += 1
        if self.player.rigour:
            drain += 2
        self.player.current_prayer = max(0, self.player.current_prayer - drain // 2)
        
        if self.player.current_prayer <= 0:
            self.player.protect_magic = False
            self.player.protect_ranged = False
            self.player.rigour = False
        
        if self.player.frozen:
            self.player.freeze_ticks -= 1
            if self.player.freeze_ticks <= 0:
                self.player.frozen = False
        
        if self.player.venomed:
            self.player.current_hp -= 6
    
    def get_observation(self) -> np.ndarray:
        obs = np.zeros(40, dtype=np.float32)
        
        obs[0] = self.player.current_hp / self.player.max_hp
        obs[1] = self.player.current_prayer / self.player.max_prayer
        obs[2] = self.player.special_attack / 100.0
        obs[3] = self.player.food_count / 8.0
        obs[4] = self.player.prayer_pots / 4.0
        obs[5] = self.player.crumble_undead_casts / 10.0
        obs[6] = 1.0 if self.player.antifire_active else 0.0
        obs[7] = 1.0 if self.player.antivenom_active else 0.0
        
        obs[8] = self.player.position[0] / self.ARENA_SIZE
        obs[9] = self.player.position[1] / self.ARENA_SIZE
        
        obs[10] = 1.0 if self.player.protect_magic else 0.0
        obs[11] = 1.0 if self.player.protect_ranged else 0.0
        obs[12] = 1.0 if self.player.rigour else 0.0
        obs[13] = 1.0 if self.player.frozen else 0.0
        
        obs[14] = min(self.player.attack_cooldown / 6.0, 1.0)
        obs[15] = min(self.player.eat_cooldown / 3.0, 1.0)
        
        obs[16] = self.vorkath.current_hp / self.vorkath.max_hp
        obs[17] = float(self.vorkath.phase) / 2.0
        obs[18] = 1.0 if self.vorkath.is_attackable else 0.0
        obs[19] = min(self.vorkath.attack_cooldown / 6.0, 1.0)
        obs[20] = self.vorkath.attacks_until_special / 6.0
        
        obs[21] = 1.0 if self.vorkath.fireball_incoming else 0.0
        obs[22] = self.vorkath.fireball_ticks / 3.0 if self.vorkath.fireball_incoming else 0.0
        obs[23] = 1.0 if self.vorkath.zombified_spawn_active else 0.0
        obs[24] = self.vorkath.spawn_ticks_until_heal / 15.0 if self.vorkath.zombified_spawn_active else 0.0
        obs[25] = len(self.vorkath.acid_pools) / 30.0
        obs[26] = 1.0 if self._player_in_acid() else 0.0
        
        px, py = self.player.position
        min_dist_n = min_dist_s = min_dist_e = min_dist_w = self.ARENA_SIZE
        for ax, ay in self.vorkath.acid_pools:
            if ax == px and ay < py:
                min_dist_n = min(min_dist_n, py - ay)
            elif ax == px and ay > py:
                min_dist_s = min(min_dist_s, ay - py)
            elif ay == py and ax > px:
                min_dist_e = min(min_dist_e, ax - px)
            elif ay == py and ax < px:
                min_dist_w = min(min_dist_w, px - ax)
        
        obs[27] = min_dist_n / self.ARENA_SIZE
        obs[28] = min_dist_s / self.ARENA_SIZE
        obs[29] = min_dist_e / self.ARENA_SIZE
        obs[30] = min_dist_w / self.ARENA_SIZE
        
        obs[31] = min(self.tick / 500.0, 1.0)
        
        return obs


class VorkathEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 2}
    
    def __init__(self, render_mode=None, max_ticks=500):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_ticks = max_ticks
        
        self.mechanics = VorkathMechanics()
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(40,), dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(len(VorkathAction))
        
        self.total_damage_dealt = 0
        self.total_damage_taken = 0
        self.kills = 0
        self.deaths = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if seed is not None:
            self.mechanics.rng = np.random.default_rng(seed)
        
        self.mechanics.reset()
        self.total_damage_dealt = 0
        self.total_damage_taken = 0
        
        return self.mechanics.get_observation(), self._get_info()
    
    def step(self, action):
        reward, done, info = self.mechanics.step(VorkathAction(action))
        
        self.total_damage_dealt += info.get("damage_dealt", 0)
        self.total_damage_taken += info.get("damage_taken", 0)
        
        if info.get("result") == "kill":
            self.kills += 1
        elif info.get("result") == "death":
            self.deaths += 1
        
        truncated = self.mechanics.tick >= self.max_ticks
        
        return self.mechanics.get_observation(), reward, done, truncated, self._get_info()
    
    def _get_info(self):
        return {
            "tick": self.mechanics.tick,
            "player_hp": self.mechanics.player.current_hp,
            "vorkath_hp": self.mechanics.vorkath.current_hp,
            "phase": self.mechanics.vorkath.phase.name,
            "damage_dealt": self.total_damage_dealt,
            "damage_taken": self.total_damage_taken,
            "kills": self.kills,
            "deaths": self.deaths,
        }
    
    def render(self):
        if self.render_mode in ["human", "ansi"]:
            v = self.mechanics.vorkath
            p = self.mechanics.player
            
            print(f"\n{'='*50}")
            print(f"Tick {self.mechanics.tick} | Phase: {v.phase.name}")
            print(f"{'='*50}")
            
            hp_pct = v.current_hp / v.max_hp
            hp_bar = '█' * int(hp_pct * 20) + '░' * (20 - int(hp_pct * 20))
            print(f"Vorkath: [{hp_bar}] {v.current_hp}/{v.max_hp}")
            
            if v.fireball_incoming:
                print(f"  ⚠️  FIREBALL INCOMING ({v.fireball_ticks} ticks)")
            if v.zombified_spawn_active:
                print(f"  💀 ZOMBIFIED SPAWN ({v.spawn_ticks_until_heal} ticks)")
            if v.phase == VorkathPhase.ACID:
                print(f"  🟢 ACID PHASE ({v.acid_phase_ticks} ticks, {len(v.acid_pools)} pools)")
            
            hp_pct = p.current_hp / p.max_hp
            hp_bar = '█' * int(hp_pct * 20) + '░' * (20 - int(hp_pct * 20))
            pray_pct = p.current_prayer / p.max_prayer
            pray_bar = '█' * int(pray_pct * 10) + '░' * (10 - int(pray_pct * 10))
            print(f"\nPlayer: [{hp_bar}] {p.current_hp}/{p.max_hp}")
            print(f"Prayer: [{pray_bar}] {p.current_prayer}/{p.max_prayer}")
            
            prayers = []
            if p.protect_magic:
                prayers.append("🛡️Mage")
            if p.protect_ranged:
                prayers.append("🛡️Range")
            if p.rigour:
                prayers.append("⚔️Rigour")
            print(f"Active: {' '.join(prayers) if prayers else 'None'}")
            
            print(f"Food: {p.food_count} | Prayer pots: {p.prayer_pots} | Spec: {p.special_attack}%")
            
            if p.frozen:
                print(f"  ❄️ FROZEN ({p.freeze_ticks} ticks)")
            if self.mechanics._player_in_acid():
                print(f"  ⚠️ STANDING IN ACID")
            
            print(f"\nDamage dealt: {self.total_damage_dealt} | Taken: {self.total_damage_taken}")


if __name__ == "__main__":
    print("Testing Vorkath environment with full mechanics...")
    
    env = VorkathEnv(render_mode="human")
    obs, info = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    actions = [
        VorkathAction.PRAY_MAGE,
        VorkathAction.TOGGLE_RIGOUR,
        VorkathAction.ATTACK_RANGED,
        VorkathAction.ATTACK_RANGED,
        VorkathAction.ATTACK_RANGED,
        VorkathAction.PRAY_RANGE,
        VorkathAction.ATTACK_RANGED,
        VorkathAction.ATTACK_RANGED,
        VorkathAction.PRAY_MAGE,
        VorkathAction.ATTACK_RANGED,
    ]
    
    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        if (i + 1) % 3 == 0:
            env.render()
        
        if terminated:
            print(f"\nFight ended: {info}")
            break
    
    print("\n✓ Vorkath environment working!")
