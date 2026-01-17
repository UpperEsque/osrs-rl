"""
Zulrah - Serpentine Boss with Rotation Patterns

Zulrah has 4 rotations, each with predictable phase sequences.
Player must:
1. Know which rotation they're in
2. Move to correct position for each phase
3. Switch prayers (mage for green, range for blue)
4. Switch gear (range for green, mage for blue)
5. Avoid venom clouds and snakelings
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import IntEnum
import gymnasium as gym
from gymnasium import spaces


class ZulrahPhase(IntEnum):
    GREEN_RANGE = 0   # Attacks with range, weak to range
    GREEN_MELEE = 1   # Melee phase, run away
    BLUE_MAGE = 2     # Attacks with mage, weak to mage
    RED_RANGE = 3     # Attacks with range, spawns snakelings


class ZulrahAction(IntEnum):
    WAIT = 0
    ATTACK_RANGED = 1
    ATTACK_MAGIC = 2
    ATTACK_SPEC = 3
    PRAY_RANGED = 4
    PRAY_MAGIC = 5
    PRAY_OFF = 6
    TOGGLE_RIGOUR = 7
    TOGGLE_AUGURY = 8
    MOVE_EAST = 9
    MOVE_WEST = 10
    MOVE_PILLAR = 11   # Hide behind pillar
    EAT_FOOD = 12
    DRINK_RESTORE = 13
    DRINK_ANTIVENOM = 14
    SWITCH_RANGE_GEAR = 15
    SWITCH_MAGE_GEAR = 16


# Zulrah rotations (simplified - 4 main rotations)
ROTATIONS = [
    # Rotation 1
    [ZulrahPhase.GREEN_RANGE, ZulrahPhase.BLUE_MAGE, ZulrahPhase.GREEN_MELEE, 
     ZulrahPhase.BLUE_MAGE, ZulrahPhase.GREEN_RANGE, ZulrahPhase.BLUE_MAGE],
    # Rotation 2  
    [ZulrahPhase.GREEN_RANGE, ZulrahPhase.BLUE_MAGE, ZulrahPhase.BLUE_MAGE,
     ZulrahPhase.GREEN_MELEE, ZulrahPhase.BLUE_MAGE, ZulrahPhase.GREEN_RANGE],
    # Rotation 3
    [ZulrahPhase.GREEN_RANGE, ZulrahPhase.BLUE_MAGE, ZulrahPhase.GREEN_RANGE,
     ZulrahPhase.BLUE_MAGE, ZulrahPhase.GREEN_MELEE, ZulrahPhase.GREEN_RANGE],
    # Rotation 4
    [ZulrahPhase.GREEN_RANGE, ZulrahPhase.GREEN_MELEE, ZulrahPhase.BLUE_MAGE,
     ZulrahPhase.GREEN_RANGE, ZulrahPhase.BLUE_MAGE, ZulrahPhase.GREEN_RANGE],
]

# Positions for each phase (simplified: east=1, west=-1, center=0)
PHASE_POSITIONS = {
    ZulrahPhase.GREEN_RANGE: 1,    # East
    ZulrahPhase.GREEN_MELEE: 0,    # Run around
    ZulrahPhase.BLUE_MAGE: -1,     # West
    ZulrahPhase.RED_RANGE: 1,      # East
}


@dataclass
class ZulrahState:
    current_hp: int = 500
    max_hp: int = 500
    rotation: int = 0
    phase_index: int = 0
    current_phase: ZulrahPhase = ZulrahPhase.GREEN_RANGE
    phase_ticks: int = 0
    phase_duration: int = 40  # Ticks per phase
    attack_cooldown: int = 0
    
    # Mechanics
    venom_clouds: List[int] = field(default_factory=list)  # Positions with venom
    snakelings: int = 0
    is_diving: bool = False  # Between phases
    dive_ticks: int = 0
    
    def reset(self, rotation: int = None, rng=None):
        self.current_hp = self.max_hp
        if rotation is not None:
            self.rotation = rotation
        elif rng:
            self.rotation = rng.integers(0, 4)
        self.phase_index = 0
        self.current_phase = ROTATIONS[self.rotation][0]
        self.phase_ticks = 0
        self.attack_cooldown = 0
        self.venom_clouds = []
        self.snakelings = 0
        self.is_diving = False
        self.dive_ticks = 0


@dataclass
class ZulrahPlayerState:
    current_hp: int = 99
    max_hp: int = 99
    current_prayer: int = 99
    max_prayer: int = 99
    special_attack: int = 100
    
    position: int = 0  # -1=west, 0=center, 1=east
    
    protect_ranged: bool = False
    protect_magic: bool = False
    rigour: bool = False
    augury: bool = False
    
    using_range_gear: bool = True
    
    attack_cooldown: int = 0
    eat_cooldown: int = 0
    
    food_count: int = 8
    restores: int = 4
    antivenoms: int = 2
    
    venomed: bool = False
    venom_damage: int = 0
    
    ranged_max: int = 45
    magic_max: int = 42
    
    def reset(self):
        self.current_hp = self.max_hp
        self.current_prayer = self.max_prayer
        self.special_attack = 100
        self.position = 0
        self.protect_ranged = False
        self.protect_magic = False
        self.rigour = False
        self.augury = False
        self.using_range_gear = True
        self.attack_cooldown = 0
        self.eat_cooldown = 0
        self.food_count = 8
        self.restores = 4
        self.antivenoms = 2
        self.venomed = False
        self.venom_damage = 0


class ZulrahMechanics:
    RANGE_DAMAGE = 41
    MAGE_DAMAGE = 41
    MELEE_DAMAGE = 50
    VENOM_TICK_DAMAGE = 6
    SNAKELING_DAMAGE = 15
    
    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()
        self.boss = ZulrahState()
        self.player = ZulrahPlayerState()
        self.tick = 0
    
    def reset(self):
        self.boss.reset(rng=self.rng)
        self.player.reset()
        self.tick = 0
        return self.boss, self.player
    
    def step(self, action: ZulrahAction):
        self.tick += 1
        reward = 0.0
        info = {"damage_dealt": 0, "damage_taken": 0}
        
        # Process player action
        reward += self._process_player(action, info)
        
        # Update player status
        self._update_player()
        
        # Process boss
        reward += self._process_boss(info)
        
        # Check phase transition
        self._check_phase_transition()
        
        # Done check
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
        
        # Can't attack while Zulrah is diving
        if self.boss.is_diving:
            if action in [ZulrahAction.ATTACK_RANGED, ZulrahAction.ATTACK_MAGIC]:
                return -0.1
        
        if action == ZulrahAction.ATTACK_RANGED:
            if self.player.attack_cooldown > 0:
                return -0.1
            
            # Check if right gear/phase
            effective = self.boss.current_phase in [ZulrahPhase.GREEN_RANGE, ZulrahPhase.GREEN_MELEE]
            
            if self.rng.random() < (0.90 if effective else 0.70):
                max_hit = self.player.ranged_max
                if self.player.rigour:
                    max_hit = int(max_hit * 1.23)
                if not self.player.using_range_gear:
                    max_hit = int(max_hit * 0.7)  # Wrong gear penalty
                
                dmg = self.rng.integers(0, max_hit + 1)
                self.boss.current_hp -= dmg
                info["damage_dealt"] = dmg
                reward += dmg / 10
                
                if effective:
                    reward += 0.3  # Bonus for right style
            
            self.player.attack_cooldown = 4
        
        elif action == ZulrahAction.ATTACK_MAGIC:
            if self.player.attack_cooldown > 0:
                return -0.1
            
            effective = self.boss.current_phase in [ZulrahPhase.BLUE_MAGE, ZulrahPhase.RED_RANGE]
            
            if self.rng.random() < (0.90 if effective else 0.65):
                max_hit = self.player.magic_max
                if self.player.augury:
                    max_hit = int(max_hit * 1.04)
                if self.player.using_range_gear:
                    max_hit = int(max_hit * 0.6)  # Wrong gear penalty
                
                dmg = self.rng.integers(0, max_hit + 1)
                self.boss.current_hp -= dmg
                info["damage_dealt"] = dmg
                reward += dmg / 10
                
                if effective:
                    reward += 0.3
            
            self.player.attack_cooldown = 5
        
        elif action == ZulrahAction.PRAY_RANGED:
            self.player.protect_ranged = True
            self.player.protect_magic = False
        
        elif action == ZulrahAction.PRAY_MAGIC:
            self.player.protect_magic = True
            self.player.protect_ranged = False
        
        elif action == ZulrahAction.PRAY_OFF:
            self.player.protect_ranged = False
            self.player.protect_magic = False
        
        elif action == ZulrahAction.TOGGLE_RIGOUR:
            self.player.rigour = not self.player.rigour
            self.player.augury = False
        
        elif action == ZulrahAction.TOGGLE_AUGURY:
            self.player.augury = not self.player.augury
            self.player.rigour = False
        
        elif action == ZulrahAction.MOVE_EAST:
            self.player.position = 1
            # Reward for correct position
            correct_pos = PHASE_POSITIONS.get(self.boss.current_phase, 0)
            if self.player.position == correct_pos:
                reward += 0.5
        
        elif action == ZulrahAction.MOVE_WEST:
            self.player.position = -1
            correct_pos = PHASE_POSITIONS.get(self.boss.current_phase, 0)
            if self.player.position == correct_pos:
                reward += 0.5
        
        elif action == ZulrahAction.MOVE_PILLAR:
            # Hide behind pillar during melee phase
            if self.boss.current_phase == ZulrahPhase.GREEN_MELEE:
                self.player.position = 0
                reward += 1.0
        
        elif action == ZulrahAction.SWITCH_RANGE_GEAR:
            self.player.using_range_gear = True
            # Reward for right gear switch
            if self.boss.current_phase in [ZulrahPhase.GREEN_RANGE, ZulrahPhase.GREEN_MELEE]:
                reward += 0.3
        
        elif action == ZulrahAction.SWITCH_MAGE_GEAR:
            self.player.using_range_gear = False
            if self.boss.current_phase in [ZulrahPhase.BLUE_MAGE, ZulrahPhase.RED_RANGE]:
                reward += 0.3
        
        elif action == ZulrahAction.EAT_FOOD:
            if self.player.food_count > 0 and self.player.eat_cooldown <= 0:
                self.player.current_hp = min(self.player.max_hp, self.player.current_hp + 22)
                self.player.food_count -= 1
                self.player.eat_cooldown = 3
        
        elif action == ZulrahAction.DRINK_RESTORE:
            if self.player.restores > 0:
                self.player.current_prayer = min(self.player.max_prayer, self.player.current_prayer + 32)
                self.player.restores -= 1
        
        elif action == ZulrahAction.DRINK_ANTIVENOM:
            if self.player.antivenoms > 0:
                self.player.venomed = False
                self.player.venom_damage = 0
                self.player.antivenoms -= 1
        
        return reward
    
    def _process_boss(self, info):
        reward = 0.0
        damage = 0
        
        if self.boss.is_diving:
            self.boss.dive_ticks -= 1
            if self.boss.dive_ticks <= 0:
                self.boss.is_diving = False
            return reward
        
        if self.boss.attack_cooldown > 0:
            self.boss.attack_cooldown -= 1
            return reward
        
        self.boss.phase_ticks += 1
        
        # Attack based on phase
        phase = self.boss.current_phase
        
        if phase == ZulrahPhase.GREEN_RANGE:
            # Range attacks
            if not self.player.protect_ranged:
                damage = self.rng.integers(15, self.RANGE_DAMAGE + 1)
            else:
                reward += 0.5
            
            # Spawn venom cloud occasionally
            if self.rng.random() < 0.15:
                self.boss.venom_clouds.append(self.player.position)
        
        elif phase == ZulrahPhase.BLUE_MAGE:
            # Mage attacks
            if not self.player.protect_magic:
                damage = self.rng.integers(15, self.MAGE_DAMAGE + 1)
            else:
                reward += 0.5
        
        elif phase == ZulrahPhase.GREEN_MELEE:
            # Melee - must hide behind pillar
            if self.player.position != 0:  # Not at pillar
                damage = self.MELEE_DAMAGE
                reward -= 1.0
            else:
                reward += 0.5
        
        elif phase == ZulrahPhase.RED_RANGE:
            # Range + snakelings
            if not self.player.protect_ranged:
                damage = self.rng.integers(15, self.RANGE_DAMAGE + 1)
            if self.rng.random() < 0.2:
                self.boss.snakelings += 1
        
        # Venom cloud damage
        if self.player.position in self.boss.venom_clouds:
            self.player.venomed = True
            self.player.venom_damage = 6
        
        # Snakeling damage
        if self.boss.snakelings > 0:
            damage += self.boss.snakelings * 5
            # Snakelings die after a bit
            if self.rng.random() < 0.2:
                self.boss.snakelings = max(0, self.boss.snakelings - 1)
        
        self.player.current_hp -= damage
        info["damage_taken"] += damage
        
        self.boss.attack_cooldown = 4
        
        return reward
    
    def _check_phase_transition(self):
        if self.boss.phase_ticks >= self.boss.phase_duration:
            # Move to next phase
            self.boss.phase_index += 1
            if self.boss.phase_index >= len(ROTATIONS[self.boss.rotation]):
                self.boss.phase_index = 0  # Loop rotation
            
            self.boss.current_phase = ROTATIONS[self.boss.rotation][self.boss.phase_index]
            self.boss.phase_ticks = 0
            
            # Zulrah dives between phases
            self.boss.is_diving = True
            self.boss.dive_ticks = 5
            
            # Clear venom clouds on phase change
            self.boss.venom_clouds = []
    
    def _update_player(self):
        # Prayer drain
        drain = 0
        if self.player.protect_ranged or self.player.protect_magic:
            drain += 1
        if self.player.rigour or self.player.augury:
            drain += 2
        self.player.current_prayer = max(0, self.player.current_prayer - drain // 2)
        
        if self.player.current_prayer <= 0:
            self.player.protect_ranged = False
            self.player.protect_magic = False
            self.player.rigour = False
            self.player.augury = False
        
        # Venom damage
        if self.player.venomed:
            self.player.current_hp -= self.player.venom_damage
            self.player.venom_damage = min(20, self.player.venom_damage + 2)
    
    def get_observation(self):
        obs = np.zeros(45, dtype=np.float32)
        
        # Player (0-14)
        obs[0] = self.player.current_hp / self.player.max_hp
        obs[1] = self.player.current_prayer / self.player.max_prayer
        obs[2] = self.player.special_attack / 100
        obs[3] = self.player.food_count / 8
        obs[4] = (self.player.position + 1) / 2  # Normalize -1 to 1 -> 0 to 1
        obs[5] = 1.0 if self.player.protect_ranged else 0.0
        obs[6] = 1.0 if self.player.protect_magic else 0.0
        obs[7] = 1.0 if self.player.rigour else 0.0
        obs[8] = 1.0 if self.player.augury else 0.0
        obs[9] = 1.0 if self.player.using_range_gear else 0.0
        obs[10] = min(self.player.attack_cooldown / 6, 1.0)
        obs[11] = 1.0 if self.player.venomed else 0.0
        
        # Boss (15-30)
        obs[15] = self.boss.current_hp / self.boss.max_hp
        obs[16] = float(self.boss.current_phase) / 3
        obs[17] = self.boss.phase_ticks / self.boss.phase_duration
        obs[18] = 1.0 if self.boss.is_diving else 0.0
        obs[19] = self.boss.snakelings / 3
        obs[20] = len(self.boss.venom_clouds) / 5
        obs[21] = 1.0 if self.player.position in self.boss.venom_clouds else 0.0
        
        # Phase one-hot (22-25)
        obs[22 + int(self.boss.current_phase)] = 1.0
        
        # Rotation hint - next phase (26-29)
        next_idx = (self.boss.phase_index + 1) % len(ROTATIONS[self.boss.rotation])
        next_phase = ROTATIONS[self.boss.rotation][next_idx]
        obs[26 + int(next_phase)] = 1.0
        
        # Correct position for current phase (30)
        correct_pos = PHASE_POSITIONS.get(self.boss.current_phase, 0)
        obs[30] = 1.0 if self.player.position == correct_pos else 0.0
        
        # Correct gear for current phase (31)
        if self.boss.current_phase in [ZulrahPhase.GREEN_RANGE, ZulrahPhase.GREEN_MELEE]:
            obs[31] = 1.0 if self.player.using_range_gear else 0.0
        else:
            obs[31] = 1.0 if not self.player.using_range_gear else 0.0
        
        # Correct prayer for current phase (32)
        if self.boss.current_phase in [ZulrahPhase.GREEN_RANGE, ZulrahPhase.RED_RANGE]:
            obs[32] = 1.0 if self.player.protect_ranged else 0.0
        elif self.boss.current_phase == ZulrahPhase.BLUE_MAGE:
            obs[32] = 1.0 if self.player.protect_magic else 0.0
        else:
            obs[32] = 1.0  # Melee phase, any prayer okay
        
        return obs


class ZulrahEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(self, render_mode=None, max_ticks=600):
        super().__init__()
        self.render_mode = render_mode
        self.max_ticks = max_ticks
        self.mechanics = ZulrahMechanics()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(45,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ZulrahAction))
        self.total_damage_dealt = 0
        self.total_damage_taken = 0
        self.kills = 0
        self.deaths = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed:
            self.mechanics.rng = np.random.default_rng(seed)
        self.mechanics.reset()
        self.total_damage_dealt = 0
        self.total_damage_taken = 0
        return self.mechanics.get_observation(), self._get_info()
    
    def step(self, action):
        reward, done, info = self.mechanics.step(ZulrahAction(action))
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
            "boss_hp": self.mechanics.boss.current_hp,
            "player_hp": self.mechanics.player.current_hp,
            "phase": self.mechanics.boss.current_phase.name,
            "rotation": self.mechanics.boss.rotation,
            "damage_dealt": self.total_damage_dealt,
            "damage_taken": self.total_damage_taken,
            "kills": self.kills,
            "deaths": self.deaths,
        }
    
    def render(self):
        if self.render_mode in ["human", "ansi"]:
            b = self.mechanics.boss
            p = self.mechanics.player
            
            phase_names = {
                ZulrahPhase.GREEN_RANGE: "🟢 GREEN (Range)",
                ZulrahPhase.GREEN_MELEE: "🟢 GREEN (Melee)",
                ZulrahPhase.BLUE_MAGE: "🔵 BLUE (Mage)",
                ZulrahPhase.RED_RANGE: "🔴 RED (Range)",
            }
            
            print(f"\n{'='*55}")
            print(f"ZULRAH | Tick {self.mechanics.tick} | Rotation {b.rotation+1}")
            print(f"Phase: {phase_names[b.current_phase]} ({b.phase_ticks}/{b.phase_duration})")
            print(f"{'='*55}")
            
            hp_bar = '█' * int(b.current_hp/b.max_hp*20) + '░' * (20-int(b.current_hp/b.max_hp*20))
            print(f"Zulrah: [{hp_bar}] {b.current_hp}/{b.max_hp}")
            
            if b.is_diving:
                print(f"  🌊 DIVING ({b.dive_ticks} ticks)")
            if b.snakelings:
                print(f"  🐍 Snakelings: {b.snakelings}")
            if b.venom_clouds:
                print(f"  ☁️ Venom at: {b.venom_clouds}")
            
            hp_bar = '█' * int(p.current_hp/p.max_hp*20) + '░' * (20-int(p.current_hp/p.max_hp*20))
            pos_name = {-1: "West", 0: "Pillar", 1: "East"}[p.position]
            print(f"\nPlayer: [{hp_bar}] {p.current_hp}/{p.max_hp} | Pos: {pos_name}")
            
            prayers = []
            if p.protect_ranged: prayers.append("🛡️Range")
            if p.protect_magic: prayers.append("🛡️Mage")
            if p.rigour: prayers.append("⚔️Rigour")
            if p.augury: prayers.append("⚔️Augury")
            gear = "🏹Range" if p.using_range_gear else "🪄Mage"
            print(f"Gear: {gear} | Prayers: {' '.join(prayers) if prayers else 'None'}")
            
            if p.venomed:
                print(f"  ☠️ VENOMED (dmg: {p.venom_damage})")


if __name__ == "__main__":
    print("Testing detailed Zulrah environment...")
    
    env = ZulrahEnv(render_mode="human")
    obs, info = env.reset()
    
    print(f"Obs shape: {obs.shape}")
    print(f"Actions: {env.action_space.n}")
    
    # Run some actions
    actions = [
        ZulrahAction.PRAY_RANGED,
        ZulrahAction.TOGGLE_RIGOUR,
        ZulrahAction.MOVE_EAST,
        ZulrahAction.ATTACK_RANGED,
        ZulrahAction.ATTACK_RANGED,
        ZulrahAction.ATTACK_RANGED,
    ]
    
    for action in actions:
        obs, reward, done, trunc, info = env.step(action)
        if done:
            break
    
    env.render()
    print(f"\n✓ Zulrah environment working!")
