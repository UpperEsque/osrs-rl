package com.osrsrl;

import net.runelite.api.*;
import net.runelite.api.coords.WorldPoint;
import net.runelite.api.widgets.Widget;
import net.runelite.api.widgets.WidgetInfo;
import lombok.extern.slf4j.Slf4j;

import java.util.Arrays;

/**
 * Extracts Vorkath fight state for RL model
 * 
 * Observation vector (35 values):
 * [0-4]   Player stats: HP%, Prayer%, Spec%, Attack cooldown, Eat cooldown
 * [5-9]   Player prayers: Protect Mage, Protect Range, Rigour, Augury, Redemption
 * [10-14] Player resources: Food count, Prayer pots, Antifire, Antivenom, Spec weapon
 * [15-19] Vorkath stats: HP%, Phase (0-2), Attack tick, Distance, Is attacking
 * [20-24] Vorkath mechanics: Fireball incoming, Acid phase, Spawn active, Frozen, Acid count
 * [25-29] Vorkath attack: Current attack type (5 one-hot)
 * [30-34] Player position: X offset, Y offset, In acid, Safe spot, Moving
 */
@Slf4j
public class VorkathStateExtractor {
    
    private static final int VORKATH_ID = 8061; // Vorkath NPC ID
    private static final int ZOMBIFIED_SPAWN_ID = 8063;
    
    private final Client client;
    
    // Track Vorkath state
    private int lastVorkathAnimation = -1;
    private int vorkathAttackTick = 0;
    private boolean fireballIncoming = false;
    private boolean acidPhaseActive = false;
    private boolean spawnActive = false;
    
    public VorkathStateExtractor(Client client) {
        this.client = client;
    }
    
    public float[] extract() {
        float[] obs = new float[35];
        Arrays.fill(obs, 0f);
        
        Player player = client.getLocalPlayer();
        if (player == null) {
            return obs;
        }
        
        // Find Vorkath
        NPC vorkath = findVorkath();
        
        // Player stats [0-4]
        obs[0] = getStat(Skill.HITPOINTS) / (float) getRealLevel(Skill.HITPOINTS);
        obs[1] = getStat(Skill.PRAYER) / (float) getRealLevel(Skill.PRAYER);
        obs[2] = client.getVarpValue(VarPlayer.SPECIAL_ATTACK_PERCENT) / 1000f;
        obs[3] = Math.min(player.getAnimation() != -1 ? 1f : 0f, 1f); // Attacking
        obs[4] = 0f; // Eat cooldown (tracked by game tick)
        
        // Player prayers [5-9]
        obs[5] = client.isPrayerActive(Prayer.PROTECT_FROM_MAGIC) ? 1f : 0f;
        obs[6] = client.isPrayerActive(Prayer.PROTECT_FROM_MISSILES) ? 1f : 0f;
        obs[7] = client.isPrayerActive(Prayer.RIGOUR) ? 1f : 0f;
        obs[8] = client.isPrayerActive(Prayer.AUGURY) ? 1f : 0f;
        obs[9] = client.isPrayerActive(Prayer.REDEMPTION) ? 1f : 0f;
        
        // Player resources [10-14]
        obs[10] = countFood() / 28f;
        obs[11] = countItem(ItemID.PRAYER_POTION4, ItemID.PRAYER_POTION3, ItemID.PRAYER_POTION2, ItemID.PRAYER_POTION1) / 8f;
        obs[12] = countItem(ItemID.EXTENDED_SUPER_ANTIFIRE4, ItemID.EXTENDED_SUPER_ANTIFIRE3, ItemID.EXTENDED_SUPER_ANTIFIRE2, ItemID.EXTENDED_SUPER_ANTIFIRE1) / 4f;
        obs[13] = countItem(ItemID.ANTIVENOM4, ItemID.ANTIVENOM3, ItemID.ANTIVENOM2, ItemID.ANTIVENOM1) / 4f;
        obs[14] = hasSpecWeapon() ? 1f : 0f;
        
        // Vorkath stats [15-19]
        if (vorkath != null) {
            int maxHp = 750;
            int currentHp = vorkath.getHealthRatio() * maxHp / vorkath.getHealthScale();
            obs[15] = currentHp / (float) maxHp;
            obs[16] = getVorkathPhase(vorkath) / 2f;
            obs[17] = vorkathAttackTick / 10f;
            obs[18] = player.getWorldLocation().distanceTo(vorkath.getWorldLocation()) / 10f;
            obs[19] = vorkath.getAnimation() != -1 ? 1f : 0f;
            
            // Update attack tracking
            updateVorkathAttack(vorkath);
        }
        
        // Vorkath mechanics [20-24]
        obs[20] = fireballIncoming ? 1f : 0f;
        obs[21] = acidPhaseActive ? 1f : 0f;
        obs[22] = spawnActive ? 1f : 0f;
        obs[23] = isPlayerFrozen() ? 1f : 0f;
        obs[24] = countAcidPools() / 30f;
        
        // Vorkath attack type one-hot [25-29]
        int attackType = getVorkathAttackType(vorkath);
        if (attackType >= 0 && attackType < 5) {
            obs[25 + attackType] = 1f;
        }
        
        // Player position [30-34]
        if (vorkath != null) {
            WorldPoint vorkLoc = vorkath.getWorldLocation();
            WorldPoint playerLoc = player.getWorldLocation();
            obs[30] = (playerLoc.getX() - vorkLoc.getX()) / 10f;
            obs[31] = (playerLoc.getY() - vorkLoc.getY()) / 10f;
        }
        obs[32] = isInAcid() ? 1f : 0f;
        obs[33] = isInSafeSpot() ? 1f : 0f;
        obs[34] = player.getPoseAnimation() != player.getIdlePoseAnimation() ? 1f : 0f;
        
        return obs;
    }
    
    private NPC findVorkath() {
        for (NPC npc : client.getNpcs()) {
            if (npc.getId() == VORKATH_ID || npc.getName() != null && npc.getName().equals("Vorkath")) {
                return npc;
            }
        }
        return null;
    }
    
    private NPC findZombifiedSpawn() {
        for (NPC npc : client.getNpcs()) {
            if (npc.getId() == ZOMBIFIED_SPAWN_ID) {
                spawnActive = true;
                return npc;
            }
        }
        spawnActive = false;
        return null;
    }
    
    private void updateVorkathAttack(NPC vorkath) {
        if (vorkath == null) return;
        
        int anim = vorkath.getAnimation();
        if (anim != lastVorkathAnimation) {
            lastVorkathAnimation = anim;
            
            // Vorkath animations
            switch (anim) {
                case 7952: // Regular dragonfire
                case 7951: // Ranged attack
                    vorkathAttackTick = 0;
                    fireballIncoming = false;
                    break;
                case 7960: // Fireball (purple)
                    fireballIncoming = true;
                    vorkathAttackTick = 0;
                    break;
                case 7957: // Acid phase start
                    acidPhaseActive = true;
                    vorkathAttackTick = 0;
                    break;
                case 7950: // Normal stance
                    acidPhaseActive = false;
                    break;
            }
        }
        
        vorkathAttackTick++;
    }
    
    private int getVorkathPhase(NPC vorkath) {
        if (acidPhaseActive) return 1;
        if (spawnActive) return 2;
        return 0; // Normal phase
    }
    
    private int getVorkathAttackType(NPC vorkath) {
        // 0: Magic, 1: Ranged, 2: Fireball, 3: Acid, 4: Spawn
        if (fireballIncoming) return 2;
        if (acidPhaseActive) return 3;
        if (spawnActive) return 4;
        
        if (vorkath != null) {
            int anim = vorkath.getAnimation();
            if (anim == 7952) return 0; // Magic
            if (anim == 7951) return 1; // Ranged
        }
        return 0;
    }
    
    private int getStat(Skill skill) {
        return client.getBoostedSkillLevel(skill);
    }
    
    private int getRealLevel(Skill skill) {
        return client.getRealSkillLevel(skill);
    }
    
    private int countFood() {
        int count = 0;
        Widget inventory = client.getWidget(WidgetInfo.INVENTORY);
        if (inventory == null) return 0;
        
        for (Widget item : inventory.getDynamicChildren()) {
            int id = item.getItemId();
            if (isFood(id)) count++;
        }
        return count;
    }
    
    private boolean isFood(int itemId) {
        // Common food IDs
        return itemId == ItemID.SHARK || itemId == ItemID.MANTA_RAY || 
               itemId == ItemID.ANGLERFISH || itemId == ItemID.DARK_CRAB ||
               itemId == ItemID.COOKED_KARAMBWAN || itemId == ItemID.TUNA_POTATO ||
               itemId == ItemID.SARADOMIN_BREW4 || itemId == ItemID.SARADOMIN_BREW3 ||
               itemId == ItemID.SARADOMIN_BREW2 || itemId == ItemID.SARADOMIN_BREW1;
    }
    
    private int countItem(int... itemIds) {
        int count = 0;
        Widget inventory = client.getWidget(WidgetInfo.INVENTORY);
        if (inventory == null) return 0;
        
        for (Widget item : inventory.getDynamicChildren()) {
            for (int id : itemIds) {
                if (item.getItemId() == id) count++;
            }
        }
        return count;
    }
    
    private boolean hasSpecWeapon() {
        // Check equipped weapon
        Widget weapon = client.getWidget(WidgetInfo.EQUIPMENT_WEAPON);
        if (weapon != null) {
            int id = weapon.getItemId();
            // Blowpipe, BGS, Dragon claws, etc.
            return id == ItemID.TOXIC_BLOWPIPE || id == ItemID.BANDOS_GODSWORD ||
                   id == ItemID.DRAGON_CLAWS || id == ItemID.DRAGON_WARHAMMER;
        }
        return false;
    }
    
    private boolean isPlayerFrozen() {
        Player player = client.getLocalPlayer();
        // Check for frozen animation/graphic
        return player != null && player.getGraphic() == 369; // Ice barrage graphic
    }
    
    private int countAcidPools() {
        int count = 0;
        for (GraphicsObject go : client.getGraphicsObjects()) {
            if (go.getId() == 1483) { // Acid pool graphic
                count++;
            }
        }
        return count;
    }
    
    private boolean isInAcid() {
        Player player = client.getLocalPlayer();
        if (player == null) return false;
        
        WorldPoint playerLoc = player.getWorldLocation();
        for (GraphicsObject go : client.getGraphicsObjects()) {
            if (go.getId() == 1483) {
                WorldPoint acidLoc = WorldPoint.fromLocal(client, go.getLocation());
                if (playerLoc.equals(acidLoc)) {
                    return true;
                }
            }
        }
        return false;
    }
    
    private boolean isInSafeSpot() {
        // Vorkath safe spots are specific tiles
        Player player = client.getLocalPlayer();
        if (player == null) return false;
        
        NPC vorkath = findVorkath();
        if (vorkath == null) return false;
        
        int dist = player.getWorldLocation().distanceTo(vorkath.getWorldLocation());
        return dist >= 6 && dist <= 8; // Optimal range
    }
    
    // Reset state when fight ends
    public void reset() {
        lastVorkathAnimation = -1;
        vorkathAttackTick = 0;
        fireballIncoming = false;
        acidPhaseActive = false;
        spawnActive = false;
    }
}
