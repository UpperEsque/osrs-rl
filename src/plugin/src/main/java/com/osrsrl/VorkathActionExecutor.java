package com.osrsrl;

import net.runelite.api.*;
import net.runelite.api.coords.LocalPoint;
import net.runelite.api.coords.WorldPoint;
import net.runelite.api.widgets.Widget;
import net.runelite.api.widgets.WidgetInfo;
import lombok.extern.slf4j.Slf4j;

/**
 * Executes Vorkath actions from RL model using RuneLite menu actions
 * 
 * Action mapping (matches training environment):
 * 0:  WAIT
 * 1:  ATTACK_RANGED
 * 2:  ATTACK_SPEC
 * 3:  PRAY_MAGE
 * 4:  PRAY_RANGE
 * 5:  PRAY_OFF
 * 6:  TOGGLE_RIGOUR
 * 7:  EAT_FOOD
 * 8:  DRINK_PRAYER
 * 9:  DRINK_ANTIFIRE
 * 10: DRINK_ANTIVENOM
 * 11: MOVE_NORTH
 * 12: MOVE_SOUTH
 * 13: MOVE_EAST
 * 14: MOVE_WEST
 * 15: WALK_AROUND (acid phase movement)
 * 16: CAST_CRUMBLE_UNDEAD
 */
@Slf4j
public class VorkathActionExecutor {
    
    private static final int VORKATH_ID = 8061;
    private static final int ZOMBIFIED_SPAWN_ID = 8063;
    
    // Spell widget IDs
    private static final int CRUMBLE_UNDEAD_WIDGET_ID = 14286870; // Spellbook Crumble Undead
    
    private final Client client;
    private String lastAction = "NONE";
    
    // Track acid walk direction
    private boolean walkingEast = true;
    
    public VorkathActionExecutor(Client client) {
        this.client = client;
    }
    
    public void execute(int action) {
        try {
            switch (action) {
                case 0:
                    lastAction = "WAIT";
                    break;
                case 1:
                    attackVorkath();
                    lastAction = "ATTACK_RANGED";
                    break;
                case 2:
                    toggleSpecialAttack();
                    lastAction = "ATTACK_SPEC";
                    break;
                case 3:
                    setPrayer(Prayer.PROTECT_FROM_MAGIC, true);
                    lastAction = "PRAY_MAGE";
                    break;
                case 4:
                    setPrayer(Prayer.PROTECT_FROM_MISSILES, true);
                    lastAction = "PRAY_RANGE";
                    break;
                case 5:
                    setPrayer(Prayer.PROTECT_FROM_MAGIC, false);
                    setPrayer(Prayer.PROTECT_FROM_MISSILES, false);
                    lastAction = "PRAY_OFF";
                    break;
                case 6:
                    togglePrayer(Prayer.RIGOUR);
                    lastAction = "TOGGLE_RIGOUR";
                    break;
                case 7:
                    eatFood();
                    lastAction = "EAT_FOOD";
                    break;
                case 8:
                    drinkPotion(ItemID.PRAYER_POTION4, ItemID.PRAYER_POTION3, 
                                ItemID.PRAYER_POTION2, ItemID.PRAYER_POTION1,
                                ItemID.SUPER_RESTORE4, ItemID.SUPER_RESTORE3,
                                ItemID.SUPER_RESTORE2, ItemID.SUPER_RESTORE1);
                    lastAction = "DRINK_PRAYER";
                    break;
                case 9:
                    drinkPotion(ItemID.EXTENDED_SUPER_ANTIFIRE4, ItemID.EXTENDED_SUPER_ANTIFIRE3,
                                ItemID.EXTENDED_SUPER_ANTIFIRE2, ItemID.EXTENDED_SUPER_ANTIFIRE1,
                                ItemID.SUPER_ANTIFIRE_POTION4, ItemID.SUPER_ANTIFIRE_POTION3,
                                ItemID.SUPER_ANTIFIRE_POTION2, ItemID.SUPER_ANTIFIRE_POTION1);
                    lastAction = "DRINK_ANTIFIRE";
                    break;
                case 10:
                    drinkPotion(ItemID.ANTIVENOM4, ItemID.ANTIVENOM3, 
                                ItemID.ANTIVENOM2, ItemID.ANTIVENOM1);
                    lastAction = "DRINK_ANTIVENOM";
                    break;
                case 11:
                    walkDirection(0, 1); // North
                    lastAction = "MOVE_NORTH";
                    break;
                case 12:
                    walkDirection(0, -1); // South
                    lastAction = "MOVE_SOUTH";
                    break;
                case 13:
                    walkDirection(1, 0); // East
                    lastAction = "MOVE_EAST";
                    break;
                case 14:
                    walkDirection(-1, 0); // West
                    lastAction = "MOVE_WEST";
                    break;
                case 15:
                    walkAround();
                    lastAction = "WALK_AROUND";
                    break;
                case 16:
                    castCrumbleUndead();
                    lastAction = "CAST_CRUMBLE_UNDEAD";
                    break;
                default:
                    lastAction = "UNKNOWN_" + action;
                    log.warn("[OSRS-RL] Unknown action: {}", action);
            }
        } catch (Exception e) {
            log.error("[OSRS-RL] Error executing action {}: {}", action, e.getMessage());
        }
        
        log.info("[OSRS-RL] Executed: {}", lastAction);
    }
    
    /**
     * Attack Vorkath NPC
     */
    private void attackVorkath() {
        NPC vorkath = findNpc(VORKATH_ID);
        if (vorkath == null) {
            // Try by name
            for (NPC npc : client.getNpcs()) {
                if (npc.getName() != null && npc.getName().contains("Vorkath")) {
                    vorkath = npc;
                    break;
                }
            }
        }
        
        if (vorkath != null) {
            // Create menu entry for attacking
            client.createMenuEntry(-1)
                .setOption("Attack")
                .setTarget("<col=ffff00>" + vorkath.getName() + "<col=ff00>  (level-" + vorkath.getCombatLevel() + ")")
                .setType(MenuAction.NPC_SECOND_OPTION)
                .setIdentifier(vorkath.getIndex())
                .setParam0(0)
                .setParam1(0)
                .onClick(e -> {
                    log.info("[OSRS-RL] Attacking Vorkath");
                });
            
            // Alternative: direct invoke (may work on some versions)
            // client.invokeMenuAction("Attack", vorkath.getName(), vorkath.getIndex(), 
            //     MenuAction.NPC_SECOND_OPTION.getId(), 0, 0);
        } else {
            log.warn("[OSRS-RL] Vorkath not found");
        }
    }
    
    /**
     * Toggle special attack
     */
    private void toggleSpecialAttack() {
        int specEnergy = client.getVarpValue(VarPlayer.SPECIAL_ATTACK_PERCENT);
        if (specEnergy < 500) { // Need 50%
            log.info("[OSRS-RL] Not enough spec energy: {}%", specEnergy / 10);
            return;
        }
        
        Widget specOrb = client.getWidget(WidgetInfo.MINIMAP_SPEC_ORB);
        if (specOrb != null) {
            client.createMenuEntry(-1)
                .setOption("Use")
                .setTarget("<col=ff9040>Special Attack</col>")
                .setType(MenuAction.CC_OP)
                .setIdentifier(1)
                .setParam0(-1)
                .setParam1(specOrb.getId())
                .onClick(e -> {
                    log.info("[OSRS-RL] Special attack activated");
                });
        }
    }
    
    /**
     * Set a prayer on or off
     * Uses raw widget IDs since WidgetInfo constants change between versions
     */
    private void setPrayer(Prayer prayer, boolean activate) {
        if (client.isPrayerActive(prayer) == activate) {
            return; // Already in desired state
        }
        
        // Check prayer points
        if (activate && client.getBoostedSkillLevel(Skill.PRAYER) <= 0) {
            log.warn("[OSRS-RL] No prayer points");
            return;
        }
        
        // Prayer widget IDs (group 541)
        // These are: groupId << 16 | childId
        int widgetId;
        switch (prayer) {
            case PROTECT_FROM_MAGIC:
                widgetId = 35454997; // 541 << 16 | 21
                break;
            case PROTECT_FROM_MISSILES:
                widgetId = 35454996; // 541 << 16 | 20
                break;
            case RIGOUR:
                widgetId = 35455003; // 541 << 16 | 27
                break;
            case AUGURY:
                widgetId = 35455005; // 541 << 16 | 29
                break;
            default:
                log.warn("[OSRS-RL] Unknown prayer: {}", prayer);
                return;
        }
        
        Widget prayerWidget = client.getWidget(widgetId);
        if (prayerWidget != null) {
            String option = activate ? "Activate" : "Deactivate";
            
            client.createMenuEntry(-1)
                .setOption(option)
                .setTarget("<col=ff9040>" + formatPrayerName(prayer) + "</col>")
                .setType(MenuAction.CC_OP)
                .setIdentifier(1)
                .setParam0(-1)
                .setParam1(prayerWidget.getId())
                .onClick(e -> {
                    log.info("[OSRS-RL] {} {}", option, prayer.name());
                });
        } else {
            log.warn("[OSRS-RL] Prayer widget not found for {}", prayer);
        }
    }
    
    /**
     * Toggle a prayer
     */
    private void togglePrayer(Prayer prayer) {
        setPrayer(prayer, !client.isPrayerActive(prayer));
    }
    
    /**
     * Eat food from inventory
     */
    private void eatFood() {
        Widget inventory = client.getWidget(WidgetInfo.INVENTORY);
        if (inventory == null) return;
        
        Widget[] items = inventory.getDynamicChildren();
        if (items == null) return;
        
        for (int i = 0; i < items.length; i++) {
            Widget item = items[i];
            int itemId = item.getItemId();
            
            if (isFood(itemId)) {
                final int slot = i;
                final int id = itemId;
                
                client.createMenuEntry(-1)
                    .setOption("Eat")
                    .setTarget("<col=ff9040>" + client.getItemDefinition(itemId).getName() + "</col>")
                    .setType(MenuAction.CC_OP)
                    .setIdentifier(2) // "Eat" is usually option 2
                    .setParam0(slot)
                    .setParam1(WidgetInfo.INVENTORY.getId())
                    .onClick(e -> {
                        log.info("[OSRS-RL] Eating item {} from slot {}", id, slot);
                    });
                return;
            }
        }
        
        log.warn("[OSRS-RL] No food found");
    }
    
    /**
     * Drink a potion by item IDs
     */
    private void drinkPotion(int... itemIds) {
        Widget inventory = client.getWidget(WidgetInfo.INVENTORY);
        if (inventory == null) return;
        
        Widget[] items = inventory.getDynamicChildren();
        if (items == null) return;
        
        for (int i = 0; i < items.length; i++) {
            Widget item = items[i];
            int itemId = item.getItemId();
            
            for (int targetId : itemIds) {
                if (itemId == targetId) {
                    final int slot = i;
                    final int id = itemId;
                    
                    client.createMenuEntry(-1)
                        .setOption("Drink")
                        .setTarget("<col=ff9040>" + client.getItemDefinition(itemId).getName() + "</col>")
                        .setType(MenuAction.CC_OP)
                        .setIdentifier(2) // "Drink" is usually option 2
                        .setParam0(slot)
                        .setParam1(WidgetInfo.INVENTORY.getId())
                        .onClick(e -> {
                            log.info("[OSRS-RL] Drinking potion {} from slot {}", id, slot);
                        });
                    return;
                }
            }
        }
        
        log.warn("[OSRS-RL] Potion not found");
    }
    
    /**
     * Walk in a direction
     */
    private void walkDirection(int dx, int dy) {
        Player player = client.getLocalPlayer();
        if (player == null) return;
        
        WorldPoint current = player.getWorldLocation();
        WorldPoint target = new WorldPoint(
            current.getX() + dx,
            current.getY() + dy,
            current.getPlane()
        );
        
        walkTo(target);
    }
    
    /**
     * Walk around (acid phase pattern)
     */
    private void walkAround() {
        Player player = client.getLocalPlayer();
        if (player == null) return;
        
        WorldPoint current = player.getWorldLocation();
        
        // Simple east-west pattern
        int dx = walkingEast ? 2 : -2;
        WorldPoint target = new WorldPoint(
            current.getX() + dx,
            current.getY(),
            current.getPlane()
        );
        
        // Toggle direction for next call
        walkingEast = !walkingEast;
        
        walkTo(target);
    }
    
    /**
     * Walk to a specific world point
     */
    private void walkTo(WorldPoint worldPoint) {
        LocalPoint localPoint = LocalPoint.fromWorld(client, worldPoint);
        if (localPoint == null) {
            log.warn("[OSRS-RL] Cannot walk to {} - not in scene", worldPoint);
            return;
        }
        
        // Calculate scene coordinates
        int sceneX = localPoint.getSceneX();
        int sceneY = localPoint.getSceneY();
        
        client.createMenuEntry(-1)
            .setOption("Walk here")
            .setTarget("")
            .setType(MenuAction.WALK)
            .setIdentifier(0)
            .setParam0(sceneX)
            .setParam1(sceneY)
            .onClick(e -> {
                log.info("[OSRS-RL] Walking to {}", worldPoint);
            });
    }
    
    /**
     * Cast Crumble Undead on zombified spawn
     */
    private void castCrumbleUndead() {
        NPC spawn = findNpc(ZOMBIFIED_SPAWN_ID);
        if (spawn == null) {
            // Try by name
            for (NPC npc : client.getNpcs()) {
                if (npc.getName() != null && npc.getName().contains("Zombified")) {
                    spawn = npc;
                    break;
                }
            }
        }
        
        if (spawn == null) {
            log.warn("[OSRS-RL] No zombified spawn found");
            return;
        }
        
        // Note: Crumble Undead will be cast directly on the spawn
        
        // Crumble Undead is in the standard spellbook
        // Widget ID may vary - this targets the spell directly
        final NPC targetSpawn = spawn;
        
        client.createMenuEntry(-1)
            .setOption("Cast")
            .setTarget("<col=00ff00>Crumble Undead</col> -> <col=ffff00>" + spawn.getName())
            .setType(MenuAction.WIDGET_TARGET_ON_NPC)
            .setIdentifier(spawn.getIndex())
            .setParam0(0)
            .setParam1(0)
            .onClick(e -> {
                log.info("[OSRS-RL] Casting Crumble Undead on spawn");
            });
    }
    
    /**
     * Find NPC by ID
     */
    private NPC findNpc(int id) {
        for (NPC npc : client.getNpcs()) {
            if (npc.getId() == id) {
                return npc;
            }
        }
        return null;
    }
    
    /**
     * Check if item is food
     */
    private boolean isFood(int itemId) {
        // Common food IDs
        switch (itemId) {
            case ItemID.SHARK:
            case ItemID.MANTA_RAY:
            case ItemID.ANGLERFISH:
            case ItemID.DARK_CRAB:
            case ItemID.COOKED_KARAMBWAN:
            case ItemID.TUNA_POTATO:
            case ItemID.SARADOMIN_BREW4:
            case ItemID.SARADOMIN_BREW3:
            case ItemID.SARADOMIN_BREW2:
            case ItemID.SARADOMIN_BREW1:
            case ItemID.MONKFISH:
            case ItemID.BASS:
            case ItemID.SWORDFISH:
            case ItemID.LOBSTER:
                return true;
            default:
                return false;
        }
    }
    
    /**
     * Format prayer name for display
     */
    private String formatPrayerName(Prayer prayer) {
        String name = prayer.name().replace("_", " ").toLowerCase();
        return name.substring(0, 1).toUpperCase() + name.substring(1);
    }
    
    public String getLastAction() {
        return lastAction;
    }
}
