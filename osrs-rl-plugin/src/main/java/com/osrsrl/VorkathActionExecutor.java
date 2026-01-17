package com.osrsrl;

import net.runelite.api.*;
import net.runelite.api.coords.WorldPoint;
import net.runelite.api.widgets.Widget;
import net.runelite.api.widgets.WidgetInfo;
import lombok.extern.slf4j.Slf4j;

/**
 * Executes Vorkath actions from RL model
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
    
    private final Client client;
    private String lastAction = "NONE";
    
    public VorkathActionExecutor(Client client) {
        this.client = client;
    }
    
    public void execute(int action) {
        switch (action) {
            case 0:
                lastAction = "WAIT";
                // Do nothing
                break;
            case 1:
                attackVorkath();
                lastAction = "ATTACK_RANGED";
                break;
            case 2:
                specialAttack();
                lastAction = "ATTACK_SPEC";
                break;
            case 3:
                activatePrayer(Prayer.PROTECT_FROM_MAGIC);
                lastAction = "PRAY_MAGE";
                break;
            case 4:
                activatePrayer(Prayer.PROTECT_FROM_MISSILES);
                lastAction = "PRAY_RANGE";
                break;
            case 5:
                deactivateProtectionPrayers();
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
                drinkPrayerPotion();
                lastAction = "DRINK_PRAYER";
                break;
            case 9:
                drinkAntifire();
                lastAction = "DRINK_ANTIFIRE";
                break;
            case 10:
                drinkAntivenom();
                lastAction = "DRINK_ANTIVENOM";
                break;
            case 11:
                move(0, 1); // North
                lastAction = "MOVE_NORTH";
                break;
            case 12:
                move(0, -1); // South
                lastAction = "MOVE_SOUTH";
                break;
            case 13:
                move(1, 0); // East
                lastAction = "MOVE_EAST";
                break;
            case 14:
                move(-1, 0); // West
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
        
        log.info("[OSRS-RL] Executed: {}", lastAction);
    }
    
    private void attackVorkath() {
        NPC vorkath = findNpc(VORKATH_ID);
        if (vorkath != null) {
            log.info("[OSRS-RL] Attacking Vorkath");
            // In real implementation, this would use menu actions
            // client.invokeMenuAction("Attack", "Vorkath", vorkath.getIndex(), MenuAction.NPC_SECOND_OPTION.getId(), 0, 0);
        }
    }
    
    private void specialAttack() {
        int specEnergy = client.getVarpValue(VarPlayer.SPECIAL_ATTACK_PERCENT);
        if (specEnergy >= 500) { // 50%
            Widget specBar = client.getWidget(WidgetInfo.MINIMAP_SPEC_ORB);
            if (specBar != null) {
                log.info("[OSRS-RL] Using special attack");
                // Click spec bar
            }
        }
    }
    
    private void activatePrayer(Prayer prayer) {
        if (!client.isPrayerActive(prayer)) {
            log.info("[OSRS-RL] Activating {}", prayer.name());
            // Deactivate conflicting prayers first
            if (prayer == Prayer.PROTECT_FROM_MAGIC) {
                if (client.isPrayerActive(Prayer.PROTECT_FROM_MISSILES)) {
                    togglePrayer(Prayer.PROTECT_FROM_MISSILES);
                }
            } else if (prayer == Prayer.PROTECT_FROM_MISSILES) {
                if (client.isPrayerActive(Prayer.PROTECT_FROM_MAGIC)) {
                    togglePrayer(Prayer.PROTECT_FROM_MAGIC);
                }
            }
            togglePrayer(prayer);
        }
    }
    
    private void deactivateProtectionPrayers() {
        if (client.isPrayerActive(Prayer.PROTECT_FROM_MAGIC)) {
            togglePrayer(Prayer.PROTECT_FROM_MAGIC);
        }
        if (client.isPrayerActive(Prayer.PROTECT_FROM_MISSILES)) {
            togglePrayer(Prayer.PROTECT_FROM_MISSILES);
        }
        log.info("[OSRS-RL] Deactivated protection prayers");
    }
    
    private void togglePrayer(Prayer prayer) {
        Widget prayerWidget = client.getWidget(prayer.getWidgetInfo());
        if (prayerWidget != null) {
            log.info("[OSRS-RL] Toggling {}", prayer.name());
            // In real implementation:
            // client.invokeMenuAction(client.isPrayerActive(prayer) ? "Deactivate" : "Activate", 
            //     prayer.name(), 1, MenuAction.CC_OP.getId(), -1, prayerWidget.getId());
        }
    }
    
    private void eatFood() {
        Widget inventory = client.getWidget(WidgetInfo.INVENTORY);
        if (inventory == null) return;
        
        for (Widget item : inventory.getDynamicChildren()) {
            int id = item.getItemId();
            if (isFood(id)) {
                log.info("[OSRS-RL] Eating food (ID: {})", id);
                // client.invokeMenuAction("Eat", item.getName(), item.getItemId(), 
                //     MenuAction.CC_OP.getId(), item.getIndex(), WidgetInfo.INVENTORY.getId());
                return;
            }
        }
    }
    
    private void drinkPrayerPotion() {
        drinkPotion(ItemID.PRAYER_POTION4, ItemID.PRAYER_POTION3, ItemID.PRAYER_POTION2, ItemID.PRAYER_POTION1);
    }
    
    private void drinkAntifire() {
        drinkPotion(ItemID.EXTENDED_SUPER_ANTIFIRE4, ItemID.EXTENDED_SUPER_ANTIFIRE3, 
                    ItemID.EXTENDED_SUPER_ANTIFIRE2, ItemID.EXTENDED_SUPER_ANTIFIRE1);
    }
    
    private void drinkAntivenom() {
        drinkPotion(ItemID.ANTIVENOM4, ItemID.ANTIVENOM3, ItemID.ANTIVENOM2, ItemID.ANTIVENOM1);
    }
    
    private void drinkPotion(int... itemIds) {
        Widget inventory = client.getWidget(WidgetInfo.INVENTORY);
        if (inventory == null) return;
        
        for (Widget item : inventory.getDynamicChildren()) {
            for (int id : itemIds) {
                if (item.getItemId() == id) {
                    log.info("[OSRS-RL] Drinking potion (ID: {})", id);
                    return;
                }
            }
        }
    }
    
    private void move(int dx, int dy) {
        Player player = client.getLocalPlayer();
        if (player == null) return;
        
        WorldPoint current = player.getWorldLocation();
        WorldPoint target = new WorldPoint(current.getX() + dx, current.getY() + dy, current.getPlane());
        
        log.info("[OSRS-RL] Moving to {}", target);
        // In real implementation, would click on scene tile
    }
    
    private void walkAround() {
        // Acid phase walking pattern
        Player player = client.getLocalPlayer();
        if (player == null) return;
        
        WorldPoint current = player.getWorldLocation();
        
        // Simple back-and-forth pattern
        int dx = (current.getX() % 2 == 0) ? 1 : -1;
        WorldPoint target = new WorldPoint(current.getX() + dx, current.getY(), current.getPlane());
        
        log.info("[OSRS-RL] Walking around (acid phase)");
    }
    
    private void castCrumbleUndead() {
        NPC spawn = findNpc(ZOMBIFIED_SPAWN_ID);
        if (spawn != null) {
            log.info("[OSRS-RL] Casting Crumble Undead on spawn");
            // Would open spellbook, click Crumble Undead, target spawn
        } else {
            log.warn("[OSRS-RL] No zombified spawn found");
        }
    }
    
    private NPC findNpc(int id) {
        for (NPC npc : client.getNpcs()) {
            if (npc.getId() == id) {
                return npc;
            }
        }
        return null;
    }
    
    private boolean isFood(int itemId) {
        return itemId == ItemID.SHARK || itemId == ItemID.MANTA_RAY || 
               itemId == ItemID.ANGLERFISH || itemId == ItemID.DARK_CRAB ||
               itemId == ItemID.COOKED_KARAMBWAN || itemId == ItemID.TUNA_POTATO ||
               itemId == ItemID.SARADOMIN_BREW4 || itemId == ItemID.SARADOMIN_BREW3;
    }
    
    public String getLastAction() {
        return lastAction;
    }
}
