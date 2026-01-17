package com.osrsrl;

import net.runelite.api.*;
import net.runelite.api.coords.WorldPoint;

import java.util.ArrayList;
import java.util.List;

/**
 * Extracts clean game state from the client
 */
public class StateExtractor {
    
    private final Client client;
    private int tickCounter = 0;
    
    public StateExtractor(Client client) {
        this.client = client;
    }
    
    public GameState extract() {
        GameState state = new GameState();
        state.tick = tickCounter++;
        
        Player local = client.getLocalPlayer();
        if (local == null) {
            return state;
        }
        
        // Player position
        WorldPoint wp = local.getWorldLocation();
        state.playerX = wp.getX();
        state.playerY = wp.getY();
        state.playerPlane = wp.getPlane();
        
        // Player stats
        state.playerHp = client.getBoostedSkillLevel(Skill.HITPOINTS);
        state.playerMaxHp = client.getRealSkillLevel(Skill.HITPOINTS);
        state.playerPrayer = client.getBoostedSkillLevel(Skill.PRAYER);
        state.playerMaxPrayer = client.getRealSkillLevel(Skill.PRAYER);
        state.playerEnergy = client.getEnergy() / 100;
        state.playerAnimation = local.getAnimation();
        state.playerIsMoving = local.getPoseAnimation() != local.getIdlePoseAnimation();
        
        // Combat state
        Actor interacting = local.getInteracting();
        state.playerInCombat = interacting != null;
        
        // Target
        if (interacting != null) {
            state.hasTarget = true;
            state.targetX = interacting.getWorldLocation().getX();
            state.targetY = interacting.getWorldLocation().getY();
            state.targetAnimation = interacting.getAnimation();
            
            if (interacting instanceof NPC) {
                NPC npc = (NPC) interacting;
                int ratio = npc.getHealthRatio();
                int scale = npc.getHealthScale();
                if (scale > 0) {
                    state.targetHp = ratio;
                    state.targetMaxHp = scale;
                }
            }
        }
        
        // Inventory
        ItemContainer inventory = client.getItemContainer(InventoryID.INVENTORY);
        if (inventory != null) {
            Item[] items = inventory.getItems();
            for (int i = 0; i < 28 && i < items.length; i++) {
                state.inventoryIds[i] = items[i].getId();
                state.inventoryQuantities[i] = items[i].getQuantity();
            }
        }
        
        // Equipment
        ItemContainer equipment = client.getItemContainer(InventoryID.EQUIPMENT);
        if (equipment != null) {
            Item[] items = equipment.getItems();
            for (int i = 0; i < 11 && i < items.length; i++) {
                state.equipmentIds[i] = items[i].getId();
            }
        }
        
        // Active prayers - use Varbits instead of deprecated method
        extractPrayers(state);
        
        // Skills
        for (Skill skill : Skill.values()) {
            int ordinal = skill.ordinal();
            if (ordinal < 23) {
                state.skillLevels[ordinal] = client.getRealSkillLevel(skill);
                state.skillXp[ordinal] = client.getSkillExperience(skill);
            }
        }
        
        // Nearby NPCs
        state.nearbyNpcs = getNearbyNpcs(local, 15);
        
        // Nearby Players
        state.nearbyPlayers = getNearbyPlayers(local, 15);
        
        // Nearby Objects
        state.nearbyObjects = getNearbyObjects(local, 10);
        
        return state;
    }
    
    private void extractPrayers(GameState state) {
        // Check common prayers via their varbits
        // This is a simplified version - just check if any prayer is active
        try {
            for (Prayer prayer : Prayer.values()) {
                int ordinal = prayer.ordinal();
                if (ordinal < 29) {
                    state.activePrayers[ordinal] = client.isPrayerActive(prayer);
                }
            }
        } catch (Exception e) {
            // Prayer API may vary, just skip if it fails
        }
    }
    
    private List<GameState.EntityInfo> getNearbyNpcs(Player local, int maxDistance) {
        List<GameState.EntityInfo> npcs = new ArrayList<>();
        WorldPoint playerLoc = local.getWorldLocation();
        
        for (NPC npc : client.getNpcs()) {
            if (npc == null) continue;
            
            WorldPoint npcLoc = npc.getWorldLocation();
            int distance = playerLoc.distanceTo(npcLoc);
            
            if (distance <= maxDistance) {
                GameState.EntityInfo info = new GameState.EntityInfo();
                info.id = npc.getId();
                info.name = npc.getName();
                info.x = npcLoc.getX();
                info.y = npcLoc.getY();
                info.distance = distance;
                info.animation = npc.getAnimation();
                
                int ratio = npc.getHealthRatio();
                int scale = npc.getHealthScale();
                if (scale > 0) {
                    info.hp = ratio;
                    info.maxHp = scale;
                }
                
                npcs.add(info);
                if (npcs.size() >= 10) break;
            }
        }
        
        return npcs;
    }
    
    private List<GameState.EntityInfo> getNearbyPlayers(Player local, int maxDistance) {
        List<GameState.EntityInfo> players = new ArrayList<>();
        WorldPoint playerLoc = local.getWorldLocation();
        
        for (Player player : client.getPlayers()) {
            if (player == null || player == local) continue;
            
            WorldPoint pLoc = player.getWorldLocation();
            int distance = playerLoc.distanceTo(pLoc);
            
            if (distance <= maxDistance) {
                GameState.EntityInfo info = new GameState.EntityInfo();
                info.name = player.getName();
                info.x = pLoc.getX();
                info.y = pLoc.getY();
                info.distance = distance;
                info.animation = player.getAnimation();
                
                players.add(info);
                if (players.size() >= 10) break;
            }
        }
        
        return players;
    }
    
    private List<GameState.ObjectInfo> getNearbyObjects(Player local, int maxDistance) {
        List<GameState.ObjectInfo> objects = new ArrayList<>();
        WorldPoint playerLoc = local.getWorldLocation();
        
        Scene scene = client.getScene();
        Tile[][][] tiles = scene.getTiles();
        int plane = client.getPlane();
        
        for (int x = 0; x < Constants.SCENE_SIZE; x++) {
            for (int y = 0; y < Constants.SCENE_SIZE; y++) {
                Tile tile = tiles[plane][x][y];
                if (tile == null) continue;
                
                WorldPoint tileLoc = tile.getWorldLocation();
                int distance = playerLoc.distanceTo(tileLoc);
                if (distance > maxDistance) continue;
                
                GameObject[] gameObjects = tile.getGameObjects();
                if (gameObjects == null) continue;
                
                for (GameObject go : gameObjects) {
                    if (go == null) continue;
                    
                    ObjectComposition comp = client.getObjectDefinition(go.getId());
                    if (comp == null) continue;
                    
                    String name = comp.getName();
                    if (name == null || name.equals("null") || name.isEmpty()) continue;
                    
                    GameState.ObjectInfo info = new GameState.ObjectInfo();
                    info.id = go.getId();
                    info.name = name;
                    info.x = tileLoc.getX();
                    info.y = tileLoc.getY();
                    info.distance = distance;
                    info.actions = comp.getActions();
                    
                    objects.add(info);
                    if (objects.size() >= 20) return objects;
                }
            }
        }
        
        return objects;
    }
}
