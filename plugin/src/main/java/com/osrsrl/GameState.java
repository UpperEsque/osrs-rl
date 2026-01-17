package com.osrsrl;

import com.google.gson.Gson;
import java.util.List;

/**
 * Complete game state snapshot - sent to Python every tick
 */
public class GameState {
    private static final Gson GSON = new Gson();
    
    // Tick counter
    public int tick;
    
    // Player state
    public int playerX;
    public int playerY;
    public int playerPlane;
    public int playerHp;
    public int playerMaxHp;
    public int playerPrayer;
    public int playerMaxPrayer;
    public int playerEnergy;
    public int playerAnimation;
    public boolean playerIsMoving;
    public boolean playerInCombat;
    
    // Target (if any)
    public boolean hasTarget;
    public int targetHp;
    public int targetMaxHp;
    public int targetX;
    public int targetY;
    public int targetAnimation;
    
    // Inventory (28 slots: item_id, -1 if empty)
    public int[] inventoryIds = new int[28];
    public int[] inventoryQuantities = new int[28];
    
    // Equipment (11 slots)
    public int[] equipmentIds = new int[11];
    
    // Prayers (29 prayers, true if active)
    public boolean[] activePrayers = new boolean[29];
    
    // Skills (23 skills)
    public int[] skillLevels = new int[23];
    public int[] skillXp = new int[23];
    
    // Nearby NPCs (up to 10)
    public List<EntityInfo> nearbyNpcs;
    
    // Nearby Players (up to 10)  
    public List<EntityInfo> nearbyPlayers;
    
    // Nearby Objects (up to 20)
    public List<ObjectInfo> nearbyObjects;
    
    public String toJson() {
        return GSON.toJson(this);
    }
    
    public static class EntityInfo {
        public int id;
        public String name;
        public int x;
        public int y;
        public int hp;
        public int maxHp;
        public int animation;
        public int distance;
    }
    
    public static class ObjectInfo {
        public int id;
        public String name;
        public int x;
        public int y;
        public int distance;
        public String[] actions;
    }
}
