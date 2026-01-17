package com.osrsrl;

import net.runelite.api.*;
import net.runelite.api.coords.LocalPoint;
import net.runelite.api.coords.WorldPoint;
import net.runelite.api.widgets.Widget;
import lombok.extern.slf4j.Slf4j;

import java.awt.Canvas;
import java.awt.event.MouseEvent;
import java.util.concurrent.ThreadLocalRandom;

@Slf4j
public class ActionExecutor {

    public static final int NOOP = 0;
    public static final int WALK_TO = 1;
    public static final int CLICK_INVENTORY = 2;
    public static final int ATTACK_NPC = 3;
    public static final int INTERACT_OBJECT = 4;
    public static final int TOGGLE_PRAYER = 5;
    public static final int TOGGLE_RUN = 6;
    public static final int SPECIAL_ATTACK = 7;

    private final Client client;

    private static final int MINIMAP_CENTER_X = 643;
    private static final int MINIMAP_CENTER_Y = 83;
    private static final int PIXELS_PER_TILE = 4;

    private static final int INV_START_X = 563;
    private static final int INV_START_Y = 213;
    private static final int SLOT_WIDTH = 42;
    private static final int SLOT_HEIGHT = 36;

    private static final int PRAYER_GROUP_ID = 541;
    private static final int PROTECT_MAGIC_CHILD = 21;
    private static final int PROTECT_RANGE_CHILD = 20;

    private static final int INVENTORY_GROUP_ID = 149;
    private static final int INVENTORY_CHILD_ID = 0;

    public ActionExecutor(Client client) {
        this.client = client;
        log.info("[OSRS-RL] ActionExecutor created (direct canvas mode)");
    }

    private void sendClick(int canvasX, int canvasY) {
        Canvas canvas = client.getCanvas();
        if (canvas == null) {
            log.warn("[OSRS-RL] Canvas is null");
            return;
        }

        int x = canvasX + ThreadLocalRandom.current().nextInt(-2, 3);
        int y = canvasY + ThreadLocalRandom.current().nextInt(-2, 3);

        long when = System.currentTimeMillis();

        MouseEvent pressed = new MouseEvent(canvas, MouseEvent.MOUSE_PRESSED,
            when, MouseEvent.BUTTON1_DOWN_MASK, x, y, 1, false, MouseEvent.BUTTON1);

        MouseEvent released = new MouseEvent(canvas, MouseEvent.MOUSE_RELEASED,
            when + 50, 0, x, y, 1, false, MouseEvent.BUTTON1);

        MouseEvent clicked = new MouseEvent(canvas, MouseEvent.MOUSE_CLICKED,
            when + 50, 0, x, y, 1, false, MouseEvent.BUTTON1);

        canvas.dispatchEvent(pressed);
        canvas.dispatchEvent(released);
        canvas.dispatchEvent(clicked);

        log.info("[OSRS-RL] Sent click to canvas at ({}, {})", x, y);
    }

    public void execute(int[] action) {
        if (action == null || action.length == 0) return;
        int actionType = action[0];
        log.info("[OSRS-RL] Execute action: {}", actionType);

        switch (actionType) {
            case NOOP:
                break;
            case WALK_TO:
                if (action.length >= 3) walkToTile(action[1], action[2]);
                break;
            case CLICK_INVENTORY:
                if (action.length >= 2) clickInventorySlot(action[1]);
                break;
            case ATTACK_NPC:
                if (action.length >= 2) attackNpc(action[1]);
                break;
            case INTERACT_OBJECT:
                if (action.length >= 2) interactObject(action[1]);
                break;
            case TOGGLE_PRAYER:
                if (action.length >= 2) togglePrayer(action[1]);
                break;
            case TOGGLE_RUN:
                toggleRun();
                break;
            case SPECIAL_ATTACK:
                specialAttack();
                break;
            default:
                log.warn("[OSRS-RL] Unknown action: {}", actionType);
        }
    }

    private void walkToTile(int worldX, int worldY) {
        log.info("[OSRS-RL] walkToTile({}, {})", worldX, worldY);
        Player local = client.getLocalPlayer();
        if (local == null) return;

        WorldPoint target = new WorldPoint(worldX, worldY, client.getPlane());
        LocalPoint localPoint = LocalPoint.fromWorld(client, target);

        if (localPoint != null) {
            net.runelite.api.Point canvasPoint = Perspective.localToCanvas(client, localPoint, client.getPlane());
            if (canvasPoint != null && canvasPoint.getX() > 5 && canvasPoint.getY() > 5 &&
                canvasPoint.getX() < client.getCanvasWidth() - 5 && canvasPoint.getY() < client.getCanvasHeight() - 5) {
                log.info("[OSRS-RL] Click game world ({}, {})", canvasPoint.getX(), canvasPoint.getY());
                sendClick(canvasPoint.getX(), canvasPoint.getY());
                return;
            }
        }

        clickMinimap(worldX, worldY);
    }

    private void clickMinimap(int worldX, int worldY) {
        Player local = client.getLocalPlayer();
        if (local == null) return;

        WorldPoint playerLoc = local.getWorldLocation();
        int dx = worldX - playerLoc.getX();
        int dy = worldY - playerLoc.getY();

        double distance = Math.sqrt(dx * dx + dy * dy);
        if (distance > 15) {
            double scale = 15 / distance;
            dx = (int) Math.round(dx * scale);
            dy = (int) Math.round(dy * scale);
        }

        int clickX = MINIMAP_CENTER_X + (dx * PIXELS_PER_TILE);
        int clickY = MINIMAP_CENTER_Y - (dy * PIXELS_PER_TILE);

        log.info("[OSRS-RL] Minimap click at ({}, {})", clickX, clickY);
        sendClick(clickX, clickY);
    }

    private void clickInventorySlot(int slot) {
        if (slot < 0 || slot > 27) {
            log.warn("[OSRS-RL] Invalid inventory slot: {}", slot);
            return;
        }

        Widget inventory = client.getWidget(INVENTORY_GROUP_ID, INVENTORY_CHILD_ID);
        if (inventory != null) {
            Widget[] children = inventory.getDynamicChildren();
            if (children != null && slot < children.length && children[slot] != null) {
                Widget item = children[slot];
                int x = item.getCanvasLocation().getX() + item.getWidth() / 2;
                int y = item.getCanvasLocation().getY() + item.getHeight() / 2;
                log.info("[OSRS-RL] Click inventory slot {} via widget at ({}, {})", slot, x, y);
                sendClick(x, y);
                return;
            }
        }

        int col = slot % 4;
        int row = slot / 4;
        int x = INV_START_X + (col * SLOT_WIDTH) + (SLOT_WIDTH / 2);
        int y = INV_START_Y + (row * SLOT_HEIGHT) + (SLOT_HEIGHT / 2);

        log.info("[OSRS-RL] Click inventory slot {} at fixed ({}, {})", slot, x, y);
        sendClick(x, y);
    }

    private void attackNpc(int npcIndex) {
        log.info("[OSRS-RL] attackNpc({})", npcIndex);

        for (NPC npc : client.getNpcs()) {
            if (npc != null && npc.getIndex() == npcIndex) {
                LocalPoint lp = npc.getLocalLocation();
                if (lp == null) continue;

                net.runelite.api.Point canvasPoint = Perspective.localToCanvas(client, lp, client.getPlane());
                if (canvasPoint != null) {
                    log.info("[OSRS-RL] Click NPC {} at ({}, {})", npc.getName(), canvasPoint.getX(), canvasPoint.getY());
                    sendClick(canvasPoint.getX(), canvasPoint.getY());
                    return;
                }
            }
        }
        log.warn("[OSRS-RL] NPC not found: {}", npcIndex);
    }

    private void interactObject(int objectId) {
        log.info("[OSRS-RL] interactObject({})", objectId);

        try {
            Tile[][][] tiles = client.getScene().getTiles();
            int plane = client.getPlane();

            for (int x = 0; x < 104; x++) {
                for (int y = 0; y < 104; y++) {
                    Tile tile = tiles[plane][x][y];
                    if (tile == null) continue;

                    for (GameObject obj : tile.getGameObjects()) {
                        if (obj != null && obj.getId() == objectId) {
                            LocalPoint lp = obj.getLocalLocation();
                            net.runelite.api.Point canvasPoint = Perspective.localToCanvas(client, lp, client.getPlane());
                            if (canvasPoint != null) {
                                log.info("[OSRS-RL] Click object {} at ({}, {})", objectId, canvasPoint.getX(), canvasPoint.getY());
                                sendClick(canvasPoint.getX(), canvasPoint.getY());
                                return;
                            }
                        }
                    }
                }
            }
        } catch (Exception e) {
            log.error("[OSRS-RL] Object interaction error", e);
        }
    }

    private void togglePrayer(int prayerId) {
        log.info("[OSRS-RL] togglePrayer({})", prayerId);

        int childId;
        switch (prayerId) {
            case 0:
                childId = PROTECT_MAGIC_CHILD;
                break;
            case 1:
                childId = PROTECT_RANGE_CHILD;
                break;
            default:
                log.warn("[OSRS-RL] Unknown prayer ID: {}", prayerId);
                return;
        }

        Widget prayerWidget = client.getWidget(PRAYER_GROUP_ID, childId);
        if (prayerWidget != null) {
            int x = prayerWidget.getCanvasLocation().getX() + prayerWidget.getWidth() / 2;
            int y = prayerWidget.getCanvasLocation().getY() + prayerWidget.getHeight() / 2;
            log.info("[OSRS-RL] Toggle prayer {} at ({}, {})", prayerId, x, y);
            sendClick(x, y);
        } else {
            log.warn("[OSRS-RL] Prayer widget not found for ID {}", prayerId);
        }
    }

    private void toggleRun() {
        log.info("[OSRS-RL] toggleRun");
        sendClick(563, 131);
    }

    private void specialAttack() {
        log.info("[OSRS-RL] specialAttack");
        sendClick(596, 148);
    }
}
