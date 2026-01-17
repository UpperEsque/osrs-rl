package com.osrsrl;

import net.runelite.api.Client;
import net.runelite.api.GameState;
import net.runelite.api.events.GameTick;
import net.runelite.client.eventbus.Subscribe;
import net.runelite.client.plugins.Plugin;
import net.runelite.client.plugins.PluginDescriptor;
import lombok.extern.slf4j.Slf4j;

import javax.inject.Inject;
import java.util.Arrays;

@Slf4j
@PluginDescriptor(
    name = "OSRS RL",
    description = "Reinforcement Learning interface for OSRS PvE bosses",
    tags = {"rl", "ai", "reinforcement", "learning", "vorkath", "pve"}
)
public class OSRSRLPlugin extends Plugin {

    @Inject
    private Client client;

    private SocketServer server;
    private VorkathStateExtractor stateExtractor;
    private VorkathActionExecutor actionExecutor;
    
    private volatile boolean actionPending = false;
    private volatile int[] pendingAction = null;
    
    private int tickCount = 0;

    @Override
    protected void startUp() {
        log.info("========================================");
        log.info("OSRS-RL Plugin Starting!");
        log.info("========================================");
        
        stateExtractor = new VorkathStateExtractor(client);
        actionExecutor = new VorkathActionExecutor(client);
        
        server = new SocketServer(5050, this::handleAction);
        server.start();
        
        log.info("[OSRS-RL] Plugin started on port 5050");
        System.out.println("[OSRS-RL] *** PLUGIN STARTED ON PORT 5050 ***");
    }

    @Override
    protected void shutDown() {
        log.info("[OSRS-RL] Plugin shutting down");
        if (server != null) {
            server.stop();
        }
    }

    @Subscribe
    public void onGameTick(GameTick event) {
        if (client.getGameState() != GameState.LOGGED_IN) {
            return;
        }
        
        tickCount++;
        
        // Extract current state
        float[] state = stateExtractor.extract();
        
        // Execute pending action if any
        if (actionPending && pendingAction != null) {
            log.debug("[OSRS-RL] Tick {}: Executing action {}", tickCount, Arrays.toString(pendingAction));
            actionExecutor.execute(pendingAction[0]);
            pendingAction = null;
            actionPending = false;
        }
        
        // Send state to Python
        server.sendState(state);
    }
    
    private void handleAction(int[] action) {
        log.debug("[OSRS-RL] Received action: {}", Arrays.toString(action));
        this.pendingAction = action;
        this.actionPending = true;
    }
    
    public Client getClient() {
        return client;
    }
}
