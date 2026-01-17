package com.osrsrl;

import net.runelite.api.Client;
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
    description = "Reinforcement Learning interface for OSRS",
    tags = {"rl", "ai", "reinforcement", "learning"}
)
public class OSRSRLPlugin extends Plugin {

    @Inject
    private Client client;

    private SocketServer server;
    private StateExtractor stateExtractor;
    private ActionExecutor actionExecutor;
    
    private volatile boolean actionPending = false;
    private volatile int[] pendingAction = null;

    @Override
    protected void startUp() {
        log.info("========================================");
        log.info("OSRS-RL Plugin Starting!");
        log.info("========================================");
        
        stateExtractor = new StateExtractor(client);
        actionExecutor = new ActionExecutor(client);
        
        server = new SocketServer(5555, this::handleAction);
        server.start();
        
        log.info("[OSRS-RL] Plugin started on port 5555");
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
        // Extract current state
        GameState state = stateExtractor.extract();
        
        // Execute pending action if any
        if (actionPending && pendingAction != null) {
            log.info("[OSRS-RL] GameTick: Executing pending action: {}", Arrays.toString(pendingAction));
            actionExecutor.execute(pendingAction);
            pendingAction = null;
            actionPending = false;
        }
        
        // Send state to Python
        server.sendState(state);
    }
    
    private void handleAction(int[] action) {
        log.info("[OSRS-RL] handleAction received: {}", Arrays.toString(action));
        this.pendingAction = action;
        this.actionPending = true;
    }
    
    public Client getClient() {
        return client;
    }
}
