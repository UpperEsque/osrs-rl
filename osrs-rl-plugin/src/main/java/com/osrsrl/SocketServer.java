package com.osrsrl;

import com.google.gson.Gson;
import lombok.extern.slf4j.Slf4j;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

/**
 * Socket server for communication with Python RL model
 * 
 * Protocol:
 * - Server sends state as JSON: {"observation": [float array]}
 * - Python responds with action: {"action": [int]}
 */
@Slf4j
public class SocketServer {
    
    private final int port;
    private final Consumer<int[]> actionCallback;
    private final Gson gson;
    
    private ServerSocket serverSocket;
    private Socket clientSocket;
    private PrintWriter out;
    private BufferedReader in;
    
    private Thread serverThread;
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final AtomicBoolean clientConnected = new AtomicBoolean(false);
    
    public SocketServer(int port, Consumer<int[]> actionCallback) {
        this.port = port;
        this.actionCallback = actionCallback;
        this.gson = new Gson();
    }
    
    public void start() {
        if (running.get()) return;
        
        running.set(true);
        serverThread = new Thread(this::runServer, "OSRS-RL-Server");
        serverThread.setDaemon(true);
        serverThread.start();
    }
    
    private void runServer() {
        try {
            serverSocket = new ServerSocket(port);
            log.info("[OSRS-RL] Server listening on port {}", port);
            
            while (running.get()) {
                try {
                    log.info("[OSRS-RL] Waiting for client connection...");
                    clientSocket = serverSocket.accept();
                    clientConnected.set(true);
                    
                    log.info("[OSRS-RL] Client connected: {}", clientSocket.getRemoteSocketAddress());
                    
                    out = new PrintWriter(clientSocket.getOutputStream(), true);
                    in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
                    
                    // Read actions from client
                    String line;
                    while (running.get() && (line = in.readLine()) != null) {
                        try {
                            ActionMessage action = gson.fromJson(line, ActionMessage.class);
                            if (action != null && action.action != null) {
                                actionCallback.accept(action.action);
                            }
                        } catch (Exception e) {
                            log.error("[OSRS-RL] Error parsing action: {}", e.getMessage());
                        }
                    }
                    
                } catch (IOException e) {
                    if (running.get()) {
                        log.warn("[OSRS-RL] Client disconnected: {}", e.getMessage());
                    }
                } finally {
                    clientConnected.set(false);
                    closeClient();
                }
            }
            
        } catch (IOException e) {
            log.error("[OSRS-RL] Server error: {}", e.getMessage());
        } finally {
            stop();
        }
    }
    
    public void sendState(float[] observation) {
        if (!clientConnected.get() || out == null) return;
        
        try {
            StateMessage msg = new StateMessage(observation);
            String json = gson.toJson(msg);
            out.println(json);
        } catch (Exception e) {
            log.error("[OSRS-RL] Error sending state: {}", e.getMessage());
        }
    }
    
    public void stop() {
        running.set(false);
        closeClient();
        
        try {
            if (serverSocket != null && !serverSocket.isClosed()) {
                serverSocket.close();
            }
        } catch (IOException e) {
            log.error("[OSRS-RL] Error closing server: {}", e.getMessage());
        }
        
        log.info("[OSRS-RL] Server stopped");
    }
    
    private void closeClient() {
        try {
            if (in != null) in.close();
            if (out != null) out.close();
            if (clientSocket != null && !clientSocket.isClosed()) clientSocket.close();
        } catch (IOException e) {
            // Ignore
        }
        in = null;
        out = null;
        clientSocket = null;
    }
    
    public boolean isClientConnected() {
        return clientConnected.get();
    }
    
    // Message classes for JSON serialization
    private static class StateMessage {
        float[] observation;
        long timestamp;
        
        StateMessage(float[] observation) {
            this.observation = observation;
            this.timestamp = System.currentTimeMillis();
        }
    }
    
    private static class ActionMessage {
        int[] action;
    }
}
