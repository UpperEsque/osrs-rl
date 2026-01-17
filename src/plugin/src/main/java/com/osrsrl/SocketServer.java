package com.osrsrl;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonElement;
import com.google.gson.JsonArray;
import lombok.extern.slf4j.Slf4j;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

@Slf4j
public class SocketServer {

    private final int port;
    private final Consumer<int[]> actionHandler;
    private final Gson gson = new Gson();

    private ServerSocket serverSocket;
    private Socket clientSocket;
    private PrintWriter out;
    private BufferedReader in;
    private ExecutorService executor;

    private volatile boolean running = false;
    private volatile boolean connected = false;

    public SocketServer(int port, Consumer<int[]> actionHandler) {
        this.port = port;
        this.actionHandler = actionHandler;
    }

    public void start() {
        executor = Executors.newFixedThreadPool(2);
        running = true;
        executor.submit(this::acceptConnections);
    }

    public void stop() {
        running = false;
        connected = false;
        try {
            if (clientSocket != null) clientSocket.close();
            if (serverSocket != null) serverSocket.close();
        } catch (IOException e) {
            log.error("[OSRS-RL] Error stopping server", e);
        }
        if (executor != null) {
            executor.shutdownNow();
        }
    }

    private void acceptConnections() {
        try {
            serverSocket = new ServerSocket(port, 50, java.net.InetAddress.getByName("0.0.0.0"));
            log.info("[OSRS-RL] Server listening on port {}", port);

            while (running) {
                try {
                    log.info("[OSRS-RL] Waiting for client connection...");
                    clientSocket = serverSocket.accept();
                    clientSocket.setTcpNoDelay(true);

                    out = new PrintWriter(
                        new OutputStreamWriter(clientSocket.getOutputStream(), StandardCharsets.UTF_8),
                        true
                    );
                    in = new BufferedReader(
                        new InputStreamReader(clientSocket.getInputStream(), StandardCharsets.UTF_8)
                    );

                    connected = true;
                    log.info("[OSRS-RL] Python client connected!");
                    readActions();

                } catch (IOException e) {
                    if (running) {
                        log.error("[OSRS-RL] Connection error", e);
                    }
                }
            }
        } catch (IOException e) {
            log.error("[OSRS-RL] Server error", e);
        }
    }

    private void readActions() {
        try {
            String line;
            while (connected && (line = in.readLine()) != null) {
                log.info("[OSRS-RL] Received: {}", line);
                try {
                    JsonObject msg = gson.fromJson(line, JsonObject.class);
                    String type = msg.get("type").getAsString();

                    if ("action".equals(type)) {
                        // Try to get action from "action" field (Python sends this)
                        JsonElement actionElement = msg.get("action");
                        
                        if (actionElement == null) {
                            // Fallback to "data" field
                            actionElement = msg.get("data");
                        }
                        
                        if (actionElement == null) {
                            log.error("[OSRS-RL] No action or data field found");
                            continue;
                        }
                        
                        int[] action;
                        if (actionElement.isJsonArray()) {
                            // It's an array like [1, 100, 200]
                            JsonArray arr = actionElement.getAsJsonArray();
                            action = new int[arr.size()];
                            for (int i = 0; i < arr.size(); i++) {
                                action[i] = arr.get(i).getAsInt();
                            }
                        } else {
                            // It's a single int like 7
                            action = new int[] { actionElement.getAsInt() };
                        }
                        
                        log.info("[OSRS-RL] Parsed action: {}", java.util.Arrays.toString(action));
                        actionHandler.accept(action);
                    }
                } catch (Exception e) {
                    log.error("[OSRS-RL] Parse error: {}", e.getMessage());
                }
            }
        } catch (IOException e) {
            log.error("[OSRS-RL] Read error: {}", e.getMessage());
        } finally {
            connected = false;
            log.info("[OSRS-RL] Python client disconnected");
        }
    }

    public void sendState(GameState state) {
        if (!connected || out == null) return;
        try {
            JsonObject msg = new JsonObject();
            msg.addProperty("type", "state");
            msg.add("data", gson.toJsonTree(state));
            out.println(gson.toJson(msg));
        } catch (Exception e) {
            log.error("[OSRS-RL] Send error: {}", e.getMessage());
            connected = false;
        }
    }

    public boolean isConnected() {
        return connected;
    }
}
