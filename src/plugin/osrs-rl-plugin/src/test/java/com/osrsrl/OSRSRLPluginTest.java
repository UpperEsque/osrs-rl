package com.osrsrl;

import net.runelite.client.RuneLite;
import net.runelite.client.externalplugins.ExternalPluginManager;

public class OSRSRLPluginTest {
    public static void main(String[] args) throws Exception {
        ExternalPluginManager.loadBuiltin(OSRSRLPlugin.class);
        RuneLite.main(args);
    }
}
