#!/bin/bash
# Run RuneLite with the OSRS-RL plugin

PLUGIN_JAR="$HOME/osrs-rl/plugin/build/libs/osrs-rl-plugin-1.0.0.jar"

if [ ! -f "$PLUGIN_JAR" ]; then
    echo "Plugin not built. Building..."
    cd ~/osrs-rl/plugin && gradle build
fi

# Check if RuneLite is installed
if command -v runelite &> /dev/null; then
    echo "Starting RuneLite..."
    runelite --developer-mode --external-plugin-dir="$HOME/osrs-rl/plugin/build/libs"
else
    echo "RuneLite not found in PATH."
    echo ""
    echo "Options:"
    echo "1. Download RuneLite from https://runelite.net/"
    echo "2. Or use the RuneLite launcher with external plugins"
    echo ""
    echo "To manually load the plugin:"
    echo "  - Copy $PLUGIN_JAR to ~/.runelite/plugins/"
    echo "  - Or run RuneLite with --developer-mode"
fi
