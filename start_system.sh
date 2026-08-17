#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' 

echo -e "${GREEN}Initializing AeroGuard System Startup...${NC}\n"

ORIGINAL_DIR=$(pwd)

# ---------------------------------------------------------
# [1/3] Pixel Streaming Signalling Server
# ---------------------------------------------------------
echo -e "${BLUE}[1/3] Starting Pixel Streaming Signalling Server...${NC}"

cd "C:/Program Files/Epic Games/UE_5.5/Engine/Plugins/Media/PixelStreaming/Resources/WebServers/SignallingWebServer" || { echo -e "${RED}Failed to find Signalling Server directory${NC}"; exit 1; }

npm install
npm run build
npm run start &
SIG_PID=$!

cd "$ORIGINAL_DIR"

# ---------------------------------------------------------
# [2/3] Telemetry Server
# ---------------------------------------------------------
echo -e "${BLUE}[2/3] Starting Telemetry Server...${NC}"
cd telemetry_server || { echo -e "${RED}Failed to find telemetry_server directory${NC}"; exit 1; }
node server.js &
SERVER_PID=$!

cd "$ORIGINAL_DIR"

echo -e "${BLUE}Waiting for servers to initialize...${NC}"
sleep 2

# ---------------------------------------------------------
# [3/3] React UI
# ---------------------------------------------------------
echo -e "${BLUE}[3/3] Starting Drone User Interface...${NC}"
cd drone_user_interface || { echo -e "${RED}Failed to find drone_user_interface directory${NC}"; exit 1; }
npm run dev &
UI_PID=$!

cd "$ORIGINAL_DIR"

echo -e "\n${GREEN}✅ All systems are online!${NC}"
echo -e "Press ${RED}[CTRL+C]${NC} to safely stop all services.\n"

trap "echo -e '\n${RED}Shutting down systems...${NC}'; kill $SIG_PID $SERVER_PID $UI_PID; exit" SIGINT SIGTERM

wait $SIG_PID $SERVER_PID $UI_PID