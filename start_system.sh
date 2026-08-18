#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' 

echo -e "${GREEN}Initializing AeroGuard System Startup...${NC}\n"

ORIGINAL_DIR=$(pwd)

# ---------------------------------------------------------
# [0] First-Run Installations
# ---------------------------------------------------------
SETUP_FLAG="$ORIGINAL_DIR/.aeroguard_setup_done"

if [ ! -f "$SETUP_FLAG" ]; then
    echo -e "${BLUE}First run detected. Running required installations...${NC}\n"
    
    echo -e "${BLUE}-> Installing Python dependencies...${NC}"
    pip install fastapi uvicorn cosysairsim
    
    echo -e "\n${BLUE}-> Installing Pixel Streaming WebServer dependencies...${NC}"
    cd "C:/Program Files/Epic Games/UE_5.5/Engine/Plugins/Media/PixelStreaming/Resources/WebServers" || { echo -e "${RED}Failed to find WebServers directory${NC}"; exit 1; }
    npm install
    npm run build
    
    cd "$ORIGINAL_DIR"
    
    touch "$SETUP_FLAG"
    echo -e "\n${GREEN}Initial setup completed successfully.${NC}\n"
else
    echo -e "${GREEN}Setup already completed in a previous run. Skipping installations.${NC}\n"
fi

# ---------------------------------------------------------
# [1/4] Pixel Streaming Signalling Server
# ---------------------------------------------------------
echo -e "${BLUE}[1/4] Starting Pixel Streaming Signalling Server...${NC}"
cd "C:/Program Files/Epic Games/UE_5.5/Engine/Plugins/Media/PixelStreaming/Resources/WebServers/SignallingWebServer" || exit 1
npm run start &
SIG_PID=$!

cd "$ORIGINAL_DIR"

# ---------------------------------------------------------
# [2/4] Telemetry Server (Node.js)
# ---------------------------------------------------------
echo -e "${BLUE}[2/4] Starting Telemetry Server...${NC}"
cd telemetry_server || { echo -e "${RED}Failed to find telemetry_server directory${NC}"; exit 1; }
node server.js &
SERVER_PID=$!

cd "$ORIGINAL_DIR"

# ---------------------------------------------------------
# [3/4] AirSim Python API Server
# ---------------------------------------------------------
echo -e "${BLUE}[3/4] Starting AirSim Python API Server...${NC}"
cd telemetry_server || { echo -e "${RED}Failed to find telemetry_server directory${NC}"; exit 1; }
python airsim_api.py &
PYTHON_PID=$!

cd "$ORIGINAL_DIR" 

echo -e "${BLUE}Waiting for servers to initialize...${NC}"
sleep 2

# ---------------------------------------------------------
# [4/4] React UI
# ---------------------------------------------------------
echo -e "${BLUE}[4/4] Starting Drone User Interface...${NC}"
cd drone_user_interface || { echo -e "${RED}Failed to find drone_user_interface directory${NC}"; exit 1; }
npm run dev &
UI_PID=$!

cd "$ORIGINAL_DIR"

echo -e "\n${GREEN} All systems are online!${NC}"
echo -e "Press ${RED}[CTRL+C]${NC} to safely stop all services.\n"

trap "echo -e '\n${RED}Shutting down systems...${NC}'; kill $SIG_PID $SERVER_PID $PYTHON_PID $UI_PID; exit" SIGINT SIGTERM

wait $SIG_PID $SERVER_PID $PYTHON_PID $UI_PID