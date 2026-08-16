#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' 

echo -e "${GREEN}Initializing AeroGuard System Startup...${NC}\n"

echo -e "${BLUE}[1/2] Starting Telemetry Server...${NC}"
cd telemetry_server || { echo -e "${RED}Failed to find telemetry_server directory${NC}"; exit 1; }
node server.js &
SERVER_PID=$!
cd ..

echo -e "${BLUE}[2/2] Starting Drone User Interface...${NC}"
cd drone_user_interface || { echo -e "${RED}Failed to find drone_user_interface directory${NC}"; exit 1; }
npm run dev &
UI_PID=$!
cd ..

echo -e "\n${GREEN}✅ All systems are online!${NC}"
echo -e "Press ${RED}[CTRL+C]${NC} to safely stop both services.\n"

trap "echo -e '\n${RED}Shutting down systems...${NC}'; kill $SERVER_PID $UI_PID; exit" SIGINT SIGTERM

wait $SERVER_PID $UI_PID