const express = require('express');
const http = require('http');
const { WebSocketServer } = require('ws');

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

app.use(express.json());
const connectedClients = new Set();

wss.on('connection', (ws) => {
    console.log('Client connected to WebSocket');
    connectedClients.add(ws);

    ws.on('message', (message) => {
        const dataString = message.toString();

        for (const client of connectedClients) {
            if (client !== ws && client.readyState === 1) { 
                client.send(dataString);
            }
        }
    });

    ws.on('close', () => {
        console.log('Client disconnected');
        connectedClients.delete(ws);
    });
});

const PORT = 4000;
server.listen(PORT, () => {
    console.log(`Telemetry Server is running!`);
    console.log(`Listening for React and Unreal WebSocket connections on ws://127.0.0.1:${PORT}`);
});