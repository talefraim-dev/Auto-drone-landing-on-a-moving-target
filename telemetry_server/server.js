const express = require('express');
const http = require('http');
const { WebSocketServer } = require('ws');

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

app.use(express.json());

const connectedClients = new Set();

wss.on('connection', (ws) => {
    console.log('React UI Connected to WebSocket');
    connectedClients.add(ws);

    ws.on('close', () => {
        console.log('React UI Disconnected');
        connectedClients.delete(ws);
    });
});

app.post('/telemetry', (req, res) => {
    const telemetryData = req.body;

    const dataString = JSON.stringify(telemetryData);

    for (const client of connectedClients) {
        if (client.readyState === 1) { 
            client.send(dataString);
        }
    }

    res.status(200).send({ status: 'Data broadcasted successfully' });
});

const PORT = 4000;
server.listen(PORT, () => {
    console.log(`Telemetry Server is running!`);
    console.log(`Listening for Unreal POST requests on http://127.0.0.1:${PORT}/telemetry`);
    console.log(`Listening for React WebSocket connections on ws://127.0.0.1:${PORT}`);
});