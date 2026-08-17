import { useState, useEffect } from 'react';

const DEFAULT_LAT = 32.0853;
const DEFAULT_LNG = 34.7818;

const translateUnrealToLat = (unrealX) => {
  const unrealMin = -120000, unrealMax = 120000;
  const latMin = 32.0000, latMax = 32.0800;
  return latMin + ((unrealX - unrealMin) * (latMax - latMin)) / (unrealMax - unrealMin);
};

const translateUnrealToLng = (unrealY) => {
  const unrealMin = -120000, unrealMax = 120000;
  const lngMin = 34.7000, lngMax = 34.81635;
  return lngMin + ((unrealY - unrealMin) * (lngMax - lngMin)) / (unrealMax - unrealMin);
};

export const useTelemetry = () => {
  const [telemetry, setTelemetry] = useState({
    alt: 0, speed: 0.0, bat: 100, lat: DEFAULT_LAT, lng: DEFAULT_LNG,
    pitch: 0, roll: 0, yaw: 0, 
    coreTemp: 42.0, escTemp: 35.0, linkQuality: 100
  });

  useEffect(() => {
    const ws = new WebSocket('ws://127.0.0.1:4001');

    ws.onmessage = (event) => {
        try {
            const values = JSON.parse(event.data);
            
            const keys = ["x", "y", "z", "speed", "roll", "pitch", "yaw"];
            let data = {};

            if (Array.isArray(values)) {
                if (values.length !== keys.length) {
                    throw new Error(`Expected ${keys.length} values, received ${values.length}`);
                }
                
                keys.forEach((key, index) => {
                    data[key] = values[index];
                });
            } 
            else if (typeof values === 'object' && values !== null) {
                data = values;
            }

            if (Object.keys(data).length > 0) {
                setTelemetry(prev => ({
                    ...prev,
                    lat: data.x !== undefined && data.x !== null ? translateUnrealToLat(data.x) : prev.lat,
                    lng: data.y !== undefined && data.y !== null ? translateUnrealToLng(data.y) : prev.lng,
                    alt: data.z !== undefined ? data.z : prev.alt, 
                    speed: data.speed !== undefined ? data.speed : prev.speed,
                    pitch: data.pitch !== undefined ? data.pitch : prev.pitch,
                    roll: data.roll !== undefined ? data.roll : prev.roll,
                    yaw: data.yaw !== undefined ? data.yaw : prev.yaw,
                }));
            }
            
        } catch (err) {
            console.error("Failed to parse telemetry from WS:", err);
            console.log("Raw string that caused error:", event.data);
        }
    };

    ws.onopen = () => console.log("Connected to Telemetry Server");
    ws.onerror = (err) => console.error("Telemetry WS Error:", err);

    const interval = setInterval(() => {
        setTelemetry(prev => ({
            ...prev,
            bat: Math.max(0, prev.bat - 0.01),
            linkQuality: Math.min(100, Math.max(50, prev.linkQuality + (Math.random() - 0.5) * 5)),
            coreTemp: Math.min(85, Math.max(30, prev.coreTemp + (Math.random() - 0.4)))
        }));
    }, 1000);

    return () => {
        clearInterval(interval);
        ws.close();
    };
  }, []); 

  return { telemetry };
};