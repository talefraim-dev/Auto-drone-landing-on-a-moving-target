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
    const ws = new WebSocket('ws://127.0.0.1:8080');

    ws.onmessage = (event) => {
        try {
            const rawStr = event.data.trim();
            let data = {};
            const keys = ["x", "y", "z", "speed", "roll", "pitch", "yaw"];

            if (rawStr.startsWith('[') && rawStr.endsWith(']')) {
                const numberMatches = rawStr.match(/[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?/g);
                
                if (!numberMatches) return;

                const values = numberMatches.map(n => parseFloat(n.replace(/,/g, '')));

                if (values.length !== keys.length) {
                    console.warn(`Expected ${keys.length} values, received ${values.length}`, rawStr);
                    return; 
                }

                keys.forEach((key, index) => {
                    data[key] = values[index];
                });
            } 
            else if (rawStr.startsWith('{') && rawStr.endsWith('}')) {
                const parsed = JSON.parse(rawStr);
                if (typeof parsed === 'object' && parsed !== null) {
                    data = parsed;
                }
            }

            if (Object.keys(data).length > 0) {
                setTelemetry(prev => ({
                    ...prev,
                    lat: data.x !== undefined && data.x !== null ? translateUnrealToLat(data.x) : prev.lat,
                    lng: data.y !== undefined && data.y !== null ? translateUnrealToLng(data.y) : prev.lng,
                    alt: data.z !== undefined ? data.z : prev.alt, 
                    pitch: data.pitch !== undefined ? data.pitch : prev.pitch,
                    roll: data.roll !== undefined ? data.roll : prev.roll,
                    yaw: data.yaw !== undefined ? data.yaw : prev.yaw,
                }));
            }
            
        } catch (err) {
            console.error("Failed to parse telemetry from WS:", err);
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
        if (ws.readyState === WebSocket.OPEN) {
            ws.close();
        } else if (ws.readyState === WebSocket.CONNECTING) {
            ws.onopen = () => ws.close();
        }
    };
  }, []); 

  useEffect(() => {
    const fetchSpeedFromAPI = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8001/api/speed');
        
        if (response.ok) {
          const data = await response.json();
          if (data.speed !== undefined) {
            setTelemetry(prev => ({
              ...prev,
              speed: parseFloat(data.speed.toFixed(2)) 
            }));
          }
        }
      } catch (error) {
         console.error("Failed to fetch speed:", error);
      }
    };

    const speedInterval = setInterval(fetchSpeedFromAPI, 100);

    return () => clearInterval(speedInterval);
  }, []);

  return { telemetry };
};