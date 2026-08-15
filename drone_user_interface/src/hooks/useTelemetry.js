import { useState, useEffect } from 'react';

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

export const useTelemetry = (stream) => {
  const [telemetry, setTelemetry] = useState({
    alt: 0, speed: 0.0, bat: 100, lat: 32.0853, lng: 34.7818,
    pitch: 0, roll: 0, coreTemp: 42.0, escTemp: 35.0, linkQuality: 100
  });

  useEffect(() => {
    const handleTelemetry = (response) => {
        try {
            const data = JSON.parse(response);

            setTelemetry(prev => ({
                ...prev,
                lat: data.X !== undefined ? translateUnrealToLat(data.X) : prev.lat,
                lng: data.Y !== undefined ? translateUnrealToLng(data.Y) : prev.lng,
                alt: data.Z !== undefined ? data.Z / 100 : prev.alt, 
                speed: data.Speed !== undefined ? data.Speed : prev.speed,
                pitch: data.Pitch !== undefined ? data.Pitch : prev.pitch,
                roll: data.Roll !== undefined ? data.Roll : prev.roll,
            }));
        } catch (err) {
            console.error("Failed to parse telemetry:", err);
        }
    };

    if (stream) {
        stream.addResponseEventListener(handleTelemetry);
    }

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
        if (stream) {
            stream.removeResponseEventListener(handleTelemetry);
        }
    };
  }, [stream]);

  return { telemetry };
};