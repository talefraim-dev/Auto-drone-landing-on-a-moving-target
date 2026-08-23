import { useState, useEffect } from 'react';

export const useTelemetry = (stream) => {
  const [telemetry, setTelemetry] = useState({ 
    alt: 0, speed: 0.0, bat: 100, lat: 32.0853, lng: 34.7818,
    pitch: 0, roll: 0, yaw: 0, coreTemp: 42.0, escTemp: 35.0, linkQuality: 100
  });

  useEffect(() => {
    const fetchTelemetry = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/api/telemetry');
        if (response.ok) {
          const data = await response.json();
          if (!data.error) {
            setTelemetry(prev => ({
              ...prev,
              lat: data.lat !== undefined ? data.lat : prev.lat,
              lng: data.lng !== undefined ? data.lng : prev.lng,
              alt: data.z !== undefined ? -data.z : prev.alt, 
              speed: data.speed !== undefined ? data.speed : prev.speed,
              pitch: data.pitch !== undefined ? data.pitch : prev.pitch,
              roll: data.roll !== undefined ? data.roll : prev.roll,
              yaw: data.yaw !== undefined ? data.yaw : prev.yaw
            }));
          }
        }
      } catch (err) {
        console.error("Failed to fetch telemetry from Python API:", err);
      }
    };

    const telemetryInterval = setInterval(fetchTelemetry, 100);

    const simInterval = setInterval(() => {
        setTelemetry(prev => ({
            ...prev,
            bat: Math.max(0, prev.bat - 0.01),
            linkQuality: Math.min(100, Math.max(50, prev.linkQuality + (Math.random() - 0.5) * 5)),
            coreTemp: Math.min(85, Math.max(30, prev.coreTemp + (Math.random() - 0.4)))
        }));
    }, 1000);

    return () => {
      clearInterval(telemetryInterval);
      clearInterval(simInterval);
    };
  }, []); 

  return { telemetry };
};