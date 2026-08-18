import { useState, useEffect } from 'react';

const DEFAULT_LAT = 32.0853;
const DEFAULT_LNG = 34.7818;

export const useTelemetry = () => {
  const [telemetry, setTelemetry] = useState({
    alt: 0, speed: 0.0, bat: 100, lat: DEFAULT_LAT, lng: DEFAULT_LNG,
    pitch: 0, roll: 0, yaw: 0, 
    coreTemp: 42.0, escTemp: 35.0, linkQuality: 100
  });

  useEffect(() => {
    const fetchTelemetryFromAPI = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/api/telemetry');
        
        if (response.ok) {
          const data = await response.json();
          
          if (!data.error) {
            setTelemetry(prev => ({
              ...prev,
              lat: data.lat !== undefined ? data.lat : prev.lat,
              lng: data.lng !== undefined ? data.lng : prev.lng,
              alt: data.z !== undefined ? parseFloat((-data.z).toFixed(2)) : prev.alt, 
              speed: data.speed !== undefined ? parseFloat(data.speed.toFixed(2)) : prev.speed,
              pitch: data.pitch !== undefined ? parseFloat(data.pitch.toFixed(2)) : prev.pitch,
              roll: data.roll !== undefined ? parseFloat(data.roll.toFixed(2)) : prev.roll,
              yaw: data.yaw !== undefined ? parseFloat(data.yaw.toFixed(2)) : prev.yaw,
            }));
          }
        }
      } catch (error) {
      }
    };

    const telemetryInterval = setInterval(fetchTelemetryFromAPI, 500);
    return () => clearInterval(telemetryInterval);
  }, []);

  useEffect(() => {
    const simInterval = setInterval(() => {
        setTelemetry(prev => ({
            ...prev,
            bat: Math.max(0, prev.bat - 0.01),
            linkQuality: Math.min(100, Math.max(50, prev.linkQuality + (Math.random() - 0.5) * 5)),
            coreTemp: Math.min(85, Math.max(30, prev.coreTemp + (Math.random() - 0.4)))
        }));
    }, 1000);

    return () => clearInterval(simInterval);
  }, []);

  return { telemetry };
};