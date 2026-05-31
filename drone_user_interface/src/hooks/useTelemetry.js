import { useState, useEffect } from 'react';

export const useTelemetry = () => {
  const [telemetry, setTelemetry] = useState({ 
    alt: 15.4, speed: 0.0, bat: 100, lat: 32.0853, lng: 34.7818,
    pitch: 0, roll: 0, coreTemp: 42.0, escTemp: 35.0, linkQuality: 100
  });

  useEffect(() => {
    // Listen to real telemetry if running in Electron
    if (window.electronAPI) {
        window.electronAPI.onTelemetry((event, data) => setTelemetry(prev => ({...prev, ...data})));
    }

    // Hardware status data simulation
    const interval = setInterval(() => {
        setTelemetry(prev => {
            const currentPitch = Number(prev.pitch) || 0;
            const currentRoll = Number(prev.roll) || 0;
            const currentAlt = Number(prev.alt) || 0;
            const currentLat = Number(prev.lat) || 32.0853;
            const currentLng = Number(prev.lng) || 34.7818;
            
            const newCore = Math.min(85, Math.max(30, prev.coreTemp + (Math.random() - 0.4)));
            const newEsc = Math.min(90, Math.max(30, prev.escTemp + (Math.random() - 0.3)));
            const newLink = Math.min(100, Math.max(50, prev.linkQuality + (Math.random() - 0.5) * 5));

            const newPitch = currentPitch + (Math.random() - 0.5) * 2;
            const newRoll = currentRoll + (Math.random() - 0.5) * 2;
            
            return {
                ...prev,
                lat: currentLat + (Math.random() - 0.5) * 0.00005,
                lng: currentLng + (Math.random() - 0.5) * 0.00005,
                pitch: newPitch * 0.9, 
                roll: newRoll * 0.9,
                alt: Math.max(0, currentAlt + (Math.random() - 0.5) * 0.1),
                coreTemp: newCore,
                escTemp: newEsc,
                linkQuality: newLink
            };
        });
    }, 100);

    return () => clearInterval(interval);
  }, []);

  return { telemetry, setTelemetry };
};