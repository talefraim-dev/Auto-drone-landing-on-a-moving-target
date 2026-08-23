import { useState } from 'react';

export const useDroneLogic = (stream) => {
  const [mode, setMode] = useState('MANUAL');
  const [target, setTarget] = useState({ x: 0, y: 0, status: 'IDLE' }); 
  const [logs, setLogs] = useState([]); 
  const [abortConfirm, setAbortConfirm] = useState(false);

  const addLog = (message, type = 'INFO') => {
    const time = new Date().toLocaleTimeString('en-GB', { hour12: false });
    setLogs(prev => [{ id: crypto.randomUUID(), time, message, type }, ...prev].slice(0, 50));
  };

  const sendToSimulator = (commandObj) => {
    if (stream) {
        stream.emitUIInteraction(commandObj);
    } else {
        console.warn("Stream disconnected. Cannot send:", commandObj);
    }
  };

  const handleCommand = (cmd) => {
    if (cmd === 'ABORT') {
        if (!abortConfirm) {
            setAbortConfirm(true);
            addLog("⚠️ ABORT ARMED! PRESS AGAIN TO CONFIRM!", "WARN");
            setTimeout(() => setAbortConfirm(false), 3000);
        } else {
            addLog("🚨 EMERGENCY ABORT EXECUTED 🚨", "ERROR");
            setMode('EMERGENCY');
            setAbortConfirm(false);
            setTarget({ ...target, status: 'IDLE' });
            sendToSimulator({ Command: "SetMode", Mode: "ABORT" }); 
        }
        return; 
    }

    if (abortConfirm) setAbortConfirm(false);

    if (cmd === 'TOGGLE_MODE') {
        const newMode = mode === 'MANUAL' ? 'AUTO' : 'MANUAL';
        setMode(newMode);
        addLog(`${newMode} Mode Engaged`, newMode === 'AUTO' ? "WARN" : "INFO");
        sendToSimulator({ Command: "SetMode", Mode: newMode }); 
    }
    else if (cmd === 'HOVER') {
        const newMode = mode === 'HOVER' ? 'MANUAL' : 'HOVER';
        setMode(newMode);
        addLog(newMode === 'HOVER' ? "Position Hold Engaged" : "Hover Cancelled", "INFO");
        sendToSimulator({ Command: "SetMode", Mode: newMode }); 
    }
    else if (cmd === 'RTH') {
        addLog("Initiating Return to Home...", "WARN");
        setMode('RTH');
        sendToSimulator({ Command: "SetMode", Mode: "RTH" }); 
    }
    else if (cmd === 'LAND') {
        if (target.status === 'LOCKED') {
            addLog("LANDING SEQUENCE STARTED", "WARN");
            setMode('LANDING');
            sendToSimulator({ Command: "ExecuteLanding" }); 
        } else {
            addLog("Landing Aborted: No valid target", "ERROR");
        }
    }
  };

  const handleTargetLock = (bbox) => {
      setTarget({ 
        x: bbox.x, 
        y: bbox.y, 
        width: bbox.width, 
        height: bbox.height, 
        status: 'SEARCHING' 
      });
      
      addLog(`Acquiring target...`, "INFO");
      
      fetch('http://127.0.0.1:8000/api/SetTargetBBox', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
              TargetX: bbox.normX,
              TargetY: bbox.normY,
              TargetW: bbox.normW,
              TargetH: bbox.normH
          })
      })
      .then(res => res.json())
      .then(data => console.log("BBox sent to API:", data))
      .catch(err => console.error("Failed to send BBox:", err));

      setTimeout(() => {
          setTarget(prev => ({ ...prev, status: 'LOCKED' }));
          addLog("Target Locked. Confidence: 98%", "SUCCESS");
      }, 1000);
  };

  return { mode, target, logs, abortConfirm, addLog, handleCommand, handleTargetLock };
};