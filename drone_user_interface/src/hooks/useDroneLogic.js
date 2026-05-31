import { useState } from 'react';

export const useDroneLogic = () => {
  const [mode, setMode] = useState('MANUAL');
  const [target, setTarget] = useState({ x: 0, y: 0, status: 'IDLE' }); 
  const [logs, setLogs] = useState([]); 
  const [abortConfirm, setAbortConfirm] = useState(false);

  // Add system log
  const addLog = (message, type = 'INFO') => {
    const time = new Date().toLocaleTimeString('en-GB', { hour12: false });
    setLogs(prev => [{ id: crypto.randomUUID(), time, message, type }, ...prev].slice(0, 50));
  };

  // UI Commands handler
  const handleCommand = (cmd) => {
    if (cmd === 'ABORT') {
        if (!abortConfirm) {
            setAbortConfirm(true);
            addLog("⚠️ ABORT ARMED! PRESS AGAIN TO CONFIRM!", "WARN");
            setTimeout(() => {
                setAbortConfirm(curr => {
                    if (curr) addLog("Abort Cancelled (Timeout)", "INFO");
                    return false;
                });
            }, 3000);
        } else {
            addLog("🚨 EMERGENCY ABORT EXECUTED 🚨", "ERROR");
            setMode('EMERGENCY');
            setAbortConfirm(false);
            setTarget({ ...target, status: 'IDLE' });
        }
        return; 
    }

    if (abortConfirm) setAbortConfirm(false);

    if (cmd === 'TOGGLE_MODE') {
        if (mode === 'MANUAL') {
            addLog("Autonomous Mode Engaged", "WARN");
            setMode('AUTO');
        } else {
            addLog("Manual Control Engaged", "INFO");
            setMode('MANUAL');
        }
        setTarget({ ...target, status: 'IDLE' });
    }
    else if (cmd === 'HOVER') {
        if (mode === 'HOVER') {
            addLog("Hover Cancelled. Returning to Manual.", "INFO");
            setMode('MANUAL');
        } else {
            addLog("Position Hold Engaged (Loiter)", "INFO");
            setMode('HOVER');
        }
    }
    else if (cmd === 'SYNC') {
        addLog("Syncing Mission Data...", "SYS");
        setTimeout(() => addLog("Data Sync Complete", "SUCCESS"), 1000);
    }
    else if (cmd === 'RTH') {
        addLog("Initiating Return to Home...", "WARN");
        setMode('RTH');
        setTarget({ ...target, status: 'IDLE' });
    }
    else if (cmd === 'FOLLOW') {
        if (target.status === 'LOCKED') {
            addLog(`Following Target at [${Number(target.x).toFixed(0)}, ${Number(target.y).toFixed(0)}]`, "INFO");
            setMode('FOLLOW');
        } else {
            addLog("Cannot Follow: No Target Locked", "ERROR");
        }
    }
    else if (cmd === 'LAND') {
        if (target.status === 'LOCKED') {
            addLog("LANDING SEQUENCE STARTED", "WARN");
            setMode('LANDING');
        } else {
            addLog("Landing Aborted: No valid target", "ERROR");
        }
    }
  };

  return { mode, target, setTarget, logs, abortConfirm, addLog, handleCommand };
};