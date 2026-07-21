import { useState, useEffect, useRef } from 'react';

export const useDroneLogic = (stream, addLog) => {
  const [mode, setMode] = useState('MANUAL');
  const [preHoverMode, setPreHoverMode] = useState('MANUAL');
  const [target, setTarget] = useState({ x: 0, y: 0, status: 'IDLE' });
  const [abortConfirm, setAbortConfirm] = useState(false);
  const targetLockTimeoutRef = useRef(null);

  useEffect(() => {
    return () => {
      if (targetLockTimeoutRef.current) clearTimeout(targetLockTimeoutRef.current);
    };
  }, []);

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
        if (mode === 'HOVER') {
            // Return to whatever mode was active before HOVER was engaged, not always MANUAL.
            setMode(preHoverMode);
            addLog("Hover Cancelled", "INFO");
            sendToSimulator({ Command: "SetMode", Mode: preHoverMode });
        } else {
            setPreHoverMode(mode);
            setMode('HOVER');
            addLog("Position Hold Engaged", "WARN");
            sendToSimulator({ Command: "SetMode", Mode: "HOVER" });
        }
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
    else if (cmd === 'SYNC') {
        addLog("Requesting telemetry re-sync with simulator...", "INFO");
        sendToSimulator({ Command: "SyncTelemetry" });
    }
  };

  // pixelX/pixelY position the target overlay in screen space; normX/normY (0-1, resolution
  // independent) are what actually gets sent to Unreal, since raw video-element pixels don't
  // mean anything to the simulator if the feed is scaled/letterboxed in the browser.
  const handleTargetLock = (pixelX, pixelY, normX, normY) => {
      if (targetLockTimeoutRef.current) {
          clearTimeout(targetLockTimeoutRef.current);
      }

      setTarget({ x: pixelX, y: pixelY, status: 'SEARCHING' });
      addLog(`Acquiring target...`, "INFO");

      sendToSimulator({ Command: "SetTarget", TargetX: normX, TargetY: normY });

      targetLockTimeoutRef.current = setTimeout(() => {
          setTarget(prev => ({ ...prev, status: 'LOCKED' }));
          addLog("Target Locked. Confidence: 98%", "SUCCESS");
          targetLockTimeoutRef.current = null;
      }, 1000);
  };

  return { mode, target, abortConfirm, handleCommand, handleTargetLock };
};
