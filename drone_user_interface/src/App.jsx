import { useState, useEffect, useRef } from 'react'
import { MapContainer, TileLayer, Popup, CircleMarker } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import './App.css'

function App() {
  // telemetry data
  const [telemetry, setTelemetry] = useState({ 
    alt: 15.4, 
    speed: 0.0, 
    bat: 100,
    lat: 32.0853, 
    lng: 34.7818,
    pitch: 0, 
    roll: 0,
    coreTemp: 42.0,
    escTemp: 35.0,
    linkQuality: 100
  });
  
  const [mode, setMode] = useState('MANUAL');
  const [target, setTarget] = useState({ x: 0, y: 0, status: 'IDLE' }); 
  const [logs, setLogs] = useState([]); 
  
  // State for abort confirmation
  const [abortConfirm, setAbortConfirm] = useState(false);

  const videoRef = useRef(null); 

  // add log
  const addLog = (message, type = 'INFO') => {
    const time = new Date().toLocaleTimeString('en-GB', { hour12: false });
    setLogs(prev => [{ id: Date.now(), time, message, type }, ...prev].slice(0, 50));
  };

  // camera activate
  useEffect(() => {
    addLog("Initializing AeroGuard System...", "SYS");
    async function setupCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
        if (videoRef.current) videoRef.current.srcObject = stream;
        addLog("Camera Feed Connected", "SUCCESS");
      } catch (err) {
        addLog("Camera Connection Failed: " + err.message, "ERROR");
      }
    }
    setupCamera();
  }, []);

  // data simulation
  useEffect(() => {
    if (window.electronAPI) {
        window.electronAPI.onTelemetry((event, data) => setTelemetry(prev => ({...prev, ...data})));
    }

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

        setTarget(prev => {
            if (prev.status !== 'LOCKED') return prev;
            return {
                ...prev,
                x: prev.x + (Math.random() - 0.5) * 3,
                y: prev.y + (Math.random() - 0.5) * 3
            };
        });

    }, 100);

    return () => clearInterval(interval);
  }, []);

  // buttons
  const handleCommand = (cmd) => {
    
    // Abort button
    if (cmd === 'ABORT') {
        if (!abortConfirm) {
            // first click on button
            setAbortConfirm(true);
            addLog("⚠️ ABORT ARMED! PRESS AGAIN TO CONFIRM!", "WARN");
            
            // 3 sec timer for cancel abort
            setTimeout(() => {
                setAbortConfirm(curr => {
                    if (curr) addLog("Abort Cancelled (Timeout)", "INFO");
                    return false;
                });
            }, 3000);
        } else {
            // second click - activate
            addLog("🚨 EMERGENCY ABORT EXECUTED 🚨", "ERROR");
            setMode('EMERGENCY');
            setAbortConfirm(false);
            setTarget({ ...target, status: 'IDLE' });
            
        }
        return; 
    }

    
    if (abortConfirm) {
        setAbortConfirm(false);
    }

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

  // video handler
  const handleVideoClick = (e) => {
    if (mode === 'LANDING' || mode === 'EMERGENCY') return;

    const rect = e.target.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setTarget({ x, y, status: 'SEARCHING' });
    addLog(`Acquiring target...`, "INFO");

    setTimeout(() => {
        setTarget(prev => ({ ...prev, status: 'LOCKED' }));
        addLog("Target Locked. Confidence: 98%", "SUCCESS");
    }, 1000);
  };

  return (
    <div className="app-container">
      
      {/* 1. Main Video Area */}
      <div className="video-section" onClick={handleVideoClick} style={{cursor: 'crosshair'}}>
        <video ref={videoRef} autoPlay playsInline muted className="live-feed" />
        <div className="video-overlay-mesh"></div>
        
        <div style={{position: 'absolute', top: 20, left: 20, display: 'flex', gap: 20, pointerEvents: 'none'}}>
            <div style={{background: 'rgba(0,0,0,0.6)', padding: '5px 10px', fontSize: 12}}>
                <span style={{color: '#ff2a2a'}}>● LIVE</span>
            </div>
            {/* Emergency log */}
            {mode === 'EMERGENCY' && (
                <div style={{background: 'red', color:'white', padding: '5px 10px', fontSize: 12, fontWeight:'bold', animation: 'urgentPulse 0.5s infinite'}}>
                    ⚠ EMERGENCY MODE
                </div>
            )}
        </div>

        <div className="attitude-indicator" 
             style={{ transform: `rotate(${-Number(telemetry.roll)}deg)` }}>
             <div style={{
                 width: '100%', height: '200%', 
                 background: 'linear-gradient(to bottom, #3b82f6 50%, #854d0e 50%)',
                 position: 'absolute',
                 top: `${-50 + Number(telemetry.pitch) * 2}%`, 
                 transition: 'top 0.1s linear'
             }}></div>
             <div className="attitude-line"></div>
             <div className="attitude-center-dot"></div>
        </div>

        <div className="video-hud-stats">
            <div className="hud-stat-box">
                <span className="hud-label">ALTITUDE</span>
                <span className="hud-value">{Number(telemetry.alt).toFixed(1)}m</span>
            </div>
            <div className="hud-stat-box">
                <span className="hud-label">SPEED</span>
                <span className="hud-value">{Number(telemetry.speed).toFixed(1)}m/s</span>
            </div>
            <div className="hud-stat-box">
                <span className="hud-label">PITCH</span>
                <span className="hud-value">{Number(telemetry.pitch).toFixed(1)}°</span>
            </div>
        </div>

        {target.status !== 'IDLE' && (
            <div className={`target-box ${target.status === 'SEARCHING' ? 'searching' : 'locked'}`} 
                 style={{ top: target.y, left: target.x }}>
                <div className="target-label" style={{background: target.status === 'LOCKED' ? 'var(--cyan)' : 'yellow'}}>
                    {target.status === 'SEARCHING' ? 'SCAN...' : `ID: TRG_01`}
                </div>
                {target.status === 'LOCKED' && (
                    <>
                        <div className="target-corner tc-tl"></div><div className="target-corner tc-tr"></div>
                        <div className="target-corner tc-bl"></div><div className="target-corner tc-br"></div>
                    </>
                )}
            </div>
        )}
      </div>

      {/* 2. Right Sidebar */}
      <div className="right-sidebar">
        
        <div style={{height: '200px', border: '1px solid var(--border)', position: 'relative'}}>
             <div className="panel-header" style={{position:'absolute', zIndex:400, top:0, left:0, background:'rgba(0,0,0,0.7)', width:'100%'}}>
                 GPS POSITION
             </div>
             <MapContainer center={[32.0853, 34.7818]} zoom={13} zoomControl={false} scrollWheelZoom={true}>
                <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" />
                <CircleMarker center={[Number(telemetry.lat), Number(telemetry.lng)]} pathOptions={{ color: 'cyan' }} radius={6}>
                    <Popup>Drone</Popup>
                </CircleMarker>
            </MapContainer>
        </div>

        <div>
            <div className="panel-header">SYSTEM STATUS</div>
            <div style={{marginTop: 10, fontSize: 12}}>
                
                <div style={{display:'flex', justifyContent:'space-between', marginBottom:2}}>
                    <span>BATTERY</span><span style={{color: telemetry.bat > 30 ? 'cyan' : 'red'}}>{Math.floor(Number(telemetry.bat))}%</span>
                </div>
                <div style={{width:'100%', height:4, background:'#333', marginBottom: 10}}>
                    <div style={{width:`${telemetry.bat}%`, height:'100%', background: telemetry.bat > 30 ? 'cyan' : 'red'}}></div>
                </div>

                <div style={{display:'flex', justifyContent:'space-between', marginBottom:2}}>
                    <span>LINK QUALITY</span><span style={{color: telemetry.linkQuality > 50 ? '#00ff00' : 'orange'}}>{Math.floor(Number(telemetry.linkQuality))}%</span>
                </div>
                <div style={{width:'100%', height:4, background:'#333', marginBottom: 10}}>
                    <div style={{width:`${telemetry.linkQuality}%`, height:'100%', background: telemetry.linkQuality > 50 ? '#00ff00' : 'orange'}}></div>
                </div>

                <div style={{display:'flex', gap: 5}}>
                    <div style={{background: '#080a10', padding: 5, flex:1, textAlign:'center', border: '1px solid #333'}}>
                        <div style={{fontSize: 9, color:'#888', marginBottom:2}}>CORE TEMP</div>
                        <div style={{fontSize: 14, color: telemetry.coreTemp > 75 ? 'red' : '#e0e6ed', fontWeight: 'bold'}}>{telemetry.coreTemp.toFixed(1)}°C</div>
                    </div>
                    <div style={{background: '#080a10', padding: 5, flex:1, textAlign:'center', border: '1px solid #333'}}>
                        <div style={{fontSize: 9, color:'#888', marginBottom:2}}>ESC TEMP</div>
                        <div style={{fontSize: 14, color: telemetry.escTemp > 80 ? 'red' : '#e0e6ed', fontWeight: 'bold'}}>{telemetry.escTemp.toFixed(1)}°C</div>
                    </div>
                </div>

            </div>
        </div>

        <div className="commands-grid">
            <button className={`cmd-btn ${mode === 'MANUAL' ? 'primary' : 'active'}`} onClick={() => handleCommand('TOGGLE_MODE')}>
                {mode === 'MANUAL' ? 'MANUAL' : 'AUTO'}
            </button>
            <button className="cmd-btn" onClick={() => handleCommand('SYNC')}>⟳ SYNC</button>
            
            <button className="cmd-btn" onClick={() => handleCommand('RTH')} style={{color:'#ff9800'}}>⟲ RTH</button>
            
            <button 
                className={`cmd-btn ${mode === 'HOVER' ? 'active' : ''}`} 
                onClick={() => handleCommand('HOVER')}
                style={{display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px'}}
            >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>
                </svg>
                HOVER
            </button>
            
            <button className={`cmd-btn ${mode === 'FOLLOW' ? 'active' : ''}`} onClick={() => handleCommand('FOLLOW')}>⊕ FOLLOW</button>
            <button className="cmd-btn danger" onClick={() => handleCommand('LAND')}>↓ LAND</button>

            {}
            <button className={`cmd-btn abort-btn ${abortConfirm ? 'confirm-state' : ''}`} onClick={() => handleCommand('ABORT')}>{abortConfirm ? 'CONFIRM ABORT?' : 'ABORT MISSION'}</button>
        </div>

        <div className="log-stream">
            {logs.length === 0 && <div style={{opacity:0.5}}>No logs yet...</div>}
            {logs.map(log => (
                <div key={log.id} className="log-line">
                    <span style={{color: '#666'}}>[{log.time}]</span>
                    <span style={{
                        color: log.type === 'ERROR' ? '#ff2a2a' : 
                               log.type === 'WARN' ? '#ff9800' : 
                               log.type === 'SUCCESS' ? '#00ff00' : '#ccc'
                    }}>{log.message}</span>
                </div>
            ))}
        </div>

      </div>

      {/* 3. Footer */}
      <div className="bottom-bar">
        <div className="footer-stat"><div className="footer-label">LAT</div><div className="footer-value">{Number(telemetry.lat).toFixed(4)}</div></div>
        <div className="footer-stat"><div className="footer-label">LON</div><div className="footer-value">{Number(telemetry.lng).toFixed(4)}</div></div>
        <div className="footer-stat"><div className="footer-label">MODE</div><div className="footer-value" style={{color: 'cyan'}}>{mode}</div></div>
      </div>

    </div>
  )
}

export default App