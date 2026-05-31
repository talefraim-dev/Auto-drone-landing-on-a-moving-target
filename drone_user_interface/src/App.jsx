import React, { useRef, useEffect } from 'react';
import { MapContainer, TileLayer, Popup, CircleMarker } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import './App.css';

// Hooks
import { useTelemetry } from './hooks/useTelemetry';
import { useDroneLogic } from './hooks/useDroneLogic';
import { usePixelStreaming } from './hooks/usePixelStreaming';

// Components
import VideoFeed from './components/VideoFeed';
import Footer from './components/Footer';

function App() {
  const videoRef = useRef(null); 
  
  // Custom Hooks initialization
  const { telemetry } = useTelemetry();
  const { mode, target, setTarget, logs, abortConfirm, addLog, handleCommand } = useDroneLogic();
  const stream = usePixelStreaming(videoRef, addLog);

  // Target lock simulation
  useEffect(() => {
    const interval = setInterval(() => {
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
  }, [setTarget]);

  // Video interaction handler
  const handleVideoClick = (e) => {
    if (mode === 'LANDING' || mode === 'EMERGENCY') return;

    const rect = e.target.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setTarget({ x, y, status: 'SEARCHING' });
    addLog(`Acquiring target...`, "INFO");

    // Optional: Send coordinates back to Unreal Engine
    if (stream) {
      stream.emitUIInteraction({ Command: "SetTarget", TargetX: x, TargetY: y });
    }

    setTimeout(() => {
        setTarget(prev => ({ ...prev, status: 'LOCKED' }));
        addLog("Target Locked. Confidence: 98%", "SUCCESS");
    }, 1000);
  };

  return (
    <div className="app-container">
      
      {/* 1. Main Video Area */}
      <VideoFeed 
        videoRef={videoRef} 
        telemetry={telemetry} 
        mode={mode} 
        target={target} 
        onVideoClick={handleVideoClick} 
      />

      {/* 2. Right Sidebar */}
      <div className="right-sidebar">
        
        {/* Map Section */}
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

        {/* System Status Section */}
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

        {/* Commands Grid Section */}
        <div className="commands-grid">
            <button className={`cmd-btn ${mode === 'MANUAL' ? 'primary' : 'active'}`} onClick={() => handleCommand('TOGGLE_MODE')}>
                {mode === 'MANUAL' ? 'MANUAL' : 'AUTO'}
            </button>
            <button className="cmd-btn" onClick={() => handleCommand('SYNC')}>⟳ SYNC</button>
            <button className="cmd-btn" onClick={() => handleCommand('RTH')} style={{color:'#ff9800'}}>⟲ RTH</button>
            <button className={`cmd-btn ${mode === 'HOVER' ? 'active' : ''}`} onClick={() => handleCommand('HOVER')} style={{display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px'}}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>
                </svg> HOVER
            </button>
            <button className={`cmd-btn ${mode === 'FOLLOW' ? 'active' : ''}`} onClick={() => handleCommand('FOLLOW')}>⊕ FOLLOW</button>
            <button className="cmd-btn danger" onClick={() => handleCommand('LAND')}>↓ LAND</button>
            <button className={`cmd-btn abort-btn ${abortConfirm ? 'confirm-state' : ''}`} onClick={() => handleCommand('ABORT')}>{abortConfirm ? 'CONFIRM ABORT?' : 'ABORT MISSION'}</button>
        </div>

        {/* Log Stream Section */}
        <div className="log-stream">
            {logs.length === 0 && <div style={{opacity:0.5}}>No logs yet...</div>}
            {logs.map(log => (
                <div key={log.id} className="log-line">
                    <span style={{color: '#666'}}>[{log.time}]</span>
                    <span style={{ color: log.type === 'ERROR' ? '#ff2a2a' : log.type === 'WARN' ? '#ff9800' : log.type === 'SUCCESS' ? '#00ff00' : '#ccc' }}>
                        {log.message}
                    </span>
                </div>
            ))}
        </div>

      </div>

      {/* 3. Footer */}
      <Footer lat={telemetry.lat} lng={telemetry.lng} mode={mode} />

    </div>
  );
}

export default App;