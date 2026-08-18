import React, { useRef } from 'react';
import './App.css';

// Hooks
import { usePixelStreaming } from './hooks/usePixelStreaming';
import { useTelemetry } from './hooks/useTelemetry';
import { useDroneLogic } from './hooks/useDroneLogic';
import { useLogger } from './hooks/useLogger';

// Components
import VideoFeed from './components/VideoFeed';
import Minimap from './components/Minimap';
import Footer from './components/Footer';

function App() {
  const videoRef = useRef(null);

  const { logs, addLog } = useLogger();
  const stream = usePixelStreaming(videoRef, addLog);

  const { telemetry } = useTelemetry(stream);
  const { mode, target, abortConfirm, handleCommand, handleTargetLock } = useDroneLogic(stream, addLog);

  const onVideoClick = (e) => {
    if (mode === 'LANDING' || mode === 'EMERGENCY') return;
    const rect = e.target.getBoundingClientRect();
    const pixelX = e.clientX - rect.left;
    const pixelY = e.clientY - rect.top;
    // Normalized (0-1) coordinates are resolution-independent, unlike raw pixel offsets,
    // so they stay meaningful to the simulator regardless of how the video element is scaled.
    const normX = rect.width > 0 ? pixelX / rect.width : 0;
    const normY = rect.height > 0 ? pixelY / rect.height : 0;
    handleTargetLock(pixelX, pixelY, normX, normY);
  };

  return (
    <div className="app-container">
      
      <VideoFeed 
        videoRef={videoRef} 
        telemetry={telemetry} 
        mode={mode} 
        target={target} 
        onVideoClick={onVideoClick} 
      />

      <div className="right-sidebar">
        
        {}
        <Minimap lat={telemetry.lat} lng={telemetry.lng} />

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

                <div style={{marginTop: 10}}>
                    <div style={{display:'flex', justifyContent:'space-between', marginBottom:2}}>
                        <span>LINK QUALITY</span><span style={{color: telemetry.linkQuality > 40 ? 'cyan' : 'red'}}>{Math.floor(telemetry.linkQuality)}%</span>
                    </div>
                    <div style={{width:'100%', height:4, background:'#333', marginBottom: 10}}>
                        <div style={{width:`${telemetry.linkQuality}%`, height:'100%', background: telemetry.linkQuality > 40 ? 'cyan' : 'red'}}></div>
                    </div>
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
            <button className={`cmd-btn ${mode === 'HOVER' ? 'active' : ''}`} onClick={() => handleCommand('HOVER')}>HOVER</button>
            <button className="cmd-btn danger" onClick={() => handleCommand('LAND')}>↓ LAND</button>
            <button className={`cmd-btn abort-btn ${abortConfirm ? 'confirm-state' : ''}`} onClick={() => handleCommand('ABORT')}>{abortConfirm ? 'CONFIRM ABORT?' : 'ABORT MISSION'}</button>
        </div>

        {/* Log Stream */}
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

      <Footer lat={telemetry.lat} lng={telemetry.lng} mode={mode} />

    </div>
  );
}

export default App;