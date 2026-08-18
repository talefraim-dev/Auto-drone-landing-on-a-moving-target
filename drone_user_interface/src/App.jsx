import React, { useRef } from 'react';

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
    const normX = rect.width > 0 ? pixelX / rect.width : 0;
    const normY = rect.height > 0 ? pixelY / rect.height : 0;
    handleTargetLock(pixelX, pixelY, normX, normY);
  };

  // Safe fallbacks for display
  const bat = telemetry.bat || 0;
  const link = telemetry.linkQuality || 0;
  const targetLockConfidence = target.confidence || (target.status === 'LOCKED' ? 98 : 0);

  return (
    <div className="dashboard-grid">
      
      {/* 1. TOP GLOBAL STATUS BAR */}
      <header className="top-bar">
        <div className="live-indicator"><div className="live-dot"></div>LIVE</div>
        
        <div className="top-bar-item">
          <span className="label">Mission Mode</span>
          <span className={`value ${mode === 'EMERGENCY' ? 'red' : 'cyan'}`}>{mode}</span>
        </div>
        
        <div className="top-bar-item">
          <span className="label">Control Mode</span>
          <span className="value">{mode === 'MANUAL' ? 'MANUAL' : 'AUTO'}</span>
        </div>
        
        <div className="top-bar-item">
          <span className="label">Target Status</span>
          <span className={`value ${target.status === 'LOCKED' ? 'green' : 'orange'}`}>
            {target.status !== 'IDLE' ? `LOCK ${targetLockConfidence}%` : 'NO TARGET'}
          </span>
        </div>

        <div className="top-bar-item">
          <span className="label">Link Quality</span>
          <span className={`value ${link > 40 ? 'green' : 'red'}`}>{Math.floor(link)}%</span>
        </div>

        <div className="top-bar-item">
          <span className="label">Battery</span>
          <span className={`value ${bat > 30 ? 'green' : 'red'}`}>{Math.floor(bat)}%</span>
        </div>
      </header>

      {/* 2. MAIN CAMERA AREA */}
      <VideoFeed 
        videoRef={videoRef} 
        telemetry={telemetry} 
        mode={mode} 
        target={target} 
        onVideoClick={onVideoClick} 
      />

      {/* 3. RIGHT SIDEBAR */}
      <aside className="sidebar">
        
        {/* Map Panel */}
        <div className="panel" style={{ padding: 0, height: '180px', position: 'relative' }}>
          <div className="panel-title" style={{ position: 'absolute', top: 8, left: 12, zIndex: 10, border: 'none', background: 'rgba(0,0,0,0.6)', padding: '2px 6px' }}>MAP</div>
          <Minimap lat={telemetry.lat} lng={telemetry.lng} isConnected={true} />
        </div>

        {/* Mission Status Panel */}
        <div className="panel">
          <div className="panel-title">MISSION STATUS</div>
          <div className="mission-grid">
            <div className="top-bar-item">
              <span className="label">Target State</span>
              <span className={`value ${target.status === 'LOCKED' ? 'green' : 'orange'}`}>{target.status}</span>
            </div>
            <div className="top-bar-item">
              <span className="label">Confidence</span>
              <span className="value">{targetLockConfidence}%</span>
            </div>
            <div className="top-bar-item">
              <span className="label">Range</span>
              <span className="value">{target.range ? target.range.toFixed(1) : '---'} m</span>
            </div>
            <div className="top-bar-item">
              <span className="label">XY Error</span>
              <span className="value orange">{target.xyError ? target.xyError.toFixed(2) : '---'} m</span>
            </div>
          </div>
        </div>

        {/* System Health Panel */}
        <div className="panel">
          <div className="panel-title">SYSTEM HEALTH</div>
          <div className="system-grid">
            <div className="top-bar-item">
              <span className="label">Core Temp</span>
              <span className={`value ${telemetry.coreTemp > 75 ? 'red' : 'cyan'}`}>
                {telemetry.coreTemp ? telemetry.coreTemp.toFixed(1) : '--'}°C
              </span>
            </div>
            <div className="top-bar-item">
              <span className="label">ESC Temp</span>
              <span className={`value ${telemetry.escTemp > 80 ? 'red' : 'cyan'}`}>
                {telemetry.escTemp ? telemetry.escTemp.toFixed(1) : '--'}°C
              </span>
            </div>
            <div className="top-bar-item" style={{ gridColumn: '1 / 3' }}>
              <span className="label" style={{ display: 'flex', justifyContent: 'space-between' }}>
                BATTERY <span style={{ color: bat > 30 ? 'var(--cyan)' : 'var(--red)' }}>{Math.floor(bat)}%</span>
              </span>
              <div className="progress-bar-bg">
                <div className="progress-bar-fill" style={{ width: `${bat}%`, background: bat > 30 ? 'var(--cyan)' : 'var(--red)' }}></div>
              </div>
            </div>
          </div>
        </div>

        {/* Flight Controls Panel */}
        <div className="panel">
          <div className="panel-title">FLIGHT CONTROLS</div>
          <div className="commands-grid">
            <button className={`btn ${mode !== 'MANUAL' ? 'active' : ''}`} onClick={() => handleCommand('TOGGLE_MODE')}>
              {mode === 'MANUAL' ? 'MANUAL' : 'AUTO'}
            </button>
            <button className="btn" onClick={() => handleCommand('SYNC')}>SYNC</button>
            <button className="btn danger" onClick={() => handleCommand('RTH')}>RTH</button>
            <button className={`btn ${mode === 'HOVER' ? 'active' : ''}`} onClick={() => handleCommand('HOVER')}>HOVER</button>
            <button className="btn full" onClick={() => handleCommand('LAND')} style={{ borderColor: 'var(--cyan)' }}>LAND</button>
            <button className={`btn btn-abort ${abortConfirm ? 'confirm-state' : ''}`} onClick={() => handleCommand('ABORT')}>
              {abortConfirm ? 'CONFIRM ABORT?' : 'ABORT MISSION'}
            </button>
          </div>
        </div>

        {/* Event Log Panel */}
        <div className="panel log-panel">
          <div className="panel-title">EVENT LOG</div>
          <div className="log-stream">
            {logs.length === 0 && <div style={{opacity:0.5}}>No events recorded...</div>}
            {logs.map(log => (
              <div key={log.id} className="log-line">
                  <span style={{color: 'var(--cyan)', marginRight: '6px'}}>[{log.time}]</span>
                  <span style={{ color: log.type === 'ERROR' ? 'var(--red)' : log.type === 'WARN' ? 'var(--orange)' : log.type === 'SUCCESS' ? 'var(--green)' : 'var(--text-main)' }}>
                      {log.message}
                  </span>
              </div>
            ))}
          </div>
        </div>

      </aside>

      {/* 4. BOTTOM STRIP */}
      <Footer telemetry={telemetry} mode={mode} />

    </div>
  );
}

export default App;