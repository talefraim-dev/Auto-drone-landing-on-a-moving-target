import React, { useState } from 'react';
import VideoFeed from './components/VideoFeed';
import Footer from './components/Footer';
import Minimap from './components/Minimap';
// import './App.css'; // אם יש לך שם דברים, אבל רצוי להשתמש ב index.css בלבד לעיצוב הגלובלי

function App() {
  // נתוני דמה לדוגמה (כאן יכנסו ה-Hooks וה-State האמיתיים שלך מהשרת)
  const [telemetry, setTelemetry] = useState({
    alt: 12.4, vs: -0.42, speed: 0.31, roll: 1.2, pitch: -0.8, yaw: 178.4, 
    lat: 32.0402, lng: 34.7583, x: 0.02, y: -0.01, z: -2.81
  });
  const [target, setTarget] = useState({ status: 'LOCKED', confidence: 98, range: 3.2, xyError: 0.18, x: '50%', y: '50%' });
  const [mode, setMode] = useState('LANDING');
  const [phase, setPhase] = useState('DESCEND');

  return (
    <div className="dashboard-grid">
      
      {/* Top Global Status Bar */}
      <header className="top-bar">
        <div className="live-indicator"><div className="live-dot"></div>LIVE</div>
        <div className="top-bar-item"><span className="label">Mission Mode</span><span className="value cyan">{mode}</span></div>
        <div className="top-bar-item"><span className="label">Control Mode</span><span className="value">AUTO</span></div>
        <div className="top-bar-item"><span className="label">Target Status</span><span className="value green">LOCK {target.confidence}%</span></div>
        <div className="top-bar-item"><span className="label">Link Quality</span><span className="value">81%</span></div>
        <div className="top-bar-item"><span className="label">Battery</span><span className="value">97%</span></div>
      </header>

      {/* Main Camera Area */}
      <VideoFeed telemetry={telemetry} target={target} mode={mode} />

      {/* Right Sidebar */}
      <aside className="sidebar">
        
        {/* Map Panel */}
        <div className="panel" style={{ padding: 0, height: '200px' }}>
          <div className="panel-title" style={{ position: 'absolute', top: 8, left: 12, zIndex: 10, border: 'none' }}>MAP</div>
          <Minimap lat={telemetry.lat} lng={telemetry.lng} isConnected={true} />
        </div>

        {/* Mission Status */}
        <div className="panel">
          <div className="panel-title">MISSION STATUS</div>
          <div className="mission-grid">
            <div className="top-bar-item"><span className="label">Target</span><span className="value green">{target.status}</span></div>
            <div className="top-bar-item"><span className="label">Confidence</span><span className="value">{target.confidence}%</span></div>
            <div className="top-bar-item"><span className="label">Range</span><span className="value">{target.range} m</span></div>
            <div className="top-bar-item"><span className="label">XY Error</span><span className="value orange">{target.xyError} m</span></div>
            <div className="top-bar-item" style={{ gridColumn: '1 / 3' }}><span className="label">Current Phase</span><span className="value cyan">{phase}</span></div>
          </div>
        </div>

        {/* System & Controls */}
        <div className="panel">
          <div className="panel-title">SYSTEM</div>
          <div className="system-grid">
            <div className="top-bar-item"><span className="label">Core Temp</span><span className="value">76.1°C</span></div>
            <div className="top-bar-item"><span className="label">ESC Temp</span><span className="value">35.0°C</span></div>
          </div>
        </div>

        <div className="panel">
          <div className="panel-title">FLIGHT CONTROLS</div>
          <div className="controls-grid">
            <button className="btn active">AUTO</button>
            <button className="btn">SYNC</button>
            <button className="btn">RTH</button>
            <button className="btn">HOVER</button>
            <button className="btn full">LAND</button>
            <button className="btn btn-abort">ABORT MISSION</button>
          </div>
        </div>
      </aside>

      {/* Bottom Mission Strip */}
      <Footer telemetry={telemetry} phase={phase} />
      
    </div>
  );
}

export default App;