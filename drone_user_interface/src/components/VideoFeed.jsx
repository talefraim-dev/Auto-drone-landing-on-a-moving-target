import React from 'react';

const VideoFeed = ({ videoRef, telemetry, mode, target, onVideoClick }) => {
  return (
    <div className="video-section" onClick={onVideoClick} style={{cursor: 'crosshair'}}>
      {/* Container element for WebRTC stream */}
      <div ref={videoRef} className="live-feed" style={{ width: '100%', height: '100%', position: 'absolute', top: 0, left: 0, zIndex: 0 }}></div>
      <div className="video-overlay-mesh"></div>
      
      <div style={{position: 'absolute', top: 20, left: 20, display: 'flex', gap: 20, pointerEvents: 'none'}}>
          <div style={{background: 'rgba(0,0,0,0.6)', padding: '5px 10px', fontSize: 12}}>
              <span style={{color: '#ff2a2a'}}>● LIVE</span>
          </div>
          {/* Emergency overlay */}
          {mode === 'EMERGENCY' && (
              <div style={{background: 'red', color:'white', padding: '5px 10px', fontSize: 12, fontWeight:'bold', animation: 'urgentPulse 0.5s infinite'}}>
                  ⚠ EMERGENCY MODE
              </div>
          )}
      </div>

      <div className="attitude-indicator" style={{ transform: `rotate(${-Number(telemetry.roll)}deg)` }}>
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
          <div className="hud-stat-box">
              <span className="hud-label">ROL</span>
              <span className="hud-value">{Number(telemetry.roll).toFixed(1)}°</span>
          </div>
          <div className="hud-stat-box">
              <span className="hud-label">YAW</span>
              <span className="hud-value">{Number(telemetry.yaw).toFixed(1)}°</span>
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
  );
};

export default VideoFeed;