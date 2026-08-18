import React from 'react';

const VideoFeed = ({ videoRef, telemetry, mode, target, onVideoClick }) => {
  return (
    <main className="camera-area" onClick={onVideoClick}>
      {/* Container element for WebRTC stream */}
      <div ref={videoRef} className="live-feed"></div>
      
      {/* Cinematic mesh overlay */}
      <div className="video-overlay-mesh"></div>
      
      {/* Emergency Status */}
      <div style={{ position: 'absolute', top: 30, left: 30, display: 'flex', gap: 15, pointerEvents: 'none', zIndex: 20 }}>
          {mode === 'EMERGENCY' && (
              <div style={{ background: 'var(--red)', color: 'white', padding: '6px 12px', borderRadius: '4px', fontWeight: 'bold', animation: 'urgentPulse 0.5s infinite' }}>
                  ⚠ EMERGENCY MODE ACTIVE
              </div>
          )}
      </div>

      {/* Artificial Horizon */}
      <div className="horizon-container" style={{ transform: `rotate(${-Number(telemetry.roll || 0)}deg)` }}>
           <div style={{
               width: '100%', height: '200%', 
               background: 'linear-gradient(to bottom, #1e3a8a 50%, #713f12 50%)',
               position: 'absolute',
               top: `${-50 + Number(telemetry.pitch || 0) * 2}%`, 
               transition: 'top 0.1s linear'
           }}></div>
           <div className="horizon-line"></div>
           <div className="horizon-center-dot"></div>
      </div>

      {/* Flight HUD Strip */}
      <div className="camera-hud-strip">
          <div className="hud-metric">
              <span className="label">ALT</span>
              <div><span className="value">{Number(telemetry.alt || 0).toFixed(1)}</span><span className="unit">m</span></div>
          </div>
          <div className="hud-metric">
              <span className="label">V/S</span>
              <div><span className="value">{Number(telemetry.vs || 0).toFixed(2)}</span><span className="unit">m/s</span></div>
          </div>
          <div className="hud-metric">
              <span className="label">XY SPD</span>
              <div><span className="value">{Number(telemetry.speed || 0).toFixed(2)}</span><span className="unit">m/s</span></div>
          </div>
          <div className="hud-metric">
              <span className="label">ROLL</span>
              <div><span className="value">{Number(telemetry.roll || 0).toFixed(1)}</span><span className="unit">°</span></div>
          </div>
          <div className="hud-metric">
              <span className="label">PITCH</span>
              <div><span className="value">{Number(telemetry.pitch || 0).toFixed(1)}</span><span className="unit">°</span></div>
          </div>
          <div className="hud-metric">
              <span className="label">YAW</span>
              <div><span className="value">{Number(telemetry.yaw || 0).toFixed(1)}</span><span className="unit">°</span></div>
          </div>
      </div>

      {/* Target Tracker Overlay */}
      {target.status !== 'IDLE' && (
          <div className={`target-bounding-box ${target.status === 'SEARCHING' ? 'searching' : 'locked'}`} 
               style={{ top: target.y, left: target.x }}>
              <div className="target-id-tag" style={{ background: target.status === 'LOCKED' ? 'var(--cyan)' : '#eab308' }}>
                  {target.status === 'SEARCHING' ? 'SCANNING...' : `TRG_LOCKED`}
              </div>
          </div>
      )}
    </main>
  );
};

export default VideoFeed;