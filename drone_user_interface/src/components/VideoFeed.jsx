import React from 'react';

const VideoFeed = ({ videoRef, telemetry, mode, target, onVideoClick }) => {
  return (
    <main className="camera-area" onClick={onVideoClick}>
      {/* WebRTC Video Container */}
      <div ref={videoRef} className="video-feed" style={{ background: '#000' }}></div>
      
      {/* Target Tracker Overlay */}
      {target.status !== 'IDLE' && (
        <div className="target-bounding-box" style={{ top: target.y, left: target.x }}>
          <div className="target-id-tag">
             {target.status === 'SEARCHING' ? 'SCAN...' : 'ID: TRG_01'}
          </div>
        </div>
      )}

      {/* Artificial Horizon */}
      <div className="horizon-container" style={{ transform: `rotate(${-Number(telemetry.roll)}deg)` }}>
           <div style={{
               width: '100%', height: '200%', 
               background: 'linear-gradient(to bottom, #1e3a8a 50%, #713f12 50%)',
               position: 'absolute',
               top: `${-50 + Number(telemetry.pitch) * 2}%`, 
               transition: 'top 0.1s linear'
           }}></div>
           <div className="horizon-line"></div>
      </div>

      {/* HUD Strip */}
      <div className="camera-hud-strip">
          <div className="hud-metric">
              <span className="label">ALT</span>
              <div><span className="value">{Number(telemetry.alt).toFixed(1)}</span><span className="unit">m</span></div>
          </div>
          <div className="hud-metric">
              <span className="label">V/S</span>
              <div><span className="value">{Number(telemetry.vs).toFixed(2)}</span><span className="unit">m/s</span></div>
          </div>
          <div className="hud-metric">
              <span className="label">XY SPEED</span>
              <div><span className="value">{Number(telemetry.speed).toFixed(2)}</span><span className="unit">m/s</span></div>
          </div>
          <div className="hud-metric">
              <span className="label">ROLL</span>
              <div><span className="value">{Number(telemetry.roll).toFixed(1)}</span><span className="unit">°</span></div>
          </div>
          <div className="hud-metric">
              <span className="label">PITCH</span>
              <div><span className="value">{Number(telemetry.pitch).toFixed(1)}</span><span className="unit">°</span></div>
          </div>
          <div className="hud-metric">
              <span className="label">YAW</span>
              <div><span className="value">{Number(telemetry.yaw).toFixed(1)}</span><span className="unit">°</span></div>
          </div>
      </div>
    </main>
  );
};

export default VideoFeed;