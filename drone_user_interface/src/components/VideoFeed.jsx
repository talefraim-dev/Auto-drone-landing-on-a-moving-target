import React, { useState, useRef } from 'react';

const VideoFeed = ({ videoRef, telemetry, mode, target, onVideoClick }) => {
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPos, setStartPos] = useState({ x: 0, y: 0 });
  const [currentPos, setCurrentPos] = useState({ x: 0, y: 0 });
  const containerRef = useRef(null);

  const getRelativeCoords = (e) => {
    const videoElement = containerRef.current.querySelector('video');
    const targetElement = videoElement || containerRef.current;
    const rect = targetElement.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      width: rect.width,
      height: rect.height
    };
  };

  const handleMouseDown = (e) => {
    const { x, y } = getRelativeCoords(e);
    setIsDrawing(true);
    setStartPos({ x, y });
    setCurrentPos({ x, y });
  };

  const handleMouseMove = (e) => {
    if (!isDrawing) return;
    const { x, y } = getRelativeCoords(e);
    setCurrentPos({ x, y });
  };

  const handleMouseUp = () => {
    if (!isDrawing) return;
    setIsDrawing(false);

    const x = Math.min(startPos.x, currentPos.x);
    const y = Math.min(startPos.y, currentPos.y);
    const width = Math.abs(currentPos.x - startPos.x);
    const height = Math.abs(currentPos.y - startPos.y);

    if (width > 15 && height > 15) {
      const videoElement = containerRef.current.querySelector('video');
      const targetElement = videoElement || containerRef.current;
      const rect = targetElement.getBoundingClientRect();

      const bbox = {
        x, y, width, height,
        normX: x / rect.width,
        normY: y / rect.height,
        normW: width / rect.width,
        normH: height / rect.height
      };
      
      if (onVideoClick) onVideoClick(bbox);
    }
  };

  const drawBoxStyle = isDrawing ? {
    position: 'absolute',
    left: Math.min(startPos.x, currentPos.x),
    top: Math.min(startPos.y, currentPos.y),
    width: Math.abs(currentPos.x - startPos.x),
    height: Math.abs(currentPos.y - startPos.y),
    border: '2px dashed #00ff00',
    backgroundColor: 'rgba(0, 255, 0, 0.1)',
    zIndex: 10,
    pointerEvents: 'none'
  } : {};

  return (
    <div 
      className="video-section" 
      ref={containerRef}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={() => setIsDrawing(false)}
      style={{cursor: 'crosshair', position: 'relative'}}
    >
      <div ref={videoRef} className="live-feed" style={{ width: '100%', height: '100%', position: 'absolute', top: 0, left: 0, zIndex: 0 }}></div>
      <div className="video-overlay-mesh"></div>
      
      {isDrawing && <div style={drawBoxStyle}></div>}

      <div style={{position: 'absolute', top: 20, left: 20, display: 'flex', gap: 20, pointerEvents: 'none', zIndex: 20}}>
          <div style={{background: 'rgba(0,0,0,0.6)', padding: '5px 10px', fontSize: 12}}>
              <span style={{color: '#ff2a2a'}}>● LIVE</span>
          </div>
          {mode === 'EMERGENCY' && (
              <div style={{background: 'red', color:'white', padding: '5px 10px', fontSize: 12, fontWeight:'bold', animation: 'urgentPulse 0.5s infinite'}}>
                  ⚠ EMERGENCY MODE
              </div>
          )}
      </div>

      <div className="attitude-indicator" style={{ transform: `rotate(${-Number(telemetry.roll)}deg)`, zIndex: 20 }}>
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

      <div className="video-hud-stats" style={{zIndex: 20}}>
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
                style={{ 
                  top: target.y, 
                  left: target.x,
                  width: target.width ? `${target.width}px` : '100px',
                  height: target.height ? `${target.height}px` : '100px',
                  position: 'absolute',
                  zIndex: 20,
                  pointerEvents: 'none'
                }}>
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