import React from 'react';

const Footer = ({ telemetry, mode }) => {
  return (
    <footer className="bottom-strip">
      <div className="position-group">
        <div className="top-bar-item">
          <span className="label">X POS</span>
          <span className="value">{Number(telemetry.x || 0).toFixed(2)} m</span>
        </div>
        <div className="top-bar-item">
          <span className="label">Y POS</span>
          <span className="value">{Number(telemetry.y || 0).toFixed(2)} m</span>
        </div>
        <div className="top-bar-item">
          <span className="label">Z / ALT</span>
          <span className="value">{Number(telemetry.z || 0).toFixed(2)} m</span>
        </div>
        <div className="top-bar-item" style={{ marginLeft: '24px' }}>
          <span className="label">System Mode</span>
          <span className="value cyan">{mode}</span>
        </div>
      </div>

      <div className="top-bar-item" style={{ alignItems: 'flex-end' }}>
        <span className="label">GPS Coordinates (Secondary)</span>
        <span className="value" style={{ fontSize: '14px', color: 'var(--text-muted)' }}>
          LAT {Number(telemetry.lat || 0).toFixed(4)} &nbsp; LON {Number(telemetry.lng || 0).toFixed(4)}
        </span>
      </div>
    </footer>
  );
};

export default Footer;