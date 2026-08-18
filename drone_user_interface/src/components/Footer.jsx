import React from 'react';

const Footer = ({ telemetry, phase }) => {
  return (
    <footer className="bottom-strip">
      <div className="position-group">
        <div className="top-bar-item">
          <span className="label">X</span>
          <span className="value">{Number(telemetry.x).toFixed(2)} m</span>
        </div>
        <div className="top-bar-item">
          <span className="label">Y</span>
          <span className="value">{Number(telemetry.y).toFixed(2)} m</span>
        </div>
        <div className="top-bar-item">
          <span className="label">Z / ALT</span>
          <span className="value">{Number(telemetry.z).toFixed(2)} m</span>
        </div>
        <div className="top-bar-item" style={{ marginLeft: '24px' }}>
          <span className="label">Landing Phase</span>
          <span className="value cyan">{phase}</span>
        </div>
      </div>

      <div className="top-bar-item" style={{ alignItems: 'flex-end' }}>
        <span className="label">GPS Coordinates (SECONDARY)</span>
        <span className="value" style={{ fontSize: '14px', color: 'var(--text-muted)' }}>
          LAT {Number(telemetry.lat).toFixed(4)} &nbsp; LON {Number(telemetry.lng).toFixed(4)}
        </span>
      </div>
    </footer>
  );
};

export default Footer;