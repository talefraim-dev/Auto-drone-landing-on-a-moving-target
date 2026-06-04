import React from 'react';

const Footer = ({ lat, lng, mode }) => {
  return (
    <div className="bottom-bar">
      <div className="footer-stat"><div className="footer-label">LAT</div><div className="footer-value">{Number(lat).toFixed(4)}</div></div>
      <div className="footer-stat"><div className="footer-label">LON</div><div className="footer-value">{Number(lng).toFixed(4)}</div></div>
      <div className="footer-stat"><div className="footer-label">MODE</div><div className="footer-value" style={{color: 'cyan'}}>{mode}</div></div>
    </div>
  );
};

export default Footer;