import React, { useState } from 'react';
import { MapContainer, ImageOverlay, Popup, CircleMarker, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import miniMapImage from '../assets/mini_map_v2.png';

const DynamicZoomMarker = ({ lat, lng, markerColor, isConnected }) => {
    const [currentZoom, setCurrentZoom] = useState(14);

    useMapEvents({
        zoomend: (e) => {
            setCurrentZoom(e.target.getZoom());
        }
    });

    const calculateRadius = () => {
        if (currentZoom >= 16) return 12;
        if (currentZoom >= 14) return 8;
        if (currentZoom >= 12) return 5;
        return 4;
    };

    return (
        <CircleMarker 
            center={[lat, lng]} 
            pathOptions={{ 
                color: markerColor, 
                fillColor: markerColor, 
                fillOpacity: 0.6 
            }} 
            radius={calculateRadius()}
        >
            <Popup>
                {isConnected ? `UAV Position: ${lat.toFixed(4)}, ${lng.toFixed(4)}` : 'No Connection'}
            </Popup>
        </CircleMarker>
    );
};

const Minimap = ({ lat, lng, isConnected }) => {
  const mapBounds = [[32.0000, 34.7000], [32.0800, 34.81635]];
  const defaultLat = 32.0400; 
  const defaultLng = 34.7580;

  const safeLat = (isConnected && lat !== undefined && !isNaN(lat)) ? Number(lat) : defaultLat;
  const safeLng = (isConnected && lng !== undefined && !isNaN(lng)) ? Number(lng) : defaultLng;
  const markerColor = isConnected ? '#00f0ff' : '#ff3333';

  return (
    <div style={{height: '200px', width: '100%', border: '1px solid var(--border)', position: 'relative'}}>
        <div className="panel-header" style={{position:'absolute', zIndex:400, top:0, left:0, background:'rgba(0,0,0,0.7)', width:'100%', color: '#fff', padding: '5px'}}>
            SIMULATOR MAP {isConnected ? '🔴 LIVE' : '⚪ DISCONNECTED'}
        </div>
        
        <MapContainer 
            center={[defaultLat, defaultLng]} 
            zoom={14} 
            minZoom={12} 
            maxZoom={16}
            maxBounds={mapBounds}
            zoomControl={false} 
            scrollWheelZoom={true} 
            style={{height: '100%', width: '100%', backgroundColor: '#0a0a0a'}}
        >
            <ImageOverlay 
                url={miniMapImage}
                bounds={mapBounds} 
                zIndex={1}
            />
            
            <DynamicZoomMarker 
                lat={safeLat} 
                lng={safeLng} 
                markerColor={markerColor} 
                isConnected={isConnected} 
            />

        </MapContainer>
    </div>
  );
};

export default Minimap;