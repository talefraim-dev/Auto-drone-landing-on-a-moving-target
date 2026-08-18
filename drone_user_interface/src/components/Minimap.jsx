import React from 'react';
import { MapContainer, ImageOverlay, Popup, CircleMarker } from 'react-leaflet';
import 'leaflet/dist/leaflet.css'; 

const Minimap = ({ lat, lng }) => {
  const mapBounds = [[32.0000, 34.7000], [32.0800, 34.81635]];

  const safeLat = (lat === undefined || isNaN(lat)) ? 32.0853 : Number(lat);
  const safeLng = (lng === undefined || isNaN(lng)) ? 34.7818 : Number(lng);

  return (
    <div style={{height: '200px', width: '100%', border: '1px solid var(--border)', position: 'relative'}}>
        <div className="panel-header" style={{position:'absolute', zIndex:400, top:0, left:0, background:'rgba(0,0,0,0.7)', width:'100%'}}>
            SIMULATOR MAP
        </div>
        
        {}
        <MapContainer 
            center={[32.0853, 34.7818]} 
            zoom={15} 
            minZoom={12} 
            maxZoom={16}
            maxBounds={mapBounds}
            zoomControl={false} 
            scrollWheelZoom={true} 
            style={{height: '100%', width: '100%', backgroundColor: '#0a0a0a'}}
        >
            <ImageOverlay 
                url="src/assets/mini_map_v2.png" 
                bounds={mapBounds} 
            />
            
            <CircleMarker center={[safeLat, safeLng]} pathOptions={{ color: 'cyan', fillColor: 'cyan', fillOpacity: 0.5 }} radius={6}>
                <Popup>UAV Position</Popup>
            </CircleMarker>
        </MapContainer>
    </div>
  );
};

export default Minimap;