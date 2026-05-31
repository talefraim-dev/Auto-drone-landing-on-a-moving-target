import { useState, useEffect } from 'react';
import { Config, PixelStreaming } from '@epicgames-ps/lib-pixelstreamingfrontend-ue5.3';

export const usePixelStreaming = (videoRef, addLog) => {
  const [stream, setStream] = useState(null);

  useEffect(() => {
    let isMounted = true;

    addLog("Initializing AeroGuard System...", "SYS");
    addLog("Connecting to Unreal Engine Simulator...", "INFO");
    
    const config = new Config({
      useUrlParams: false,
      initialSettings: {
        ss: "ws://127.0.0.1:80", 
        AutoPlayVideo: true,
        AutoConnect: true,
        StartVideoMuted: true,
        HoveringMouse: true,
      }
    });

    if (videoRef.current) {
        const newStream = new PixelStreaming(config, {
            videoElementParent: videoRef.current 
        });

        
        
        newStream.addEventListener('playStream', () => {
            if (isMounted) {
                addLog("Simulator Feed Connected", "SUCCESS");
            }
        });

        setStream(newStream);

        return () => {
            isMounted = false;
            if (newStream) {
                newStream.disconnect();
            }
        };
    }
    
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); 

  return stream;
};