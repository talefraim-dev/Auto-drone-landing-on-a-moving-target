import { useState, useCallback } from 'react';

export const useLogger = () => {
  const [logs, setLogs] = useState([]);

  const addLog = useCallback((message, type = 'INFO') => {
    const time = new Date().toLocaleTimeString('en-GB', { hour12: false });
    setLogs(prev => [{ id: crypto.randomUUID(), time, message, type }, ...prev].slice(0, 50));
  }, []);

  return { logs, addLog };
};
