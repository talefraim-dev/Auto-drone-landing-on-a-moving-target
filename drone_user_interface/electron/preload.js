const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  //React -> Telemetry -> drone 
  onTelemetry: (callback) => ipcRenderer.on('telemetry-data', callback),
  // commands to drone
  sendCommand: (command) => ipcRenderer.send('drone-command', command)
});