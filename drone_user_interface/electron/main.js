const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');


const isDev = !app.isPackaged;
function createWindow() {
  
  const mainWindow = new BrowserWindow({
    width: 1280,
    height: 720,
    backgroundColor: '#111', 
    webPreferences: {
      nodeIntegration: false, 
      contextIsolation: true, 
      preload: path.join(__dirname, 'preload.js'), 
    },
  });

  
  if (isDev) {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools(); 
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }
  startFakeTelemetry(mainWindow);
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

function startFakeTelemetry(window) {
  let alt = 10.0;
  let speed = 0.0;
  let battery = 100;
  let pitch = 0;
  let roll = 0;

  setInterval(() => {
    alt += (Math.random() - 0.5) * 0.2;
    speed = Math.abs(speed + (Math.random() - 0.5) * 0.5);
    battery -= 0.01; 
    pitch = (Math.random() - 0.5) * 10; 
    roll = (Math.random() - 0.5) * 10;

    const data = {
      alt: alt.toFixed(1),
      speed: speed.toFixed(1),
      bat: Math.floor(battery),
      attitude: { pitch, roll }
    };

    if (!window.isDestroyed()) {
      window.webContents.send('telemetry-data', data);
    }
  }, 100); 
}