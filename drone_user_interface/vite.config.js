import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: './', // קריטי ל-Electron כדי למצוא קבצים בנתיב יחסי
  server: {
    port: 5173, // מוודא שאנחנו רצים בפורט ש-main.js מצפה לו
  }
})