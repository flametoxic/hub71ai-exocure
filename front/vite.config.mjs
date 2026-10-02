import { defineConfig } from 'vite';
import { resolve } from 'node:path';
export default defineConfig({
  define: { 'process.env.NODE_ENV': JSON.stringify('production') },
  build: {
    outDir: 'dist/renderer', emptyOutDir: true,
    lib: { entry: resolve('electron/renderer/main.jsx'), formats: ['es'], fileName: () => 'renderer.js', cssFileName: 'renderer' },
    rolldownOptions: { output: { codeSplitting: false } },
  },
});
