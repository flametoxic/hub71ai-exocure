import { defineConfig } from 'vite';
import desktopConfig from './vite.config.mjs';
const shell = `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self' data:; connect-src 'self'">
<title>CURE</title><link rel="stylesheet" href="/renderer.css"></head>
<body><div id="root"></div><script type="module" src="/renderer.js"></script></body></html>`;
export default defineConfig({
  ...desktopConfig,
  build: {...desktopConfig.build, outDir: 'dist/web'},
  plugins: [{name: 'web-pages', generateBundle() {
    for (const fileName of ['index.html', 'console.html', 'phone.html', 'app.html', 'hub.html']) {
      this.emitFile({type: 'asset', fileName, source: shell});
    }
  }}],
});
