const fs = require('node:fs/promises');
const path = require('node:path');
function isDesktopURL(value) {
  try { const url = new URL(value); return url.protocol === 'cure:' && url.hostname === 'desktop'; }
  catch { return false; }
}
// Chromium needs a document container. Visible layouts are React components;
// this generated bootstrap document does not require a standalone HTML file.
const documentShell = `<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self' data:; connect-src 'none'">
<link rel="stylesheet" href="/renderer.css"></head><body><div id="root"></div>
<script type="module" src="/renderer.js"></script></body></html>`;
function createProtocolHandler(root) {
  return async request => {
    const url = new URL(request.url);
    if (!isDesktopURL(request.url)) return new Response('Not found', { status: 404 });
    if (['/console', '/phone', '/hub'].includes(url.pathname)) {
      return new Response(documentShell, { headers: { 'Content-Type': 'text/html; charset=utf-8' } });
    }
    const asset = { '/renderer.js': 'text/javascript', '/renderer.css': 'text/css' }[url.pathname];
    if (!asset) return new Response('Not found', { status: 404 });
    try {
      return new Response(await fs.readFile(path.join(root, 'dist/renderer', url.pathname.slice(1))), {
        headers: { 'Content-Type': `${asset}; charset=utf-8` },
      });
    } catch { return new Response('Renderer build missing. Run npm.cmd run build.', { status: 503 }); }
  };
}
function apiPath(value, body) {
  const url = new URL(value, 'http://cure-api');
  if (url.origin !== 'http://cure-api' || !value.startsWith('/') || value.startsWith('//') || url.hash) throw new Error('Invalid API path');
  const allowed = body === undefined
    ? /^\/(health|state|memory|mind|agents\/dynamic(?:\/[a-zA-Z0-9_-]+)?|moments|why\/[a-zA-Z0-9_-]+)$/
    : /^\/(reset|door|bootstrap|ask|act|mind\/(recall|infer)|agents\/dynamic(?:\/[a-zA-Z0-9_-]+\/archive)?|device\/open|moment\/[a-zA-Z0-9_-]+)$/;
  if (!allowed.test(url.pathname)) throw new Error('Unknown CURE API route');
  return url.pathname + url.search;
}
module.exports = { isDesktopURL, createProtocolHandler, apiPath };
