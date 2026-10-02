const { app, BrowserWindow, Menu, dialog, protocol, ipcMain, shell } = require('electron');
const path = require('node:path');
const { backendContract } = require('../backend-contract.mjs');
const { startBackend } = require('./backend.cjs');
const { isDesktopURL, createProtocolHandler, apiPath } = require('./desktop-protocol.cjs');
protocol.registerSchemesAsPrivileged([{ scheme: 'cure', privileges: { standard: true, secure: true, supportFetchAPI: true } }]);
const root = path.resolve(__dirname, '..');
let backend, starting;
let shutdownComplete = false, quitting = false, registered = false;
const webPreferences = { nodeIntegration: false, contextIsolation: true, sandbox: true, preload: path.join(__dirname, 'preload.cjs') };
function protectWindow(window) {
  window.webContents.on('before-input-event', (event, input) => {
    if (input.type === 'keyDown' && input.key === 'F11' && !input.isAutoRepeat) {
      event.preventDefault();
      window.setFullScreen(!window.isFullScreen());
    }
  });
  window.webContents.on('will-navigate', (event, url) => { if (!isDesktopURL(url)) event.preventDefault(); });
  window.webContents.setWindowOpenHandler(({url}) => {
    try { if (['https:', 'http:'].includes(new URL(url).protocol)) shell.openExternal(url).catch(() => {}); } catch {}
    return {action: 'deny'};
  });
}
function openView(view, query = '', { show = true, offscreen = false } = {}) {
  if (!['phone', 'hub'].includes(view)) throw new Error('Unknown desktop view');
  if (query && !query.startsWith('?')) throw new Error('Invalid view query');
  if (!query) {
    const windows = BrowserWindow.getAllWindows();
    const sources = [BrowserWindow.getFocusedWindow(), ...windows.filter(w => w.webContents.getURL().startsWith('cure://desktop/console')), ...windows];
    for (const source of sources) {
      if (!source || source.isDestroyed()) continue;
      const value = source.webContents.getURL();
      if (!isDesktopURL(value)) continue;
      const rid = new URL(value).searchParams.get('rid');
      if (rid) { query = '?rid=' + encodeURIComponent(rid); break; }
    }
  }
  const window = new BrowserWindow({
    width: view === 'phone' ? 470 : 1120, height: view === 'phone' ? 960 : 760,
    backgroundColor: '#0B1422', autoHideMenuBar: true, show, webPreferences: { ...webPreferences, offscreen },
  });
  protectWindow(window);
  window.loadURL(`cure://desktop/${view}${query}`).catch(error => dialog.showErrorBox('CURE', error.message));
  return window;
}
function registerDesktop() {
  if (registered) return;
  registered = true;
  protocol.handle('cure', createProtocolHandler(root));
  ipcMain.handle('cure:request', async (event, route, body) => {
    if (!isDesktopURL(event.senderFrame.url) || !backend) throw new Error('CURE backend unavailable');
    const endpoint = apiPath(route, body);
    const response = await fetch(`${backend.url}${endpoint}`, {
      method: body === undefined ? 'GET' : 'POST',
      ...(body === undefined ? {} : { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
      signal: AbortSignal.timeout(backendContract.requestTimeoutMs),
    });
    if (!response.ok) throw new Error((await response.text()).slice(0, 500));
    return response.json();
  });
  ipcMain.handle('cure:open-view', (event, view, query) => {
    if (!isDesktopURL(event.senderFrame.url)) throw new Error('Unknown desktop caller');
    openView(view, query);
  });
}
async function startApplication({ show = true, offscreen = false } = {}) {
  registerDesktop();
  starting = startBackend(root);
  backend = await starting;
  if (quitting) { await backend.stop(); throw new Error('CURE startup cancelled'); }
  const window = new BrowserWindow({
    width: 1800, height: 1000, minWidth: 1100, minHeight: 760, title: 'CURE',
    backgroundColor: '#0B1422', autoHideMenuBar: true, show: false, webPreferences: { ...webPreferences, offscreen },
  });
  protectWindow(window);
  const session = window.webContents.session;
  session.setPermissionRequestHandler((contents, permission, callback, details) => {
    callback(isDesktopURL(details.requestingUrl || contents.getURL()) && permission === 'media');
  });
  session.setPermissionCheckHandler((contents, permission, origin) => isDesktopURL(origin) && permission === 'media');
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    { label: 'CURE', submenu: [
      { label: 'Console', click: () => window.loadURL('cure://desktop/console') },
      { label: 'Phone', click: () => openView('phone') }, { label: 'Home hub', click: () => openView('hub') },
      { type: 'separator' }, { role: 'quit' },
    ] },
    { label: 'Edit', submenu: [{ role: 'undo' }, { role: 'redo' }, { type: 'separator' },
      { role: 'cut' }, { role: 'copy' }, { role: 'paste' }, { role: 'selectAll' }] },
    { label: 'View', submenu: [{ role: 'reload' }, { role: 'toggleDevTools' },
      { role: 'resetZoom' }, { role: 'zoomIn' }, { role: 'zoomOut' }, { role: 'togglefullscreen' }] },
  ]));
  await window.loadURL('cure://desktop/console');
  if (show) {
    window.setFullScreen(true);
    window.show();
  }
  return { window, backend };
}
app.on('before-quit', event => {
  quitting = true;
  if (shutdownComplete || (!backend && !starting)) return;
  event.preventDefault();
  Promise.resolve(backend || starting).then(value => value.stop()).catch(() => {}).finally(() => {
    shutdownComplete = true; app.quit();
  });
});
app.on('window-all-closed', () => app.quit());
// Electron's bootstrap may not assign require.main to the application's module.
const entry = path.resolve(process.argv[1] || '');
if (require.main === module || entry === root || entry === __filename) {
  app.whenReady().then(() => startApplication()).catch(error => {
    console.error(error);
    if (!quitting) dialog.showErrorBox('CURE could not start', error.message);
    app.quit();
  });
}
module.exports = { startApplication, openView };
