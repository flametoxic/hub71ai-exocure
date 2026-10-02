const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('cureDesktop', {
  request: (path, body) => ipcRenderer.invoke('cure:request', path, body),
  openView: (view, query = '') => ipcRenderer.invoke('cure:open-view', view, query),
});
