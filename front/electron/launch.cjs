// Detach the desktop process so the launch command immediately returns to the shell.
const { spawn } = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');

const root = path.resolve(__dirname, '..');
const executable = path.join(root, 'node_modules/electron/dist', process.platform === 'win32' ? 'electron.exe' :
  process.platform === 'darwin' ? 'Electron.app/Contents/MacOS/Electron' : 'electron');
if (!fs.existsSync(executable)) {
  console.error('Electron is not installed. Run npm.cmd install, then npm.cmd exec install-electron.');
  process.exit(1);
}
const env = { ...process.env };
if (!fs.existsSync(path.join(root, 'dist/renderer/renderer.js'))) {
  console.error('Desktop interface is not built. Run npm.cmd run build first.');
  process.exit(1);
}
delete env.ELECTRON_RUN_AS_NODE;
const child = spawn(executable, [root], {
  cwd: root, env, detached: true, windowsHide: false,
  stdio: ['ignore', fs.openSync(path.join(root, 'desktop.log'), 'a'), fs.openSync(path.join(root, 'desktop.log'), 'a')],
});
child.on('error', error => { console.error(error.message); process.exitCode = 1; });
child.unref();
