const { spawn, execFile } = require('node:child_process');
const fs = require('node:fs');
const net = require('node:net');
const path = require('node:path');

function backendURL(value) {
  const url = new URL(value);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password ||
      url.pathname !== '/' || url.search || url.hash) {
    throw new Error('CURE_BACKEND_URL must be an HTTP(S) origin, e.g. http://127.0.0.1:8000');
  }
  return url.origin;
}

function pythonCommand(root, env = process.env) {
  if (env.CURE_PYTHON) return env.CURE_PYTHON;
  const executable = process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python';
  for (const directory of [root, path.resolve(root, '../front'), path.resolve(root, '..')]) {
    const local = path.join(directory, '.venv', executable);
    if (fs.existsSync(local)) return local;
  }
  return 'python';
}

function backendRoot(frontRoot, env = process.env) {
  const directory = env.CURE_BACKEND_DIR
    ? path.resolve(frontRoot, env.CURE_BACKEND_DIR)
    : path.resolve(frontRoot, '../backend');
  if (!fs.existsSync(path.join(directory, 'tools/run_demo.py'))) {
    throw new Error(`Backend launcher missing: ${directory}. Set CURE_BACKEND_DIR to the backend folder.`);
  }
  return directory;
}

async function freePort(requested = 0) {
  const server = net.createServer();
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(requested, '127.0.0.1', resolve);
  });
  const port = server.address().port;
  await new Promise(resolve => server.close(resolve));
  return port;
}

async function waitForBackend(url, failure, timeout = 30000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    const error = failure();
    if (error) throw error;
    try {
      const health = await fetch(`${url}/health`, { signal: AbortSignal.timeout(1000) });
      if (!health.ok || (await health.json()).ready !== true) throw new Error('Backend health check failed');
      const response = await fetch(`${url}/twin/state`, { signal: AbortSignal.timeout(1000) });
      if (response.ok) {
        const state = await response.json();
        if (typeof state.clock === 'string' && state.city) return;
      }
    } catch { /* Retry while the Python process initializes. */ }
    await new Promise(resolve => setTimeout(resolve, 200));
  }
  throw new Error(`CURE backend did not become ready at ${url} within ${timeout / 1000}s`);
}

async function startBackend(root, env = process.env) {
  if (env.CURE_BACKEND_URL) {
    const url = backendURL(env.CURE_BACKEND_URL);
    await waitForBackend(url, () => null);
    return { url, owned: false, stop: async () => {} };
  }
  root = backendRoot(root, env);
  const requested = env.CURE_PORT || env.PORT;
  if (requested && (!/^\d+$/.test(requested) || +requested < 1 || +requested > 65535)) {
    throw new Error('CURE_PORT must be between 1 and 65535');
  }
  const port = await freePort(requested ? Number(requested) : 0);
  const child = spawn(pythonCommand(root, env), [path.join(root, 'tools/run_demo.py')], {
    cwd: root, windowsHide: true,
    env: { ...env, PORT: String(port), HOST: '127.0.0.1', PYTHONUTF8: '1' },
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: process.platform !== 'win32',
  });
  let diagnostic = '';
  let failure = null;
  for (const stream of [child.stdout, child.stderr]) {
    stream.on('data', chunk => { diagnostic = (diagnostic + chunk.toString()).slice(-6000); });
  }
  child.on('error', error => { failure = error; });
  child.on('exit', (code, signal) => {
    failure = new Error(`Python backend stopped (${code ?? signal}).\n${diagnostic}`);
  });
  let stopping;
  function stop() {
    if (stopping) return stopping;
    stopping = new Promise(resolve => {
      if (!child.pid) return resolve();
      if (process.platform === 'win32') {
        execFile('taskkill', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true }, () => resolve());
      } else {
        try { process.kill(-child.pid, 'SIGTERM'); } catch { /* Already stopped. */ }
        resolve();
      }
    });
    return stopping;
  }
  try {
    const url = `http://127.0.0.1:${port}`;
    await waitForBackend(url, () => failure);
    return { url, owned: true, stop, child };
  } catch (error) {
    await stop();
    throw new Error(`${error.message}\n${diagnostic}\nInstall Python dependencies from ${path.join(root, 'requirements.txt')}`);
  }
}

module.exports = { backendURL, backendRoot, pythonCommand, waitForBackend, startBackend };
