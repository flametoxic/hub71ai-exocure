// Explicit opt-in live API/UI acceptance; uses an isolated resident store.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {app} = require('electron');
const store = fs.mkdtempSync(path.join(os.tmpdir(), 'cure-live-ui-'));
process.env.ARRIVAL_CORE_DIR = store;
process.env.CURE_LANGUAGE_MODE = 'openai';
delete process.env.CURE_BACKEND_URL;
const {startApplication} = require('./main.cjs');

app.whenReady().then(async () => {
  let backend;
  const report = {live:true, passed:false};
  const artifacts = path.resolve(__dirname, '../../artifacts');
  fs.mkdirSync(artifacts, {recursive:true});
  try {
    const started = await startApplication({show:false});
    backend = started.backend;
    const window = started.window;
    const request = (route, body) => window.webContents.executeJavaScript(
      `window.cureDesktop.request(${JSON.stringify(route)}, ${JSON.stringify(body) ?? 'undefined'})`);
    const waitUntil = async (check, milliseconds = 15000) => {
      const deadline = Date.now() + milliseconds;
      while (Date.now() < deadline) {
        if (await check()) return;
        await new Promise(resolve => setTimeout(resolve,250));
      }
      throw new Error('Timed out waiting for live renderer state');
    };
    await waitUntil(() => window.webContents.executeJavaScript("!!document.querySelector('.profile-switch')"));
    await window.webContents.executeJavaScript("document.querySelectorAll('.profile-switch button')[1].click()");
    await waitUntil(() => window.webContents.executeJavaScript("!!document.querySelector('.phone-view #chatbox textarea')"));
    const task = 'I need to move my cat from Cairo to Abu Dhabi. Figure out what I need.';
    await window.webContents.executeJavaScript(`
      const input = document.querySelector('.phone-view #chatbox textarea');
      input.value = ${JSON.stringify(task)};
      input.dispatchEvent(new Event('input',{bubbles:true}));
      input.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
    `);
    let agent;
    await waitUntil(async () => {
      const result = await request('/agents/dynamic?rid=leila');
      agent = result.agents[0];
      return !!agent;
    }, 90000);
    await waitUntil(() => window.webContents.executeJavaScript("document.querySelector('.phone-view #screen').innerText.includes('NEED_DETECTED')"));
    console.log('LIVE UI: agent lifecycle visible');
    await waitUntil(async () => {
      agent = await request('/agents/dynamic/' + agent.agent_id + '?rid=leila');
      return ['COMPLETED','WAITING_USER','FAILED'].includes(agent.status);
    }, 300000);
    report.agent = agent;
    assert.ok(['COMPLETED','WAITING_USER'].includes(agent.status), agent.error);
    await waitUntil(() => window.webContents.executeJavaScript("document.querySelector('.phone-view #screen').innerText.includes('Official source')"));
    await window.webContents.executeJavaScript("document.querySelectorAll('.phone-view #tabs button')[3].click()");
    await waitUntil(() => window.webContents.executeJavaScript("document.querySelector('.phone-view #screen').innerText.toLowerCase().includes('associative memory')"));
    const mind = await request('/mind?rid=leila');
    assert.ok(mind.memories.some(m => m.provenance === 'agent' && m.status === 'pending'));
    await window.webContents.executeJavaScript("document.querySelector('.sidebar-toggle').click()");
    await waitUntil(() => window.webContents.executeJavaScript("!document.querySelector('.console-view').classList.contains('right-collapsed') && document.querySelector('#hood').innerText.includes('SIGNED')"));
    report.phone_mind = true;
    report.console_lifecycle = true;
    await window.webContents.executeJavaScript("new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))");
    await new Promise(resolve => setTimeout(resolve, 250));
    const screenshot = await window.webContents.capturePage();
    fs.writeFileSync(path.join(artifacts,'live-mind-console.png'),screenshot.toPNG());
    report.passed = true;
    console.log('PASS: live unknown-task agent, Phone result/sources, Mind pending memory, Console lifecycle');
  } catch (error) {
    report.error = error.message;
    console.error(error);
    process.exitCode = 1;
  } finally {
    fs.writeFileSync(path.join(artifacts,'live-ui-verification.json'),JSON.stringify(report,null,2));
    await backend?.stop();
    const verifiedStore = fs.realpathSync(store);
    assert.ok(verifiedStore.startsWith(fs.realpathSync(os.tmpdir()) + path.sep));
    fs.rmSync(verifiedStore,{recursive:true,force:true});
    app.exit(process.exitCode || 0);
  }
});
