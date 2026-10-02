const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { app } = require('electron');
const store = fs.mkdtempSync(path.join(os.tmpdir(), 'cure-integration-'));
process.env.ARRIVAL_CORE_DIR = store;
process.env.CURE_LANGUAGE_MODE = 'local';
delete process.env.CURE_BACKEND_URL;
const { startApplication, openView } = require('./main.cjs');

app.whenReady().then(async () => {
  let backend;
  try {
    const started = await startApplication({ show: false });
    backend = started.backend;
    const window = started.window;
    const errors = [];
    window.webContents.on('console-message', (_event, level, message) => {
      if (level >= 3) errors.push(message);
    });
    const request = (route, body) => window.webContents.executeJavaScript(
      `window.cureDesktop.request(${JSON.stringify(route)}, ${JSON.stringify(body) ?? 'undefined'})`);
    const waitUntil = async check => {
      const deadline = Date.now() + 12000;
      while (Date.now() < deadline) {
        if (await check()) return;
        await new Promise(resolve => setTimeout(resolve, 200));
      }
      throw new Error('Timed out waiting for renderer state');
    };
    await waitUntil(() => window.webContents.executeJavaScript("!!document.querySelector('.door textarea')"));
    assert.equal(await window.webContents.executeJavaScript("document.querySelector('.door textarea').value"), '');
    assert.equal((await request('/health')).ready, true);
    assert.equal((await request('/state')).resident, 'leila');
    for (const [route, body] of [
      ['/addon/seed', {rid:'leila'}],
      ['/addon/infer', {rid:'leila'}],
      ['/addon/life_map?rid=leila'],
      ['/addon/chronicle?rid=leila'],
      ['/addon/city_gets?rid=leila'],
      ['/addon/city_profile?lang=ru'],
      ['/addon/cascade', {rid:'leila'}],
    ]) {
      const response = await fetch(backend.url + route, body ? {
        method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body),
      } : undefined);
      assert.equal(response.status, 200, route);
      assert.ok(await response.json(), route);
    }
    assert.equal((await request('/bootstrap', { lang: 'en', device: 'phone' })).rid, 'leila');
    const state = await request('/state');
    assert.ok(state.profile, 'New backend profile contract must be available');
    assert.equal(state.district, 'A');
    const answer = await request('/ask', { text: 'покажи план', rid: 'leila', device: 'phone' });
    assert.ok(answer.text);
    assert.equal(answer.lang, 'ru');
    for (const moment of state.moments) {
      const reply = await request(`/moment/${moment.id}`, { rid: 'leila', device: 'phone' });
      assert.ok(reply.text, `Missing reply for ${moment.id}`);
    }
    await request('/device/open', { rid: 'leila', device: 'hub' });
    const hub = openView('hub', '?rid=leila', { show: false });
    if (hub.webContents.isLoading()) await new Promise(resolve => hub.webContents.once('did-finish-load', resolve));
    const hubState = await hub.webContents.executeJavaScript("window.cureDesktop.request('/state?rid=leila')");
    assert.equal(hubState.resident, 'leila');
    assert.ok(hubState.devices.includes('hub'));
    assert.ok((await request('/memory?rid=leila')).model);
    // Create a profile through the actual UI; no prefilled family or resident ID.
    await window.webContents.executeJavaScript(`
      document.querySelector('.door textarea').value = 'Я переезжаю, нас 1 человек, бюджет на жильё 12000 в месяц';
      document.querySelector('.door .primary').click();
    `);
    await waitUntil(() => window.webContents.executeJavaScript("new URLSearchParams(location.search).get('rid')?.startsWith('guest-') && !!document.querySelector('.phone-view .msg.cure')"));
    const resident = await window.webContents.executeJavaScript("new URLSearchParams(location.search).get('rid')");
    const userState = await request('/state?rid=' + resident);
    assert.equal(userState.profile.household_size.value, 1);
    assert.equal(userState.profile.housing_budget_aed_month.value, 12000);
    await waitUntil(() => window.webContents.executeJavaScript("!!document.querySelector('.phone-view .msg.cure')"));
    const messages = await window.webContents.executeJavaScript("[...document.querySelectorAll('.phone-view .msg.cure')].map(x => x.innerText)");
    assert.equal(await window.webContents.executeJavaScript("document.querySelector('.phone-view #screen').querySelectorAll('details, pre.json').length"), 0);
    assert.deepEqual(await window.webContents.executeJavaScript("[...document.querySelectorAll('.phone-view #tabs button')].map(x => [...x.childNodes].filter(n => n.nodeType === Node.TEXT_NODE).map(n => n.textContent).join('').trim())"), ['Today', 'Plan', 'CURE', 'Knows', 'Trust']);
    for (const reply of userState.conversation.filter(x => x.role === 'cure')) {
      assert.ok(messages.some(text => text.includes(reply.text)), 'Backend response must appear verbatim in UI');
    }
    assert.ok(await window.webContents.executeJavaScript("!document.body.innerText.includes('Week of Leila') && !document.body.innerText.includes('22–28 Nov 2026')"));
    await window.webContents.executeJavaScript("document.querySelectorAll('.phone-view #tabs button')[3].click()");
    await waitUntil(() => window.webContents.executeJavaScript("!!document.querySelector('.knowledge-note')"));
    const knowledgeText = await window.webContents.executeJavaScript("document.querySelector('.phone-view #screen').innerText");
    assert.ok(knowledgeText.includes('What CURE knows'));
    assert.ok(knowledgeText.includes('12,000 AED / month'));
    assert.ok(knowledgeText.toLowerCase().includes('mind · current situation'));
    assert.ok(knowledgeText.toLowerCase().includes('associative memory'));
    const mind = await request('/mind?rid=' + resident);
    assert.equal(mind.identity.resident_id, resident);
    assert.ok(mind.graph.nodes.length);
    assert.equal(mind.mode, 'local');
    const recall = await request('/mind/recall', {rid:resident, query:'family', embeddings:false});
    assert.equal(recall.mode, 'local_graph_lexical');
    assert.ok(recall.memories.length);
    assert.deepEqual((await request('/agents/dynamic?rid=' + resident)).agents, []);
    assert.ok(!knowledgeText.includes('On disk') && !knowledgeText.includes('Open on another device'));
    await window.webContents.executeJavaScript("document.querySelectorAll('.phone-view #tabs button')[2].click()");
    // Delay the real request to verify the composer while a response is in flight.
    const originalFetch = global.fetch;
    global.fetch = async (url, options) => {
      if (String(url).includes('/twin/ask')) await new Promise(resolve => setTimeout(resolve, 1000));
      return originalFetch(url, options);
    };
    const composerCheck = await window.webContents.executeJavaScript(`
      (() => {
        const input = document.querySelector('.phone-view #chatbox textarea');
        window.composerReference = input;
        const text = ('Хочу уточнить вопрос\\n' + 'Длинное сообщение без чисел. '.repeat(25)).trim();
        input.value = text;
        input.dispatchEvent(new Event('input', {bubbles: true}));
        const height = parseFloat(input.style.height);
        const shiftAllowed = input.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter', shiftKey:true, bubbles:true, cancelable:true}));
        const enterAllowed = input.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter', bubbles:true, cancelable:true}));
        const visible = [...document.querySelectorAll('.phone-view .msg.you')].some(x => x.innerText === text);
        const focused = document.activeElement === input;
        input.value = 'Следующий черновик\\nЕщё одна строка';
        input.dispatchEvent(new Event('input', {bubbles:true}));
        return {text, height, shiftAllowed, enterAllowed, visible, focused, disabled:input.disabled};
      })()
    `);
    assert.ok(composerCheck.height > 46 && composerCheck.height <= 160);
    assert.equal(composerCheck.shiftAllowed, true);
    assert.equal(composerCheck.enterAllowed, false);
    assert.equal(composerCheck.visible, true);
    assert.equal(composerCheck.focused, true);
    assert.equal(composerCheck.disabled, false);
    await waitUntil(() => window.webContents.executeJavaScript("!document.querySelector('.phone-view .reply-pending')"));
    const finalComposer = await window.webContents.executeJavaScript(`({
      same: window.composerReference === document.querySelector('.phone-view #chatbox textarea'),
      draft: window.composerReference.value,
      focused: document.activeElement === window.composerReference,
      messages: [...document.querySelectorAll('.phone-view .msg.you')].map(x => x.innerText)
    })`);
    assert.equal(finalComposer.same, true);
    assert.equal(finalComposer.draft, 'Следующий черновик\nЕщё одна строка');
    assert.equal(finalComposer.focused, true);
    assert.equal(finalComposer.messages.filter(text => text === composerCheck.text).length, 1);
    const savedConversation = (await request('/state?rid=' + resident)).conversation;
    assert.equal(savedConversation.filter(m => m.role === 'you' && m.text === composerCheck.text).length, 1);
    assert.ok(savedConversation.some(m => m.role === 'you' && m.text.includes('нас 1 человек')));
    global.fetch = originalFetch;
    const residentHub = openView('hub', '', {show: false});
    if (residentHub.webContents.isLoading()) await new Promise(resolve => residentHub.webContents.once('did-finish-load', resolve));
    assert.equal(new URL(residentHub.webContents.getURL()).searchParams.get('rid'), resident);
    await waitUntil(() => residentHub.webContents.executeJavaScript("!!document.querySelector('#say .msg.cure')"));
    global.fetch = async (url, options) => {
      if (String(url).includes('/twin/ask')) await new Promise(resolve => setTimeout(resolve, 1000));
      return originalFetch(url, options);
    };
    const hubComposer = await residentHub.webContents.executeJavaScript(`(() => {
      const input = document.querySelector('#ask textarea');
      window.hubInput = input;
      input.value = 'Сообщение из Home hub';
      input.dispatchEvent(new Event('input', {bubbles: true}));
      document.querySelector('#ask .primary').click();
      const visible = [...document.querySelectorAll('#say .msg.you')].some(x => x.innerText === 'Сообщение из Home hub');
      input.value = 'Следующее сообщение';
      input.dispatchEvent(new Event('input', {bubbles: true}));
      return {visible, focused: document.activeElement === input, disabled: input.disabled};
    })()`);
    assert.deepEqual(hubComposer, {visible:true, focused:true, disabled:false});
    await waitUntil(() => residentHub.webContents.executeJavaScript("!document.querySelector('.reply-pending')"));
    assert.equal(await residentHub.webContents.executeJavaScript("window.hubInput.value"), 'Следующее сообщение');
    const hubConversation = (await request('/state?rid=' + resident)).conversation;
    assert.equal(hubConversation.filter(m => m.role === 'you' && m.text === 'Сообщение из Home hub').length, 1);
    await waitUntil(() => window.webContents.executeJavaScript("[...document.querySelectorAll('.phone-view .msg.you')].some(x => x.innerText === 'Сообщение из Home hub')"));
    global.fetch = originalFetch;
    if (process.env.CURE_CAPTURE_DIR) {
      await new Promise(resolve => setTimeout(resolve, 2000));
      fs.mkdirSync(process.env.CURE_CAPTURE_DIR, {recursive:true});
      fs.writeFileSync(path.join(process.env.CURE_CAPTURE_DIR, 'console-glass.png'), (await window.webContents.capturePage()).toPNG());
      fs.writeFileSync(path.join(process.env.CURE_CAPTURE_DIR, 'hub-glass.png'), (await residentHub.webContents.capturePage()).toPNG());
    }
    await new Promise(resolve => setTimeout(resolve, 2000));
    for (const view of [window, hub]) {
      const result = await view.webContents.executeJavaScript(
        "({text: document.body.innerText, toasts: [...document.querySelectorAll('.toast')].map(x => x.innerText)})");
      assert.ok(result.text.length > 100);
      assert.deepEqual(result.toasts, []);
      assert.ok(!result.text.includes('render error'), 'Card rendering failed');
    }
    assert.deepEqual(errors, []);
    const chatRequests = [];
    global.fetch = (url, options) => {
      if (String(url).includes('/twin/ask')) chatRequests.push(JSON.parse(options.body));
      return originalFetch(url, options);
    };
    await window.webContents.executeJavaScript("document.querySelectorAll('.profile-switch button')[1].click()");
    await waitUntil(() => window.webContents.executeJavaScript("new URLSearchParams(location.search).get('rid') === 'leila' && !!document.querySelector('.phone-view .msg.cure')"));
    assert.equal((await request('/state?rid=leila')).district, 'A');
    await window.webContents.executeJavaScript(`
      const input = document.querySelector('.phone-view #chatbox textarea');
      input.value = 'Where should we live?';
      input.dispatchEvent(new Event('input', {bubbles:true}));
      input.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter', bubbles:true}));
    `);
    await waitUntil(() => chatRequests.some(x => x.text === 'Where should we live?'));
    assert.deepEqual(chatRequests.find(x => x.text === 'Where should we live?'), {rid:'leila', text:'Where should we live?', device:'phone'});
    await waitUntil(() => window.webContents.executeJavaScript("!document.querySelector('.phone-view .reply-pending')"));
    await window.webContents.executeJavaScript("document.querySelectorAll('.profile-switch button')[0].click()");
    await waitUntil(() => window.webContents.executeJavaScript("!!document.querySelector('.door textarea')"));
    assert.equal(await window.webContents.executeJavaScript("document.querySelector('.door textarea').value"), '');
    const newResident = await window.webContents.executeJavaScript("new URLSearchParams(location.search).get('rid')");
    assert.match(newResident, /^guest-[a-f0-9]{12}$/);
    assert.notEqual(newResident, resident);
    await window.webContents.executeJavaScript(`
      document.querySelector('.door textarea').value = 'We are 2 people moving to Dubai';
      document.querySelector('.door .primary').click();
    `);
    await waitUntil(() => chatRequests.some(x => x.text === 'We are 2 people moving to Dubai'));
    assert.deepEqual(chatRequests.find(x => x.text === 'We are 2 people moving to Dubai'), {rid:newResident, text:'We are 2 people moving to Dubai', device:'phone'});
    await waitUntil(() => window.webContents.executeJavaScript("!!document.querySelector('.phone-view .msg.cure')"));
    assert.equal((await request('/state?rid=' + newResident)).resident, newResident);
    assert.ok((await request('/state?rid=leila')).conversation.some(x => x.role === 'you' && x.text === 'Where should we live?'));
    global.fetch = originalFetch;
    assert.ok((await request('/state?rid=' + resident)).resident, 'New must preserve the existing guest profile');
    const child = backend.child;
    await backend.stop();
    if (child.exitCode === null && child.signalCode === null) await new Promise(resolve => child.once('exit', resolve));
    await waitUntil(() => window.webContents.executeJavaScript("document.querySelector('.backend-status').dataset.status === 'unavailable' && document.querySelector('.phone-view #screen').innerText.includes('Backend unavailable')"));
    await waitUntil(() => hub.webContents.executeJavaScript("document.querySelector('#say').innerText.includes('Backend unavailable')"));
    assert.equal(await hub.webContents.executeJavaScript("document.querySelector('#air').innerText + document.querySelector('#now').innerText + document.querySelector('#appr').innerText"), '');
    console.log('PASS: backend health, addon routes, IPC, chat, moments, shared state, exact Leila/guest payloads, user profiles, verbatim replies, disconnect, shutdown');
  } catch (error) {
    console.error(error);
    process.exitCode = 1;
  } finally {
    await backend?.stop();
    fs.rmSync(store, { recursive: true, force: true });
    app.exit(process.exitCode || 0);
  }
});
