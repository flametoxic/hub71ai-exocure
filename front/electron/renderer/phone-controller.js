import { createShared } from './shared.js';
import colorLogo from '../logocolor.svg';
import { knowledgeEntries } from './knowledge.js';
export function mount(root, search, onResident = () => {}) {
const { Q, CURE, EN, agentName, h, n0, n1, pct, mid, toast, card, kv, json, chips, win, spark, hoursRow, decide, R, renderCard, renderAgent, renderConversation, micButton, speak } = createShared(search);

if (Q.get('embed')) root.classList.add('embed');
CURE.device = Q.get('device') || 'phone';
CURE.voiceOn = Q.get('voice') === '1';
const TABS = [['today', '◐', 'Today'], ['plan', '⟶', 'Plan'], ['chat', '✦', 'CURE'], ['memory', '◈', 'Knows'], ['trust', '⛨', 'Trust']];
let tab = TABS.some(([id]) => id === Q.get('tab')) ? Q.get('tab') : 'chat', ST = null, lastSpoken = null, sending = false, outgoing = null;
let composer, sendButton;
function resizeComposer() {
  if (!composer) return;
  composer.style.height = 'auto';
  composer.style.height = Math.min(composer.scrollHeight, 160) + 'px';
}
function updateSendButton() { if (sendButton) sendButton.disabled = sending || !ST?.resident || !composer?.value.trim(); }
const $ = id => root.querySelector(`[id="${id}"]`);

function drawTabs() {
  $('tabs').innerHTML = '';
  TABS.forEach(([id, i, t]) => $('tabs').append(h('button', {class: tab === id ? 'on' : '', onclick: () => { tab = id; draw(); }},
    h('span', {class: 'i'}, i), t, id === 'today' && ST && ST.pending && ST.pending.length ? h('span', {class: 'dot'}, ST.pending.length) : null)));
}
function header(t, s) { return [h('h2', {}, t), s ? h('div', {class: 'sub'}, s) : null]; }
function fmtClock(iso) { const d = new Date(iso); return d.toLocaleDateString('en-GB', {weekday: 'short', day: 'numeric', month: 'short', timeZone: 'UTC'}) + ' · ' + iso.slice(11, 16); }

function doorScreen() {
  const input = h('textarea', {rows: 4, placeholder: 'Tell CURE about yourself and what you need'});
  const go = async () => {
    const voice = input.value.trim();
    if (!voice) { input.focus(); return; }
    if (!CURE.rid) CURE.rid = 'guest-' + crypto.randomUUID().replaceAll('-', '').slice(0, 12);
    const reply = await CURE.ask(voice);
    CURE.rid = reply.rid;
    if (!Q.has('embed')) {
      const url = new URL(location.href);
      url.searchParams.set('rid', reply.rid);
      history.replaceState(null, '', url);
    }
    onResident(reply.rid);
    tab = 'chat';
    CURE.kick();
  };
  return h('div', {class: 'door'}, h('button', {class: 'orb logo-orb', onclick: go, title: 'Open the door', 'aria-label': 'Open the door to CURE'}, h('img', {src: colorLogo, alt: ''})),
    h('div', {style: 'text-align:center'}, h('h2', {}, 'Open the door to CURE'), h('div', {class: 'sub'}, 'Say who you are and what you need.')),
    h('div', {class: 'row'}, h('span', {class: 'grow'}), micButton(input, go, () => 'en')), input,
    h('button', {class: 'primary', onclick: go}, 'Open the door'));
}

function todayScreen() {
  const L = ST.last || {}, T = ST.today || {};
  const out = [...header('Today', fmtClock(ST.clock))];
  if (T.forecast !== null && T.pm10_by_hour) {
    out.push(card('Air now', `PM10 ${T.pm10_now} · ${n1(T.temperature_now)}°${T.data_mode ? ' · ' + T.data_mode : ''}`,
      spark(T.pm10_by_hour.slice(6, 21), {limits: [{v: T.general_limit, c: '#f3b54a', label: 'general'}, ...Object.entries(T.limits_pm10).filter(([k, v]) => v < T.general_limit).map(([k, v]) => ({v, c: '#ff6b6b', label: 'personal'}))], now: T.hour, h0: 6}),
      Object.entries(T.windows).map(([p, w]) => h('div', {class: 'row', style: 'margin:3px 0'}, h('span', {style: 'width:92px'}, p), h('div', {class: 'grow'}, hoursRow(w, 6, 21, T.hour)))),
      h('div', {class: 'row', style: 'margin:3px 0'}, h('b', {style: 'width:92px'}, 'family'), h('div', {class: 'grow'}, hoursRow(T.family, 6, 21, T.hour)))));
  }
  if (ST.pending && ST.pending.length) out.push(card('Waiting for your yes', ST.pending.length, ST.pending.map(p => h('div', {class: 'row', style: 'margin:6px 0'}, h('span', {class: 'grow'}, p.title, h('div', {class: 'dim', style: 'font-size:11px'}, p.label)),
    h('button', {class: 'primary', onclick: () => decide(p.id, true)}, 'Yes'), h('button', {onclick: () => decide(p.id, false)}, 'No')))));
  ['morning', 'home', 'conflict', 'beach', 'guest', 'budget'].forEach(k => { if (L[k]) out.push(renderCard(L[k])); });
  if (L.reply) out.push(card('Last from CURE', L.reply.at.slice(11, 16), h('div', {}, L.reply.text)));
  if (out.length <= 2) out.push(h('div', {class: 'muted'}, 'Nothing yet — talk to CURE.'));
  return out;
}

function chatScreen() {
  const list = h('div', {});
  renderConversation(list, ST.conversation || [], {onQuick: q => send(q)});
  if (outgoing && (ST.conversation || []).filter(m => m.role === 'you' && m.text === outgoing.text).length <= outgoing.previousCount) {
    list.append(h('div', {class: 'msg you'}, outgoing.text));
  }
  if (sending) list.append(h('div', {class: 'reply-pending', role: 'status'}, ST?.lang === 'ru' ? 'CURE думает…' : 'CURE is thinking…'));
  CURE.agents().then(({agents}) => {
    if (tab === 'chat' && list.isConnected) agents.filter(a => a.status !== 'ARCHIVED').forEach(a => list.append(renderAgent(a)));
  }).catch(() => {});
  setTimeout(() => { $('screen').scrollTop = 1e9; }, 0);
  return [list];
}
async function send(text) {
  text = (text || '').trim();
  if (!text || sending || !ST?.resident) return;
  sending = true;
  outgoing = {text, previousCount: (ST?.conversation || []).filter(m => m.role === 'you' && m.text === text).length};
  if (composer) { composer.value = ''; resizeComposer(); composer.focus({preventScroll: true}); }
  updateSendButton();
  draw();
  try {
    const r = await CURE.ask(text);
    if (r && r.text) speak(r.text, r.lang);
    await CURE.kick();
  } catch {
    if (composer && !composer.value) { composer.value = text; resizeComposer(); }
    outgoing = null;
  } finally {
    sending = false;
    updateSendButton();
    draw();
  }
}
function drawChatbox() {
  const box = $('chatbox'); box.innerHTML = '';
  composer = h('textarea', {rows: 1, placeholder: 'Ask CURE…', 'aria-label': 'Message CURE',
    oninput: () => { resizeComposer(); updateSendButton(); },
    onkeydown: e => {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(composer.value); }
    }});
  sendButton = h('button', {class: 'primary', onclick: () => send(composer.value)}, 'Send');
  box.append(micButton(composer, t => send(t), () => /[а-яё]/i.test(composer.value) ? 'ru' : ST?.lang), composer, sendButton);
  updateSendButton();
  requestAnimationFrame(() => { resizeComposer(); composer.focus({preventScroll: true}); });
}

function planScreen() {
  const L = ST.last || {};
  const drafts = (ST.drafts || []).slice().reverse();
  return [...header('Plan', ST.arrival ? 'target ' + ST.arrival : 'no arrival date yet'),
    h('div', {class: 'row wrap'}, h('button', {onclick: () => CURE.act('plan')}, 'Recompute plan'), h('button', {onclick: () => CURE.act('agent_search')}, 'Agents: check official durations')),
    L.plan ? renderCard(L.plan) : null, L.plan_change ? renderCard(L.plan_change) : null, L.why ? renderCard(L.why) : null,
    drafts.length ? h('div', {class: 'tag', style: 'margin-top:12px'}, 'Drafts — you send them') : null, drafts.map(d => renderCard({type: 'draft', ...d}))];
}

async function memoryScreen(el) {
  const state = ST;
  el.append(h('div', {class: 'muted'}, 'Loading saved notes…'));
  const [m, mind] = await Promise.all([CURE.memory(), CURE.mind()]);
  if (tab !== 'memory' || ST !== state || !el.isConnected) return;
  const name = (state.household || []).find(person => person.role === 'adult')?.name;
  const notes = knowledgeEntries(state, m);
  el.replaceChildren(...header('What CURE knows', name ? `${name}’s CURE` : 'Your CURE'),
    h('div', {class: 'knowledge-summary'}, `${notes.length} saved notes`),
    h('div', {class: 'knowledge-notes'}, notes.length ? notes.map(note =>
      h('article', {class: 'card knowledge-note'},
        h('div', {class: 'knowledge-note-meta'}, note.title, h('span', {}, note.kind)),
        h('div', {class: 'knowledge-note-text'}, note.text),
        note.at ? h('time', {class: 'muted', datetime: note.at}, note.at.slice(0, 10)) : null))
      : h('div', {class: 'muted'}, 'No saved notes yet. Tell CURE about yourself in chat.')));
  const situation = mind.current_situation;
  el.append(card('Mind · current situation', mind.mode,
    h('p', {}, situation.district ? 'District: ' + situation.district : 'District not selected'),
    h('p', {}, situation.arrival ? 'Arrival: ' + situation.arrival : 'Arrival date not set'),
    h('p', {}, 'Scenario clock: ' + situation.scenario_clock),
    mind.goals.map(goal => h('p', {}, goal.title + ' · ' + goal.provenance))));
  el.append(card('Life tasks', 'CURE state', mind.tasks.map(task => h('p', {}, task.title + ' · ' + task.status))));
  const recallInput = h('input', {placeholder: 'Recall a memory…', 'aria-label': 'Recall memory'});
  const recalled = h('div', {});
  el.append(card('Associative memory', 'provenance', recallInput,
    h('button', {onclick: async () => {
      const reply = await CURE.api('/mind/recall', {rid:CURE.rid, query:recallInput.value, embeddings:mind.mode === 'openai'});
      recalled.replaceChildren(h('p', {}, reply.mode), ...reply.memories.map(memory => h('p', {}, memory.text + ' · ' + memory.provenance)));
    }}, 'Recall'), recalled,
    mind.memories.slice(-12).map(memory => h('p', {}, memory.text + ' · ' + memory.provenance))));
  el.append(card('Recent inferences', 'evidence, not confirmed facts',
    mind.recent_inferences.map(inference => h('p', {}, inference.assessment + ' · ' + inference.kind + ' · ' + inference.provenance)),
    h('button', {disabled:mind.mode !== 'openai', onclick:async () => {
      await CURE.api('/mind/infer', {rid:CURE.rid, kind:'routine'}); await draw();
    }}, 'Infer routine')));
  mind.agents.filter(agent => agent.status !== 'ARCHIVED').forEach(agent => el.append(renderAgent(agent)));
}
function trustScreen() {
  return [...header('Trust', ''),
    R.trustTable(ST.trust || [], ST.trust_policy || {})];
}

async function draw() {
  const s = $('screen');
  if (!ST) {
    $('clock').textContent = '';
    $('who').textContent = '';
    $('tabs').classList.add('hidden');
    $('chatbox').classList.toggle('hidden', !composer || tab !== 'chat');
    updateSendButton();
    s.replaceChildren(h('div', {class: 'muted'}, 'Backend unavailable. Waiting for connection…'));
    return;
  }
  $('clock').textContent = fmtClock(ST.clock);
  if (!ST.resident) { $('who').textContent = ''; $('tabs').classList.add('hidden'); $('chatbox').classList.add('hidden'); s.innerHTML = ''; s.append(doorScreen()); return; }
  $('who').textContent = (ST.household || []).length + ' people';
  $('tabs').classList.remove('hidden'); drawTabs();
  $('chatbox').classList.toggle('hidden', tab !== 'chat');
  if (tab === 'chat' && !$('chatbox').children.length) drawChatbox();
  updateSendButton();
  const keep = s.scrollTop;
  s.innerHTML = '';
  if (tab === 'memory') return memoryScreen(s);
  const f = {today: todayScreen, chat: chatScreen, plan: planScreen, trust: trustScreen}[tab];
  s.append(...[].concat(f()).filter(Boolean));
  if (tab !== 'chat') s.scrollTop = keep;
}
CURE.onState(st => {
  const conv = st?.conversation || [];
  const last = conv[conv.length - 1];
  if (last && last.role === 'cure' && last.at !== lastSpoken && ST) { lastSpoken = last.at; speak(last.text, st.lang); }
  ST = st; return draw();
});
CURE.watch(1500);
const composerObserver = new ResizeObserver(() => {
  root.style.setProperty('--composer-space', ($('chatbox').offsetHeight ? 120 + $('chatbox').offsetHeight : 110) + 'px');
});
composerObserver.observe($('chatbox'));

return { client: CURE, dispose: () => { composerObserver.disconnect(); CURE.dispose(); } };
}

