import { createShared } from './shared.js';
import {mountAgentFeed} from './mind-view.js';
export function mount(root, search, onResident = () => {}) {
const shared = createShared(search);
const { Q, CURE, EN, agentName, h, n0, n1, pct, mid, toast, card, kv, json, chips, win, spark, hoursRow, decide, R, renderCard, renderConversation, micButton, speak } = shared;

if (Q.get('embed')) root.classList.add('embed');
CURE.device = 'home hub';
CURE.voiceOn = Q.get('voice') === '1';
const $ = id => root.querySelector(`[id="${id}"]`);
let ST = null, sending = false, outgoing = null;
let disposeAgents = null;
function resizeInput() { input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, 120) + 'px'; }
function updateSend() { sendButton.disabled = sending || !ST?.resident || !input.value.trim(); }
const input = h('textarea', {rows: 1, placeholder: 'Ask CURE…', 'aria-label': 'Message CURE',
  oninput: () => { resizeInput(); updateSend(); },
  onkeydown: e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); go(input.value); } }});
const sendButton = h('button', {class: 'primary', onclick: () => go(input.value)}, 'Send');
async function go(t) {
  t = (t || '').trim(); if (!t || sending || !ST?.resident) return;
  sending = true;
  outgoing = {text: t, previousCount: (ST.conversation || []).filter(m => m.role === 'you' && m.text === t).length};
  input.value = ''; resizeInput(); input.focus({preventScroll: true}); drawConversation(); updateSend();
  try { const r = await CURE.ask(t); speak(r.text, r.lang); await CURE.kick(); }
  catch { if (!input.value) { input.value = t; resizeInput(); } outgoing = null; }
  finally { sending = false; updateSend(); drawConversation(); }
}
$('ask').append(micButton(input, go, () => ST && ST.lang), input, sendButton);
updateSend();
function drawConversation() {
  const el = $('say');
  if (!ST) { el.textContent = 'Backend unavailable. Waiting for connection…'; return; }
  if (!ST.resident) { el.textContent = 'No contour open. Open the door on the phone.'; return; }
  const conv = ST.conversation || [];
  renderConversation(el, conv, {cards: false});
  if (outgoing && conv.filter(m => m.role === 'you' && m.text === outgoing.text).length <= outgoing.previousCount) el.append(h('div', {class: 'msg you'}, outgoing.text));
  if (sending) el.append(h('div', {class: 'reply-pending', role: 'status'}, ST.lang === 'ru' ? 'CURE думает…' : 'CURE is thinking…'));
  el.scrollTop = el.scrollHeight;
}

function draw() {
  for (const id of ['air', 'now', 'appr', 'bottom', 'clock', 'date']) $(id).replaceChildren();
  updateSend(); drawConversation();
  if (!ST) return;
  $('clock').textContent = ST.clock.slice(11, 16);
  $('date').textContent = new Date(ST.clock).toLocaleDateString('en-GB', {weekday: 'long', day: 'numeric', month: 'long', timeZone: 'UTC'});
  if (!ST.resident) return;
  const T = ST.today || {}, L = ST.last || {};
  // air now: per person, can they be outside this hour?
  const air = $('air'); air.innerHTML = '';
  if (T.pm10_by_hour) {
    const inWin = ws => (ws || []).some(w => T.hour >= +w.from.slice(0, 2) && T.hour < +w.to.slice(0, 2));
    air.append(h('div', {class: 'row'}, h('span', {class: 'big'}, 'PM10 ' + T.pm10_now), h('span', {class: 'grow'}), h('span', {class: 'mini'}, n1(T.temperature_now) + '°')),
      ...Object.entries(T.windows).map(([p, w]) => h('div', {class: 'pp'}, h('span', {}, p), inWin(w) ? h('span', {class: 'pill-ok'}, 'outside ok') : h('span', {class: 'pill-bad'}, 'stay in'))));
  } else air.append(h('div', {class: 'mini'}, 'No forecast returned by backend for this date.'));
  // now
  const now = $('now'); now.innerHTML = '';
  const rows = [];
  if (L.morning) { const d = L.morning.departures[0]; rows.push(['School run', `leave ${d.best ? d.best.depart : '—'} (not ${d.usual.depart})`]); }
  if (L.home) rows.push(['Home', `cooling from ${L.home.start_at} · ≤ ${n1(L.home.target_c)}° by ${L.home.return_at}`]);
  if (L.conflict) rows.push(['Car', `${L.conflict.meeting}: ${L.conflict.move.from} → ${L.conflict.move.to}`]);
  if (L.beach) rows.push(['Beach', `${L.beach.window.from}–${L.beach.window.to} · reminder ${L.beach.remind}`]);
  if (L.errands) rows.push(['Errands', `${L.errands.start_at} · one trip`]);
  now.append(h('div', {class: 'tag', style: 'margin-bottom:6px'}, 'Today'), rows.length ? kv(Object.fromEntries(rows)) : h('div', {class: 'mini'}, 'Nothing scheduled yet.'));
  // what CURE said
  // approvals
  const a = $('appr'); a.innerHTML = '';
  (ST.pending || []).forEach(p => a.append(h('div', {class: 'item'}, h('div', {class: 't'}, p.title), h('div', {class: 'mini', style: 'margin:-4px 0 8px'}, p.label),
    h('div', {class: 'row'}, h('button', {class: 'primary', onclick: () => decide(p.id, true)}, 'Yes'), h('button', {onclick: () => decide(p.id, false)}, 'No')))));
  if (!(ST.pending || []).length) a.append(h('div', {class: 'mini'}, 'Nothing waiting.'));
  // windows strip
  const b = $('bottom'); b.innerHTML = '';
  if (T.windows) {
    b.append(h('span', {}), h('div', {class: 'hourscale'}, Array.from({length: 15}, (_, i) => h('span', {}, (6 + i) + ''))));
    Object.entries(T.windows).forEach(([p, w]) => b.append(h('span', {}, p), hoursRow(w, 6, 21, T.hour)));
  }
}
CURE.onState(async st => {
  disposeAgents?.(); disposeAgents = null;
  ST = st; draw();
  if (!st?.resident) return;
  const feed = h('div', {class: 'hub-specialists'});
  $('say').append(feed);
  disposeAgents = mountAgentFeed(feed, shared);
  try {
    const mind = await CURE.mind();
    if (ST !== st || CURE.disposed) return;
    const upcoming = (mind.week?.events || []).filter(event => event.date >= st.clock.slice(0, 10))
      .sort((a, b) => a.date.localeCompare(b.date) || a.start - b.start).slice(0, 4);
    if (upcoming.length) $('now').append(h('div', {class: 'tag', style: 'margin-top:12px'}, 'Upcoming family plans'),
      upcoming.map(event => h('p', {class: 'mini'}, `${event.date} · ${String(Math.floor(event.start)).padStart(2, '0')}:${String(Math.round(event.start % 1 * 60)).padStart(2, '0')} · ${event.en || event.id}`)));
  } catch { /* Keep the current hub state when the optional schedule cannot load. */ }
});
CURE.watch(1500);

return { client: CURE, dispose: () => { disposeAgents?.(); CURE.dispose(); } };
}
