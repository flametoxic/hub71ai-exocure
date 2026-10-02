import { createShared } from './shared.js';
export function mount(root, search, onResident = () => {}) {
const { Q, CURE, h, card, chips, renderCard } = createShared(search);
const $ = id => root.querySelector(`[id="${id}"]`);
function setRid(rid) {
  CURE.rid = rid;
  const url = new URL(location.href);
  if (rid) url.searchParams.set('rid', rid);
  else url.searchParams.delete('rid');
  history.replaceState(null, '', url);
  onResident(rid);
  CURE.lastRev = null;
  CURE.kick();
}
function selectNew() { setRid('guest-' + crypto.randomUUID().replaceAll('-', '').slice(0, 12)); }
async function resetAll() {
  if (!confirm('Reset all resident data? This erases all contours.')) return;
  await CURE.api('/reset', {});
  selectNew();
}
async function selectLeila() {
  await CURE.api('/state?rid=leila');
  setRid('leila');
}
async function hood(st) {
  $('city').replaceChildren();
  if (!st) { $('hood').textContent = 'Backend unavailable. Waiting for connection…'; return; }
  const conv = (st.conversation || []).filter(m => m.role === 'cure');
  const m = conv[conv.length - 1];
  const el = $('hood'); el.innerHTML = '';
  if (st.resident) {
    const rid = CURE.rid;
    try {
      const mind = await CURE.mind();
      if (CURE.disposed || CURE.rid !== rid || !el.isConnected) return;
      el.append(card('Family overview', '',
        h('p', {}, `${mind.week?.event_count || 0} events · ${mind.week?.issues?.length || 0} schedule conflicts`),
        h('p', {}, `${mind.memories?.length || 0} memories · ${mind.tasks?.filter(task => task.status === 'blocked').length || 0} blocked tasks`),
        (mind.agents || []).filter(agent => agent.status !== 'ARCHIVED').map(agent => h('p', {}, `${agent.task} · ${agent.status.replaceAll('_', ' ').toLowerCase()}`))));
    } catch { el.append(h('p', {class: 'muted'}, 'Family overview unavailable.')); }
  }
  if (!m) { el.append(h('div', {class: 'muted'}, 'No replies yet.')); return; }
  el.append(card('Route', m.route || '—', h('div', {}, m.text)));
  for (const t of (m.trace_ids || []).slice(0, 4)) {
    try {
      const w = await CURE.why(t);
      el.append(card('Engine trace', t.slice(0, 8) + ' · ' + (w.data_mode || ''), h('div', {}, w.label || ''), chips(w.formulas || [], 'acc'),
        h('div', {class: 'dim', style: 'margin-top:6px;font-size:11px'}, 'data: ' + (w.sources || []).map(s => s.dataset + (s.data_mode ? ' (' + s.data_mode + ')' : '')).join(' · '))));
    } catch (e) {}
  }
  const cityTypes = ['cohort', 'chain', 'storm', 'neighbors', 'city_request', 'learning', 'city_family', 'impact'];
  const c = (m.cards || []).filter(x => cityTypes.includes(x.type));
  if (c.length) { $('city').innerHTML = ''; c.forEach(x => $('city').append(renderCard(x))); }
}
CURE.onState(hood);
setRid(Q.get('rid') || 'leila');
CURE.watch(1500);
return { client: CURE, dispose: () => CURE.dispose(), resetAll, setRid, selectLeila, selectNew };
}
