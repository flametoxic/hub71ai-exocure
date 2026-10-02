import { createShared } from './shared.js';
export function mount(root, search, onResident = () => {}) {
const { Q, CURE, h, card, chips, renderCard, renderAgent, json } = createShared(search);
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
    try {
      const mind = await CURE.mind();
      el.append(card('World Model', mind.mode, h('p', {}, `${mind.memories.length} memories · ${mind.graph.nodes.length} graph nodes`),
        json(mind.current_situation), json(mind.last_trace),
        mind.recent_inferences.map(inference => card('Inference', inference.provenance, h('p', {}, inference.assessment), chips(inference.evidence_ids))),
        mind.agents.map(renderAgent)));
    } catch {}
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
if (Q.get('rid')) setRid(Q.get('rid')); else selectNew();
CURE.watch(1500);
return { client: CURE, dispose: () => CURE.dispose(), resetAll, setRid, selectLeila, selectNew };
}
