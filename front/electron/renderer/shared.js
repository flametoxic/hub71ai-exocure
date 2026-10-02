import { request } from './api-client.mjs';
export function createShared(search = "") {
/* CURE windows: shared API client + card renderers. Windows keep no data: everything lives in the resident's
   contour in the CURE core; every window re-reads /twin/state. */
const Q = new URLSearchParams(search);
const CURE = {
  rid: Q.get('rid') || '',
  device: 'phone',
  listeners: [],
  lastRev: null,
  disposed: false,
  async api(path, body, quiet = false) {
    try { return await request(path, body); }
    catch (error) { if (!quiet) toast('Error: ' + error.message.slice(0, 160)); throw error; }
  },
  act(action, args = {}) { return this.api('/act', {rid: this.rid, action, args, device: this.device}).then(r => (this.kick(), r)); },
  ask(text) { return this.api('/ask', {rid: this.rid, text, device: this.device}).then(r => (this.kick(), r)); },
  state() { return this.api('/state?rid=' + encodeURIComponent(this.rid)); },
  memory() { return this.api('/memory?rid=' + encodeURIComponent(this.rid)); },
  mind() { return this.api('/mind?rid=' + encodeURIComponent(this.rid)); },
  agents() { return this.api('/agents/dynamic?rid=' + encodeURIComponent(this.rid)); },
  why(tid) { return this.api('/why/' + tid); },
  onState(cb) { this.listeners.push(cb); },
  async kick() {
    if (this.disposed || this.polling) return null;
    this.polling = true;
    const rid = this.rid;
    try {
      const s = await this.api('/state?rid=' + encodeURIComponent(rid), undefined, true);
      if (this.disposed || rid !== this.rid) return null;
      const key = JSON.stringify(s);
      if (key !== this.lastRev) { this.lastRev = key; await Promise.all(this.listeners.map(f => f(s))); }
      return s;
    } catch (e) {
      if (!this.disposed && rid === this.rid) {
        this.lastRev = null;
        await Promise.allSettled(this.listeners.map(f => f(null)));
      }
      return null;
    } finally { this.polling = false; }
  },
  dispose() { this.disposed = true; clearInterval(this.timer); this.listeners = []; },
  watch(ms = 1500) { clearInterval(this.timer); this.kick(); this.timer = setInterval(() => this.kick(), ms); },
};


// Interface labels
const EN = {
  agents: {documents: 'Documents', housing: 'Housing', school: 'School & kids', family_health: 'Family health: air and heat', budget: 'Budget', business: 'Business'},
  levels: ['shadow', 'advises', 'proposes', 'acts alone'],
  trips: {school_am: 'School run', office_am: 'To the office', school_pm: 'From school', office_pm: 'From the office', park: 'Park'},
};
const agentName = by => EN.agents[(by || '').replace('agent:', '')] || by;

// ---------------------------------------------------------------- helpers
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style') el.style.cssText = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (k === 'html') el.innerHTML = v;
    else el.setAttribute(k, v);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}
const n0 = x => x == null || isNaN(x) ? '—' : Math.round(+x).toLocaleString('en-US').replace(/,/g, ' ');
const n1 = x => x == null || isNaN(x) ? '—' : (Math.abs(x - Math.round(x)) < 0.05 ? String(Math.round(x)) : (+x).toFixed(1));
const pct = x => x == null ? '—' : Math.round(100 * x) + '%';
const mid = v => v && typeof v === 'object' ? v.mid : v;
function toast(t) { const d = h('div', {class: 'toast'}, t); document.body.append(d); setTimeout(() => d.remove(), 3500); }
function card(title, tagline, ...body) { return h('div', {class: 'card'}, h('h4', {}, title, tagline ? h('span', {class: 'tagline'}, tagline) : null), ...body); }
function kv(obj) { return h('div', {class: 'kv'}, Object.entries(obj).map(([k, v]) => [h('div', {class: 'k'}, k), h('div', {}, v)])); }
function json(o) { return h('details', {}, h('summary', {}, 'details'), h('pre', {class: 'json'}, JSON.stringify(o, null, 2))); }
function chips(list, cls = '') { return h('div', {class: 'row wrap'}, list.map(x => h('span', {class: 'chip ' + cls}, x))); }
function win(ws) { return (ws || []).map(w => `${w.from}–${w.to}`).join(', ') || '—'; }

function spark(values, {limits = [], now = null, h0 = 0, maxY = null} = {}) {
  const W = 300, H = 56, mx = maxY || Math.max(...values, ...limits.map(l => l.v)) * 1.1 || 1;
  const x = i => i * W / (values.length - 1), y = v => H - 4 - (v / mx) * (H - 10);
  const pts = values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const svg = `<svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    <polyline points="${pts}" fill="none" stroke="#3D7CF5" stroke-width="2"/>
    ${limits.map(l => `<line x1="0" x2="${W}" y1="${y(l.v)}" y2="${y(l.v)}" stroke="${l.c}" stroke-dasharray="4 4" stroke-width="1"/>
      <text x="${W - 2}" y="${y(l.v) - 3}" fill="${l.c}" font-size="9" text-anchor="end">${l.label}</text>`).join('')}
    ${now != null ? `<line x1="${x(now - h0)}" x2="${x(now - h0)}" y1="0" y2="${H}" stroke="#E0EBFC" stroke-width="1"/>` : ''}
  </svg>`;
  return h('div', {html: svg});
}

function hoursRow(windows, h0 = 6, h1 = 21, now = null) {
  const ok = hr => (windows || []).some(w => hr >= +w.from.slice(0, 2) && hr < +w.to.slice(0, 2));
  const cells = [];
  for (let hr = h0; hr < h1; hr++) cells.push(h('span', {class: (ok(hr) ? 'ok' : '') + (now === hr ? ' now' : ''), title: hr + ':00'}));
  return h('div', {class: 'hours'}, cells);
}

// ---------------------------------------------------------------- actions from cards
async function decide(id, yes) { await CURE.act('decide', {item_id: id, approved: yes}); }

// ---------------------------------------------------------------- card renderers
const R = {
  dynamic_agent() { return card('Specialist', '', h('p', {}, 'Research is starting. Follow the progress and review findings below.')); },
  profile_intake(c) {
    const fields = Object.values(c.fields || {});
    const known = fields.filter(f => ['known', 'derived'].includes(f.status)).length;
    return card('Your profile', c.complete ? 'ready' : 'getting to know you',
      h('div', {class: 'profile-progress', role: 'progressbar', 'aria-valuenow': known, 'aria-valuemin': 0, 'aria-valuemax': fields.length},
        h('i', {style: `width:${fields.length ? known / fields.length * 100 : 0}%`})),
      h('div', {class: 'muted'}, `${known} / ${fields.length}`));
  },
  door(c) {
    return card('Door', 'what happened to your words',
      kv({'Heard': c.heard, 'Sent to the language model': c.sent_to_language_model,
          'Sealed in your contour': h('span', {class: 'chip sealed'}, `${c.sealed_items} item(s) · ${c.sealed_where}`),
          'Derived constraints (visible)': Object.entries(c.constraints || {}).map(([w, v]) => h('div', {}, `${w}: ` + Object.entries(v).map(([k, x]) => `${k}=${x}`).join(', '))),
          'Pseudonyms': h('span', {class: 'mono'}, Object.entries(c.pseudonyms || {}).map(([k, v]) => `${k}:${v}`).join('  ')),
          'Consents': chips((c.consents || []).map(x => `${x.purpose} · ${x.status}`), 'acc')}));
  },
  districts(c) {
    const names = c.axes.names, units = c.axes.units;
    return card('Districts', `by your priorities · focus: ${c.focus}`,
      h('table', {class: 't'}, h('tr', {}, h('th', {}, ''), h('th', {class: 'num'}, 'AED / month'), h('th', {class: 'num'}, 'road h / month'), h('th', {class: 'num'}, `${c.focus}: heat+dust min / month`), h('th', {}, '')),
        Object.entries(c.axes.values).map(([d, v]) => h('tr', {class: d === c.recommended ? 'hl' : ''}, h('td', {}, h('b', {}, d)),
          h('td', {class: 'num'}, n0(v[0])), h('td', {class: 'num'}, n1(v[1])), h('td', {class: 'num'}, n0(v[2])),
          h('td', {}, d === c.recommended ? h('span', {class: 'badge'}, 'recommended') : (c.pareto.includes(d) ? h('span', {class: 'chip'}, 'Pareto') : h('span', {class: 'chip bad'}, 'dominated')))))),
      h('div', {class: 'row wrap', style: 'margin-top:8px'}, h('span', {class: 'muted'}, 'Your weights λ:'),
        names.map(n => h('span', {class: 'chip acc'}, `${n} ${(+c.lambda[n]).toFixed(2)}`)),
        c.switch ? h('span', {class: 'chip warn'}, `switch to ${c.switch.to} if money ≥ ${c.switch.money_weight}`) : null));
  },
  why_district(c) {
    return card('Why this district', 'λ-weighted gap per axis', kv(Object.fromEntries(Object.entries(c.advantage_by_axis).map(([k, v]) => [k, (+v).toFixed(3)]))),
      h('div', {class: 'dim mono', style: 'margin-top:6px'}, c.formula));
  },
  day(c) {
    const f = o => `${n0(mid(o.transit_min))} min road`;
    return card(`Day in ${c.month}`, `district ${c.district}`, kv({[c.month]: f(c.after), [c.base_month + ' (hardest)']: f(c.before)}));
  },
  day_compare(c) {
    return card(`Your day in ${c.month}`, 'per district', h('table', {class: 't'}, h('tr', {}, h('th', {}, ''), h('th', {class: 'num'}, 'road min/day'), h('th', {class: 'num'}, `${c.focus}: heat+dust min`)),
      Object.entries(c.districts).map(([d, v]) => h('tr', {}, h('td', {}, d), h('td', {class: 'num'}, n0(v.transit_min)), h('td', {class: 'num'}, n0(v.harsh_min))))));
  },
  model(c) {
    const l = c.model.lambda, s = c.model.setpoint_c;
    return card('Your model', `${l.status} · ${l.observations} choices`,
      Object.entries(l.mean).map(([k, v]) => h('div', {class: 'row'}, h('span', {style: 'width:70px'}, k), h('div', {class: 'bar grow'}, h('i', {style: `width:${Math.round(v * 100)}%`})), h('span', {class: 'mono'}, (+v).toFixed(2)))),
      h('div', {class: 'muted', style: 'margin-top:6px'}, `AC setpoint ≈ ${n1(s.mid)}° (${n1(s.low)}–${n1(s.high)}) · ${s.status}`));
  },
  apartment(c) {
    return card('Cooling bill', `listing ${c.listing}`, kv({'Year (data months)': `${n0(c.total.low)}–${n0(c.total.high)} AED`, 'Answers': Object.keys(c.answers || {}).length ? JSON.stringify(c.answers) : '—'}),
      c.question && !Object.keys(c.answers || {}).includes(c.ask) ? h('div', {class: 'col', style: 'margin-top:8px'}, h('div', {}, c.question),
        h('div', {class: 'row wrap'}, (c.options || []).map(o => h('button', {onclick: () => CURE.act('apartment', {answer_key: o.key})}, o.label)))) : null);
  },
  plan(c) {
    const max = Math.max(...c.steps.map(s => s.finish_day), 1);
    return card('Move-in plan', `ready ${c.ready_date.mid} · P(by ${c.target}) ${pct(c.p_by_target)}`,
      h('div', {class: 'timeline'}, c.steps.map(s => h('div', {class: 'tl-row'}, h('span', {}, s.title, s.source_card ? h('span', {class: 'chip ok', title: s.source_card}, 'fact') : null),
        h('div', {class: 'tl-track'}, h('i', {class: s.critical ? 'crit' : '', style: `left:0;width:${(100 * s.finish_day / max).toFixed(1)}%`})), h('span', {class: 'mono num'}, 'd' + n1(s.finish_day))))),
      h('div', {class: 'muted', style: 'margin-top:6px'}, `Critical path: ${c.critical_path.join(' → ')}`));
  },
  plan_change(c) {
    return card('Plan changed', 'visa delay', kv({'Before': `ready ${c.before.ready_date.mid} · P(target) ${pct(c.before.p_by_target)}`,
      'After': `ready ${c.after.ready_date.mid} (${c.after.ready_date.low} – ${c.after.ready_date.high}) · P(target) ${pct(c.after.p_by_target)}`,
      'Temporary housing': n0(c.temporary_housing_aed) + ' AED'}));
  },
  why(c) {
    const chain = c.chain || [];
    return card('Why · causal engine', c.engine ? `${c.engine.facade} · ${c.engine.scm_version}` : '',
      h('div', {class: 'chain'}, chain.map((x, i) => [i ? h('span', {class: 'arrow'}, '→') : null, h('span', {class: 'chip warn'}, x)])),
      c.untouched && c.untouched.length ? h('div', {class: 'muted', style: 'margin-top:6px'}, 'Not affected: ' + c.untouched.join(', ')) : null,
      c.counterfactual ? h('div', {style: 'margin-top:8px'}, `Without the delay: day ${n1(c.counterfactual.without_delay_day)} instead of ${n1(c.counterfactual.with_delay_day)} (−${n1(c.counterfactual.days_lost)} d)`) : null,
      c.levers ? h('div', {class: 'col', style: 'margin-top:8px'}, h('div', {class: 'tag'}, 'what helps · counterfactual per lever'),
        c.levers.map(l => h('div', {class: 'row'}, h('span', {class: 'grow'}, l.title || l.id), h('div', {class: 'bar', style: 'width:120px'}, h('i', {style: `width:${Math.min(100, 30 * l.days_saved)}%`})), h('span', {class: 'mono'}, n1(l.days_saved) + ' d')))) : null,
      c.engine ? h('div', {class: 'dim mono', style: 'margin-top:6px'}, (c.engine.process || []).join(' → ') + ` · real_action=${c.engine.real_action}`) : null);
  },
  counterfactual(c) { return card('Counterfactual', 'abduction → action → prediction', kv({'With delay': 'day ' + n1(c.with_delay_day), 'Without delay': 'day ' + n1(c.without_delay_day), 'Days lost': n1(c.days_lost)})); },
  levers(c) { return R.why({levers: c.levers}); },
  proposal(c) {
    const st = c.result || c.status;
    const label = {needs_you: 'waiting for your yes', ready: 'done by CURE (trusted)', blocked: 'blocked', done: 'done', declined: 'declined'}[st] || st;
    return card('CURE proposes', c.action, h('div', {style: 'font-size:15px'}, c.title),
      h('div', {class: 'row wrap', style: 'margin-top:8px'}, h('span', {class: 'chip ' + (st === 'blocked' ? 'bad' : st === 'needs_you' ? 'warn' : 'ok')}, label),
        c.gate && !c.gate.allowed ? h('span', {class: 'chip bad'}, 'failed: ' + c.gate.failed.join(', ')) : null,
        c.external_call === false ? h('span', {class: 'chip'}, 'no external call') : null),
      c.status === 'needs_you' && !c.result ? h('div', {class: 'row', style: 'margin-top:8px'}, h('button', {class: 'primary', onclick: () => decide(c.id, true)}, 'Approve'), h('button', {onclick: () => decide(c.id, false)}, 'Decline')) : null);
  },
  draft(c) {
    return card('Draft · ' + agentName(c.by), 'agents only draft — you send', h('div', {class: 'quote'}, c.text),
      h('div', {class: 'row', style: 'margin-top:8px'}, h('button', {onclick: () => { navigator.clipboard && navigator.clipboard.writeText(c.text); toast('Copied — send it yourself'); }}, 'Copy'), h('span', {class: 'chip'}, 'CURE never sends on your behalf')));
  },
  fact_card(c) {
    const f = c.card;
    return card('Fact card · agent ' + c.agent, c.status, kv({'Claim': `${f.subject}: ${f.low}–${f.high} ${f.unit || ''}`, 'Source': h('span', {class: 'mono'}, f.source_url), 'Page says': h('i', {}, c.page_text)}),
      c.status === 'rejected' ? h('div', {class: 'chip bad', style: 'margin-top:6px'}, 'rejected: ' + c.reasons.join(', ')) :
        h('div', {class: 'row', style: 'margin-top:8px'}, h('span', {class: 'chip'}, c.status), h('button', {class: 'primary', onclick: () => decide(c.id, true)}, 'Approve → update plan')));
  },
  teacher(c) { return card('Fact → city teacher', c.verdict, kv({'Status': c.status, 'Sent (no personal data)': h('span', {class: 'mono'}, JSON.stringify(c.sent_to_teacher)), 'Compatibility': c.compatibility.result})); },
  effect(c) {
    const e = c.effect || {};
    return card('Executor', c.result, h('div', {}, c.title), h('div', {class: 'row wrap', style: 'margin-top:6px'}, h('span', {class: 'chip ' + (c.result === 'done' ? 'ok' : 'warn')}, c.result), e.external_call === false ? h('span', {class: 'chip'}, 'written to contour log · no external call') : null));
  },
  ics(c) {
    const url = URL.createObjectURL(new Blob([c.ics], {type: 'text/calendar'}));
    return card('Calendar', '.ics', h('a', {href: url, download: 'cure-event.ics', style: 'color:var(--accent)'}, 'Download event (.ics)'));
  },
  trust(c) { return R.trustTable(c.levels); },
  trustTable(levels, withButtons = false) {
    return card('Trust ladder', 'shadow → advises → proposes → acts', levels.map(l => h('div', {style: 'padding:6px 0;border-bottom:1px solid var(--line)'},
      h('div', {class: 'row'}, h('b', {class: 'grow'}, l.action.replace(/_/g, ' ')), h('span', {class: 'chip ' + (l.level === 3 ? 'ok' : l.level === 2 ? 'warn' : '')}, `${l.level} · ${EN.levels[l.level]}`)),
      h('div', {class: 'row', style: 'margin-top:4px'}, h('span', {class: 'dim grow'}, `approvals in a row: ${l.streak}`),
        withButtons ? (l.level === 3 ? h('button', {class: 'danger', onclick: () => CURE.act('revoke_trust', {action: l.action})}, 'Revoke')
          : !(withButtons.autonomy_allowed || []).includes(l.action) ? h('span', {class: 'chip'}, 'always asks you')
          : l.streak >= withButtons.promote_after ? h('button', {onclick: () => CURE.act('grant_trust', {action: l.action})}, 'Allow alone')
          : h('span', {class: 'chip'}, `${withButtons.promote_after - l.streak} more yes to unlock`)) : null))));
  },
  week(c) {
    const people = Object.keys(c.days[0].people);
    return card('Week plan', 'outdoor windows by person · 06–21', h('table', {class: 't'}, h('tr', {}, h('th', {}, ''), people.map(p => h('th', {}, p))),
      c.days.map(d => h('tr', {}, h('td', {}, d.weekday), people.map(p => h('td', {style: 'min-width:70px'}, hoursRow(d.people[p])))))),
      h('div', {class: 'muted', style: 'margin-top:6px'}, `Errands: ${c.errands.order} · ${c.errands.date} ${c.errands.start_at} · Cooling from ${c.home.start_at}`));
  },
  morning(c) {
    const s = c.departures[0];
    return card('Morning', c.date,
      c.departures.map(d => h('div', {class: 'row', style: 'margin:4px 0'}, h('b', {style: 'width:100px'}, EN.trips[d.trip] || d.trip), h('span', {class: 'chip ok'}, `leave ${d.best ? d.best.depart : '—'} · ${pct(d.best && d.best.p_on_time)}`), h('span', {class: 'chip bad'}, `usual ${d.usual.depart} · ${pct(d.usual.p_on_time)}`), h('span', {class: 'muted'}, `by ${d.arrive_by}`))),
      h('div', {class: 'tag', style: 'margin-top:8px'}, 'PM10 by hour'),
      spark(c.pm10_by_hour, {limits: [{v: c.general_limit, c: '#EFB529', label: 'general'}, ...Object.entries(c.limits_pm10).filter(([k, v]) => v < c.general_limit).map(([k, v]) => ({v, c: '#FF3B61', label: k}))]}));
  },
  home(c) { return card('Home on return', 'RP-1R1C physics', kv({'Start cooling': c.start_at, 'You arrive': c.return_at, 'Target': `≤ ${n1(c.target_c)}°`, 'Without cooling': `${n1(mid(c.without_cooling_c))}°`, 'With plan': c.with_plan_c ? `${n1(mid(c.with_plan_c))}°` : '—', 'Off-peak feasible': pct(c.p_off_peak)})); },
  errands(c) { return card('Errands', `${c.date} · ${c.start_at}`, kv({'Order': c.order, 'Separately': `${n0(c.separate_min)} min · ${c.trips.separate} trips`, 'One trip': `${n0(c.chain_min)} min`, 'Saved': n0(c.saved_min) + ' min'})); },
  windows(c) {
    const days = c.family.map(d => d.date);
    return card('Outdoor windows', 'green = inside personal limits', h('table', {class: 't'}, h('tr', {}, h('th', {}, ''), days.map(d => h('th', {}, d.slice(5)))),
      Object.entries(c.people).map(([p, list]) => h('tr', {}, h('td', {}, p), list.map(x => h('td', {}, hoursRow(x.windows))))),
      h('tr', {}, h('td', {}, h('b', {}, 'family')), c.family.map(x => h('td', {}, hoursRow(x.windows))))));
  },
  reminder(c) { return card('Reminder', c.date, json(c)); },
  conflict(c) {
    return card('One car, two meetings', `${c.date} ${c.at}`, h('table', {class: 't'},
      h('tr', {class: c.recommended === 'taxi' ? 'hl' : ''}, h('td', {}, 'Taxi for ' + c.taxi.who), h('td', {}, `${n0(c.taxi.cost_aed.mid)} AED`), h('td', {}, `${n0(c.taxi.wait_outside_min.mid)} min waiting outside`)),
      h('tr', {class: c.recommended === 'move' ? 'hl' : ''}, h('td', {}, `Move "${c.meeting}"`), h('td', {}, `${c.move.from} → ${c.move.to}`), h('td', {}, `${n0(c.move.drive_new_min.mid)} min road instead of ${n0(c.move.drive_now_min.mid)}`))));
  },
  car_taxi(c) {
    const k = Object.keys(c.year);
    return card('Second car or taxi', 'one year · Monte Carlo', h('table', {class: 't'}, h('tr', {}, h('th', {}, ''), h('th', {class: 'num'}, 'AED / year'), h('th', {class: 'num'}, 'waiting h'), h('th', {class: 'num'}, 'heat+dust min')),
      k.map(x => h('tr', {class: x === c.recommended ? 'hl' : ''}, h('td', {}, x), h('td', {class: 'num'}, n0(c.year[x].money_aed.mid)), h('td', {class: 'num'}, n0(c.year[x].waiting_h.mid)), h('td', {class: 'num'}, n0(c.year[x].adam_harsh_min.mid))))),
      c.rules && c.rules.length ? chips(c.rules, 'acc') : null);
  },
  license(c) { return card('Company', 'plan engine', kv({'Licence ≈': c.license_date, 'First hire ≈': c.hire_date, 'Hire lag (median)': n0(c.hire_lag_days) + ' d'})); },
  beach(c) { return card('Beach', 'family window', kv({'Window': `${c.window.from}–${c.window.to}`, 'Reminder': c.remind})); },
  guest(c) {
    return card('Guest arrival', `${c.guest} · ${c.month}`, h('table', {class: 't'}, h('tr', {}, h('th', {}, 'lands'), h('th', {class: 'num'}, 'heat min without CURE'), h('th', {class: 'num'}, 'with CURE'), h('th', {class: 'num'}, 'room ready')),
      c.rows.map(r => h('tr', {class: r.hour === c.best_hour ? 'hl' : ''}, h('td', {}, r.hour), h('td', {class: 'num'}, n0(r.heat_min_without_cure.mid)), h('td', {class: 'num'}, n0(r.heat_min_with_cure)), h('td', {class: 'num'}, pct(r.room_ready_p))))));
  },
  constraints(c) { return card('Sealed + derived', c.who, h('span', {class: 'chip sealed'}, `${c.sealed_items} sealed note(s)`), ' ', chips(Object.entries(c.derived).map(([k, v]) => `${k}=${v}`), 'acc')); },
  budget(c) {
    const cats = Object.keys(c.plan);
    return card('Budget', c.month + ' · plan vs fact', h('table', {class: 't'}, h('tr', {}, h('th', {}, ''), h('th', {class: 'num'}, 'plan'), h('th', {class: 'num'}, 'fact'), h('th', {class: 'num'}, 'diff')),
      cats.map(k => h('tr', {}, h('td', {}, k), h('td', {class: 'num'}, n0(c.plan[k])), h('td', {class: 'num'}, n0(c.fact[k])), h('td', {class: 'num ' + (c.diff[k] <= 0 ? 'pill-ok' : 'pill-bad')}, n0(c.diff[k]))))),
      h('div', {class: 'muted', style: 'margin-top:6px'}, `Summer reserve (cooling): ${n0(c.summer_reserve)} AED · CURE never moves money`));
  },
  habits(c) { return card('Noticed habits', 'model changes only after your yes', c.found.map(f => h('div', {}, f.kind === 'return_shift' ? `${f.weekday}: home ≈ ${f.at} (usual ${f.usual})` : `taxi in peak × ${f.observations}`))); },
  memory(c) { return R.memoryFull(c); },
  memoryFull(m) {
    return h('div', {},
      card('Where it lives', '', kv({'Stored': m.stored_where, 'On disk': h('span', {class: 'mono'}, m.on_disk_looks_like || '—'), 'Devices (windows)': (m.devices || []).join(', ') || '—'})),
      card('Sealed', 'never to the language model or the city', (m.sealed || []).length ? m.sealed.map(s => h('div', {}, h('span', {class: 'chip sealed'}, `${s.who}: ${s.items} sealed`))) : h('span', {class: 'muted'}, 'nothing sealed')),
      card('Derived constraints', 'the only thing planners and the city see', Object.entries(m.constraints || {}).map(([w, v]) => h('div', {}, `${w}: ` + Object.entries(v).map(([k, x]) => `${k}=${x}`).join(', ')))),
      R.model({model: m.model}),
      card('What I remember', `${(m.facts || []).length} records`, h('table', {class: 't'}, (m.facts || []).slice().reverse().slice(0, 14).map(f => h('tr', {}, h('td', {class: 'mono'}, f.at.slice(5, 16).replace('T', ' ')), h('td', {}, f.kind), h('td', {}, f.text), h('td', {class: 'dim'}, f.source))))),
      card('What left the contour', `${(m.left_the_contour || []).length}`, (m.left_the_contour || []).slice(-8).reverse().map(o => h('div', {class: 'mono', style: 'margin:3px 0'}, `${o.at.slice(5, 16)} → ${o.to}: ${typeof o.what === 'string' ? o.what : JSON.stringify(o.what).slice(0, 140)}`))),
      card('Who asked for my data', 'ConsentLedger', (m.requests_to_my_data || []).slice(-6).reverse().map(r => h('div', {class: 'mono'}, `${r.purpose} · ${r.role} · ${r.fields.join(',')} → ${r.allowed ? 'allowed' : 'denied'}`))),
      card('What CURE did', `${(m.effects || []).length}`, (m.effects || []).slice(-8).reverse().map(e => h('div', {}, `✓ ${e.title} `, h('span', {class: 'dim'}, e.external_call ? '' : '(log only)')))),
      card('Consents', '', h('div', {class: 'row wrap'}, (m.consents || []).map(x => h('span', {class: 'chip ' + (x.status === 'active' ? 'acc' : 'bad')}, `${x.purpose}: ${x.status}`)))));
  },
  consents(c) { return card('Consents', '', chips(c.consents.map(x => `${x.purpose}: ${x.status}`), 'acc')); },
  city_request(c) { return card('Request to the city', 'this is ALL the city gets', kv({'Pseudonym': c.pseudonym, 'Objective': c.objective, ...Object.fromEntries(Object.entries(c.constraints).map(([k, v]) => [k, String(v)]))}), h('div', {class: 'chip ok', style: 'margin-top:6px'}, 'no diagnosis · no names')); },
  chain(c) {
    const a = c.arrival;
    return card('Door of the plane → bed', `landing ${String(a.arrival_hour).padStart(2, '0')}:00 · vision · not executed`,
      h('table', {class: 't'}, c.links.map(l => h('tr', {}, h('td', {}, l.title), h('td', {class: 'mono'}, l.action), h('td', {}, h('span', {class: 'chip ' + (l.status === 'ok' ? 'ok' : 'warn')}, l.status))))),
      kv({'Outside without CURE': `${n0(a.without.outdoor_min.mid)} min (dust ${n0(a.without.dust_min.mid)})`, 'With CURE': `${n0(a.with.outdoor_min.mid)} min`, 'PM10 at landing': a.pm10_at_arrival}));
  },
  storm(c) {
    return card('Dust storm', 'safe mode', h('div', {class: 'row wrap'}, h('span', {class: 'chip bad'}, `driverless: ${c.av.allowed ? 'allowed' : 'stopped'}`), Object.entries(c.agents_mode).map(([k, v]) => h('span', {class: 'chip'}, `${k}: ${v}`))),
      h('table', {class: 't', style: 'margin-top:6px'}, c.links.map(l => h('tr', {}, h('td', {}, l.title), h('td', {}, h('span', {class: 'chip ' + (l.status === 'ok' ? 'ok' : 'warn')}, l.status))))),
      h('div', {class: 'muted', style: 'margin-top:6px'}, 'Constraint kept: ' + JSON.stringify(c.constraint_kept)));
  },
  neighbors(c) {
    return card('Neighbours · arbitration', c.emergency ? 'emergency weights' : 'normal weights', kv({'Chosen': c.chosen_note, 'Rejected': Object.entries(c.infeasible).map(([k, v]) => h('div', {}, `${c.notes[k] || k}: ${v.join(', ')}`))}));
  },
  cohort(c) {
    return card(`${c.families} families this week`, c.stress ? `stress · ramp cars ${c.ramp_cars}` : `ramp cars ${c.ramp_cars}`,
      h('div', {class: 'row wrap'}, h('span', {class: 'chip ok'}, `limit kept with CURE ${c.limits.kept.with}/${c.limits.families_with_personal_limit}`), h('span', {class: 'chip bad'}, `without ${c.limits.kept.without}/${c.limits.families_with_personal_limit}`),
        h('span', {class: 'chip'}, `wasted trips ${c.centers.wasted_trips.without} → ${c.centers.wasted_trips.with}`), h('span', {class: 'chip'}, `hot homes ${c.grid.hot_home_arrivals.without} → ${c.grid.hot_home_arrivals.with}`)),
      h('table', {class: 't', style: 'margin-top:8px'}, h('tr', {}, h('th', {}, 'family (label for viewers)'), h('th', {}, 'sent to city'), h('th', {class: 'num'}, 'harsh min w/o'), h('th', {class: 'num'}, 'with'), h('th', {}, 'limit')),
        c.rows.map(r => h('tr', {}, h('td', {}, r.who), h('td', {class: 'mono'}, Object.entries(r.sent_to_city).map(([k, v]) => `${k}=${v}`).join(' ')), h('td', {class: 'num'}, n0(r.harsh_min.without)), h('td', {class: 'num'}, n0(r.harsh_min.with)),
          h('td', {}, r.limit_kept == null ? '—' : h('span', {class: 'chip ' + (r.limit_kept.with ? 'ok' : 'bad')}, r.limit_kept.with ? 'kept' : 'broken'))))));
  },
  learning(c) { return card('City teacher', 'facts, not weights', h('table', {class: 't'}, c.steps.map(s => h('tr', {}, h('td', {}, s.title), h('td', {class: 'num'}, s.facts), h('td', {class: 'num'}, pct(s.share_from_facts)), h('td', {class: 'dim'}, s.origin))))); },
  city_family(c) { return card('Your family in the city', c.id, kv({'Limits sent': JSON.stringify(c.limits), 'Arrival': `day ${c.arrival_day}, ${c.arrival_h}:00`})); },
  impact(c) { return card('Model comparison', '', kv({'Days earlier': `${n1(c.days_saved.mid)} (${n1(c.days_saved.low)}–${n1(c.days_saved.high)})`})); },
  deleted(c) { return card('Contour deleted', c.resident, h('span', {class: 'chip ' + (c.file_exists ? 'bad' : 'ok')}, c.file_exists ? 'file still exists' : 'file erased')); },
  refused(c) { return card('Not enough data', 'no guessing', h('div', {}, (c.answer.missing || []).join(', '))); },
  device(c) { return card('Opened on another device', '', kv({'Windows': c.devices.join(', '), 'On disk (ciphertext)': h('span', {class: 'mono'}, c.on_disk_looks_like || '—')})); },
  official_search(c) { return card('Official sites only', '', chips(c.sources, 'ok')); },
};
function renderCard(c) {
  try { return (R[c.type] || (x => card(x.type || 'card', '', json(x))))(c); }
  catch (e) { return card(c.type || 'card', 'render error', json(c)); }
}

// ---------------------------------------------------------------- chat
function renderConversation(el, conv, {cards = true, onQuick} = {}) {
  el.innerHTML = '';
  conv.forEach((m, i) => {
    const b = h('div', {class: 'msg ' + (m.role === 'you' ? 'you' : 'cure')}, m.role === 'cure' ? h('div', {class: 'who'}, 'CURE') : null, m.text);
    el.append(b);
    if (cards && m.cards) m.cards.forEach(c => {
      if (!R[c.type] || ['door', 'memory', 'model', 'profile_intake', 'error'].includes(c.type)) return;
      el.append(renderCard(c));
    });
    if (i === conv.length - 1 && m.quick && m.quick.length && onQuick) el.append(h('div', {class: 'quick'}, m.quick.map(q => h('button', {onclick: () => onQuick(q)}, q))));
  });
  el.scrollTop = el.scrollHeight;
}

// ---------------------------------------------------------------- voice (browser speech API when available)
function micButton(input, onDone, langFn) {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const b = h('button', {title: 'Speak', 'aria-label': 'Speak', class: 'ghost mic-button'});
  b.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M6 10v2a6 6 0 0 0 12 0v-2M12 18v3M9 21h6"/></svg>';
  if (!SR) { b.disabled = true; b.title = 'Speech recognition is not available in this browser — type instead'; return b; }
  b.onclick = () => {
    const r = new SR(); r.lang = (langFn && langFn()) === 'ru' ? 'ru-RU' : 'en-US'; r.interimResults = true;
    b.classList.add('is-listening'); r.onresult = e => { input.value = Array.from(e.results).map(x => x[0].transcript).join(' '); };
    r.onend = () => { b.classList.remove('is-listening'); if (input.value.trim()) onDone(input.value.trim()); }; r.start();
  };
  return b;
}
function speak(text, lang) {
  if (!window.speechSynthesis || !CURE.voiceOn) return;
  const u = new SpeechSynthesisUtterance(text); u.lang = lang === 'ru' ? 'ru-RU' : 'en-US'; speechSynthesis.cancel(); speechSynthesis.speak(u);
}

return {Q, CURE, EN, agentName, h, n0, n1, pct, mid, toast, card, kv, json, chips, win, spark, hoursRow, decide, R, renderCard, renderConversation, micButton, speak};
}
