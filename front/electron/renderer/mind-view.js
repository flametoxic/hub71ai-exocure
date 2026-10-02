const TERMINAL = new Set(['COMPLETED', 'WAITING_USER', 'FAILED']);
const label = value => String(value || '').replaceAll('_', ' ').toLowerCase();
const provenanceLabel = value => ({synthetic: 'Demo', core: 'Calculated', user: 'You', agent: 'Specialist', inference: 'Inferred', assumption: 'Assumption'}[value] || label(value));
function readableMemory(memory) {
  const text = memory.text || '';
  const start = text.indexOf('{');
  if (start < 0) return text;
  try {
    const data = JSON.parse(text.slice(start));
    const names = {pm10_ugm3_max: 'PM10 limit', outdoor_harsh_min_max: 'Time outdoors in harsh conditions',
      arrive_by: 'Arrive by', usual_depart: 'Usual departure', from: 'From', to: 'To', who: 'Family members',
      start_at: 'Start', return_at: 'Return', target_c: 'Target temperature'};
    const valueText = value => value && typeof value === 'object' ? (Array.isArray(value) ? value.map(valueText).join(', ') :
      Object.entries(value).map(([key, item]) => `${names[key] || key.replaceAll('_', ' ')}: ${valueText(item)}`).join(' · ')) : String(value);
    const prefix = text.slice(0, start).replace(/\s*\([^)]*\)/g, '').replace('dust/outdoor hard constraints:', 'outdoor limits:').trim();
    return `${prefix}${prefix ? ' ' : ''}${valueText(data)}`;
  } catch { return text; }
}
function sourceLink(url, h) {
  try {
    const source = new URL(url);
    if (source.protocol === 'https:' || source.protocol === 'http:') {
      return h('a', {href: source.href, target: '_blank', rel: 'noopener noreferrer'}, source.hostname);
    }
  } catch {}
  return null;
}

export function agentCards(agents, {h, card, CURE}, onChange = async () => {}) {
  return agents.filter(agent => agent.status !== 'ARCHIVED').map(agent => {
    const error = h('p', {class: 'mind-error', role: 'alert'});
    async function perform(button, route, body) {
      const controls = [...button.parentElement.querySelectorAll('button')];
      controls.forEach(control => { control.disabled = true; }); error.textContent = '';
      try { await CURE.api(route, body, true); await onChange(); }
      catch (failure) { error.textContent = failure.message; controls.forEach(control => { control.disabled = false; }); }
    }
    const result = agent.result;
    const findings = agent.review?.items || result?.findings || [];
    return card('Specialist', label(agent.status),
      h('h3', {class: 'specialist-task'}, agent.task),
      h('div', {class: 'agent-stages'}, (agent.history || []).map(step => h('span', {}, label(step.stage)))),
      agent.error ? h('p', {class: 'mind-error'}, agent.error) : null,
      result?.summary ? h('p', {}, result.summary) : null,
      result?.checklist?.length ? h('ul', {}, result.checklist.map(text => h('li', {}, text))) : null,
      findings.map(finding => {
        const buttons = h('div', {class: 'row mind-actions'});
        const decision = finding.decision || 'pending';
        if (finding.finding_id && decision === 'pending' && ['COMPLETED', 'WAITING_USER'].includes(agent.status)) {
          for (const [action, text] of [['approve', 'Accept task'], ['reject', 'Dismiss']]) {
            const button = h('button', {class: action === 'approve' ? 'primary' : '', onclick: () =>
              perform(button, `/agents/dynamic/${agent.agent_id}/review`,
                {rid: CURE.rid, device: CURE.device, decisions: [{finding_id: finding.finding_id, decision: action}]})}, text);
            buttons.append(button);
          }
        }
        return h('article', {class: 'agent-finding'}, h('p', {}, finding.text), sourceLink(finding.source_url, h),
          h('span', {class: 'mind-provenance'}, {pending: 'Needs your review', approve: 'Accepted', reject: 'Dismissed'}[decision] || label(decision)), buttons);
      }),
      result?.questions?.map(question => h('p', {}, question)),
      h('p', {class: 'mind-help'}, 'Accepted findings become tasks. They do not change your personal constraints.'),
      TERMINAL.has(agent.status) ? (() => {
        const button = h('button', {class: 'ghost', onclick: () => perform(button,
          `/agents/dynamic/${agent.agent_id}/archive`, {rid: CURE.rid, device: CURE.device})}, 'Archive specialist');
        return button;
      })() : null, error);
  });
}

export function mountAgentFeed(host, shared) {
  let disposed = false, busy = false, previous = '';
  async function refresh() {
    if (disposed || busy || !shared.CURE.rid) return;
    busy = true;
    try {
      const result = await shared.CURE.agents();
      if (disposed || !host.isConnected) return;
      const next = JSON.stringify(result.agents);
      if (next !== previous) { previous = next; host.replaceChildren(...agentCards(result.agents || [], shared, async () => { previous = ''; await refresh(); await shared.CURE.kick(); })); }
    } catch { /* Connection status and history are handled by the parent screen. */ }
    finally { busy = false; }
  }
  refresh();
  const timer = setInterval(refresh, 2500);
  return () => { disposed = true; clearInterval(timer); };
}

export function mountMindView(host, shared) {
  const {h, card, CURE} = shared;
  let disposed = false, busy = false, current = null, fingerprint = '', inferring = false, starting = false;
  const status = h('p', {class: 'mind-help', role: 'status'}, 'Loading your memory…');
  const situation = h('div', {}), tasks = h('div', {}), memories = h('div', {}), inferences = h('div', {}), agents = h('div', {});
  const query = h('input', {placeholder: 'Search your memories…', 'aria-label': 'Search memories'});
  const recalled = h('div', {class: 'mind-recall-results', 'aria-live': 'polite'});
  const recallButton = h('button', {class: 'primary', onclick: async () => {
    if (!query.value.trim()) return;
    recallButton.disabled = true;
    try {
      const answer = await CURE.api('/mind/recall', {rid: CURE.rid, query: query.value.trim(), embeddings: current?.mode === 'openai'}, true);
      if (disposed) return;
      recalled.replaceChildren(h('p', {class: 'mind-help'}, answer.embedding_error ? 'Using local memory search.' : `${answer.memories?.length || 0} matching memories`),
        ...(answer.memories || []).map(memory => memoryCard(memory)));
    } catch (error) { recalled.replaceChildren(h('p', {class: 'mind-error', role: 'alert'}, error.message)); }
    finally { recallButton.disabled = false; }
  }}, 'Search');
  query.addEventListener('keydown', event => { if (event.key === 'Enter') { event.preventDefault(); recallButton.click(); } });
  const kind = h('select', {'aria-label': 'Inference type'},
    ['routine', 'availability', 'preference', 'current_situation'].map(value => h('option', {value}, label(value))));
  const inferButton = h('button', {disabled: true, onclick: async () => {
    inferring = true; inferButton.disabled = true;
    try { await CURE.api('/mind/infer', {rid: CURE.rid, kind: kind.value, query: query.value.trim() || 'Current family schedule and previous decisions'}, true); fingerprint = ''; await refresh(); }
    catch (error) { status.textContent = error.message; status.className = 'mind-error'; }
    finally { inferring = false; inferButton.disabled = current?.mode !== 'openai'; }
  }}, 'Find insights');
  const task = h('textarea', {rows: 2, placeholder: 'Describe a task for a specialist…', 'aria-label': 'Specialist task'});
  const startButton = h('button', {disabled: true, class: 'primary', onclick: async () => {
    if (!task.value.trim()) return;
    starting = true; startButton.disabled = true;
    try { await CURE.api('/agents/dynamic', {rid: CURE.rid, text: task.value.trim(), device: CURE.device}, true); task.value = ''; fingerprint = ''; await refresh(); }
    catch (error) { status.textContent = error.message; status.className = 'mind-error'; }
    finally { starting = false; updateStart(); }
  }}, 'Start specialist');
  function updateStart() {
    startButton.disabled = starting || current?.mode !== 'openai' || !task.value.trim() || (current?.agents || []).some(agent => !TERMINAL.has(agent.status) && agent.status !== 'ARCHIVED');
  }
  task.addEventListener('input', updateStart);
  const aiHint = h('p', {class: 'mind-help'});
  host.append(status, situation, card('Memory search', '', h('div', {class: 'row'}, query, recallButton), recalled), tasks,
    card('Insights', 'Inferences, not confirmed facts', h('div', {class: 'row wrap'}, kind, inferButton), inferences),
    card('Specialists', 'Research and drafts', task, h('div', {class: 'mind-actions'}, startButton), aiHint), agents, memories);
  function memoryCard(memory) {
    return h('article', {class: 'mind-memory'}, h('p', {}, readableMemory(memory)),
      h('span', {class: 'mind-provenance'}, provenanceLabel(memory.provenance)),
      memory.status && memory.status !== 'active' ? h('span', {class: 'mind-provenance'}, memory.status === 'pending' ? 'Needs review' : label(memory.status)) : null);
  }
  async function refresh() {
    if (disposed || busy) return;
    busy = true;
    try {
      const mind = await CURE.mind();
      if (disposed || !host.isConnected) return;
      current = mind;
      const next = JSON.stringify(mind);
      if (next === fingerprint) return;
      fingerprint = next;
      status.className = 'mind-help'; status.textContent = `${mind.memories?.length || 0} memories · ${mind.tasks?.length || 0} tasks`;
      const now = mind.current_situation || {};
      situation.replaceChildren(card('Your situation', '',
        h('p', {}, now.district ? `District ${now.district}` : 'District not selected'),
        h('p', {}, now.arrival ? `Arrival ${now.arrival}` : 'Arrival date not set'),
        h('p', {class: 'mind-help'}, now.scenario_date_text || now.scenario_clock),
        (mind.goals || []).map(goal => h('p', {}, goal.title))));
      tasks.replaceChildren(card('Life tasks', '', (mind.tasks || []).length ?
        mind.tasks.map(item => h('article', {class: 'mind-task'}, h('strong', {}, item.title),
          h('span', {class: `task-status status-${item.status}`}, label(item.status)),
          item.blockers?.length ? h('p', {class: 'mind-help'}, 'Waiting for: ' + item.blockers.join(', ').replaceAll('_', ' ')) : null,
          h('span', {class: 'mind-provenance'}, provenanceLabel(item.provenance)))) : h('p', {class: 'mind-help'}, 'No tasks yet.')));
      inferences.replaceChildren(...(mind.recent_inferences || []).map(item => h('article', {class: 'mind-memory'},
        h('p', {}, item.assessment), h('span', {class: 'mind-provenance'}, `${label(item.kind)} · ${Math.round(item.confidence * 100)}% confidence`),
        item.assumptions?.length ? h('p', {class: 'mind-help'}, 'Assumptions: ' + item.assumptions.join('; ')) : null)));
      inferButton.disabled = inferring || mind.mode !== 'openai';
      aiHint.textContent = mind.mode === 'openai' ? 'Specialists research tasks and return findings for your review.' : 'Insights and specialists require AI mode. Memory search is available locally.';
      updateStart();
      agents.replaceChildren(...agentCards(mind.agents || [], shared, async () => { fingerprint = ''; await refresh(); await CURE.kick(); }));
      memories.replaceChildren(card('Saved evidence', 'Source labels preserved', (mind.memories || []).slice(-20).reverse().map(memoryCard)));
    } catch (error) { if (!disposed) { status.className = 'mind-error'; status.textContent = error.message; } }
    finally { busy = false; }
  }
  refresh();
  const timer = setInterval(refresh, 3000);
  return () => { disposed = true; clearInterval(timer); };
}
