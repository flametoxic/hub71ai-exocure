// Change this file when connecting the new backend. Shared by browser and Electron.
export const backendContract = {
  requestTimeoutMs: 90000,
  startupTimeoutMs: 30000,
  routes: [
    ['GET', '/health', '/health'],
    ['GET', '/state', '/twin/state'],
    ['GET', '/memory', '/twin/memory'],
    ['GET', '/week', '/twin/mind'],
    ['GET', '/mind', '/twin/mind'],
    ['POST', '/mind/recall', '/twin/mind/recall'],
    ['POST', '/mind/infer', '/twin/mind/infer'],
    ['GET', '/agents/dynamic', '/twin/agents/dynamic'],
    ['POST', '/agents/dynamic', '/twin/agents/dynamic'],
    ['GET', '/agents/dynamic/:id', '/twin/agents/dynamic/:id'],
    ['POST', '/agents/dynamic/:id/review', '/twin/agents/dynamic/:id/review'],
    ['POST', '/agents/dynamic/:id/archive', '/twin/agents/dynamic/:id/archive'],
    ['GET', '/moments', '/twin/moments'],
    ['GET', '/why/:id', '/twin/why/:id'],
    ['POST', '/reset', '/twin/reset'],
    ['POST', '/door', '/twin/door'],
    ['POST', '/bootstrap', '/twin/bootstrap'],
    ['POST', '/ask', '/twin/ask'],
    ['POST', '/act', '/twin/act'],
    ['POST', '/device/open', '/twin/device/open'],
    ['POST', '/moment/:id', '/twin/moment/:id'],
  ],
};

export function resolveAPI(route, body) {
  const url = new URL(route, 'http://cure-api');
  if (!route.startsWith('/') || route.startsWith('//') || url.origin !== 'http://cure-api' || url.hash) {
    throw new Error('Invalid API path');
  }
  const method = body === undefined ? 'GET' : 'POST';
  for (const [verb, logical, endpoint] of backendContract.routes) {
    if (verb !== method) continue;
    const match = url.pathname.match(new RegExp('^' + logical.replace(':id', '([a-zA-Z0-9_-]+)') + '$'));
    if (match) return {method, path: endpoint.replace(':id', match[1] || '') + url.search};
  }
  throw new Error('Unknown CURE API route');
}

// Adapt payloads here once the new backend's actual schema is available.
// Both transports use the same hooks; UI components keep their existing data shape.
export function prepareRequest(route, body) { return {route, body}; }
export function adaptResponse(route, data) {
  if (route.split('?')[0] !== '/week') return data;
  const events = data.week?.events || [];
  const dates = [...new Set(events.map(event => event.date))].sort();
  const places = Object.fromEntries((data.week?.places || []).map(place => [place.id, place.en || place.id]));
  const people = Object.fromEntries((data.household || []).map(person => [person.id.replace(/^person:/, ''), person.title]));
  return {
    days: dates.map(date => ({date, events: events.filter(event => event.date === date).sort((a, b) => a.start - b.start),
      issues: (data.week?.issues || []).filter(issue => issue.date === date), family: [], people: {}})),
    places, people, tasks: data.tasks || [], mode: data.mode,
    data_mode: events.length && events.every(event => event.provenance === 'synthetic') ? 'synthetic' : 'user',
  };
}
export function isBackendReady(health, state) {
  return health?.ready === true && typeof state?.clock === 'string' && !!state?.city;
}
