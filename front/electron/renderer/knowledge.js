// Present confirmed backend fields and saved facts as readable notes.
export function knowledgeEntries(state, memory) {
  const entries = [];
  if (state.household?.length) entries.push({
    title: 'Family',
    text: state.household.map(person => `${person.name} · ${person.role}`).join('\n'),
    kind: 'Saved',
  });
  const labels = {
    household_size: ['Household', value => `${value} people moving`],
    children_count: ['Children', value => `${value} children moving`],
    is_founder: ['Work', value => value ? 'Building a company' : 'Not building a company'],
    arrival_date: ['Arrival', value => value],
    housing_budget_aed_month: ['Housing budget', value => `${Number(value).toLocaleString('en-US')} AED / month`],
    preferred_setpoint_c: ['At home', value => `Preferred temperature: ${value}°C`],
    return_home_hour: ['Daily routine', value => `Usually home at ${String(Math.floor(value)).padStart(2, '0')}:${String(Math.round((value % 1) * 60)).padStart(2, '0')}`],
  };
  for (const [key, field] of Object.entries(state.profile || {})) {
    if (!['known', 'derived'].includes(field.status) || field.value == null) continue;
    const [title, format] = labels[key] || [key.replaceAll('_', ' '), value => String(value)];
    entries.push({title, text: format(field.value), kind: field.status === 'derived' ? 'Inferred' : 'Confirmed'});
  }
  if (memory.district) entries.push({title: 'Neighbourhood', text: `Selected district: ${memory.district}`, kind: 'Saved'});
  if (memory.arrival && !entries.some(entry => entry.title === 'Arrival')) entries.push({title: 'Arrival', text: memory.arrival, kind: 'Saved'});
  for (const fact of [...(memory.facts || [])].reverse()) {
    if (!fact.text) continue;
    entries.push({title: fact.kind === 'agent_finding' ? 'Specialist finding' : 'Saved note', text: fact.text,
      kind: fact.status === 'pending' ? 'Needs review' : fact.provenance === 'inference' ? 'Inference' : 'Remembered', at: fact.at});
  }
  return entries;
}
