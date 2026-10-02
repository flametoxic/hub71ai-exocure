// A canvas particle globe with readable, orbiting cards from the backend week.
export function mountWeekGlobe(host, week, name, h, onOptimize) {
  const canvas = h('canvas', {'aria-hidden': 'true'});
  const period = week.days?.length ? `${week.days[0].date} — ${week.days.at(-1).date}` : 'Weekly plan unavailable';
  host.append(h('div', {class: 'globe-heading'},
    h('div', {class: 'globe-title-row'}, h('span', {}, `${name}’s week`),
      h('button', {class: 'globe-optimize', onclick: onOptimize, 'aria-label': 'Optimize weekly family plan'}, 'Optimize')),
    h('small', {}, period)), canvas);
  const details = h('section', {class: 'globe-details', 'aria-label': 'Selected day details', 'aria-live': 'polite'},
    h('div', {class: 'globe-details-hint'}, 'Select a day on the globe to explore your plan.'));
  host.after(details);
  const formatWindows = windows => windows?.length ? windows.map(w => `${w.from}–${w.to}`).join(' · ') : 'No outdoor window available';
  const clock = value => { const minutes = Math.round(Number(value) * 60); return `${String(Math.floor(minutes / 60)).padStart(2, '0')}:${String(minutes % 60).padStart(2, '0')}`; };
  const eventTitle = event => event.en || event.title || event.id;
  const issueTitles = {person_overlap: 'Overlapping commitments', car_conflict: 'Family car conflict', docs_not_ready: 'Documents not ready', unsafe_outdoor: 'Outdoor conditions conflict'};
  const personLabel = id => id === 'family' ? 'Whole family' : week.people?.[id] || id;
  function selectDay(day, node) {
    host.querySelectorAll('.globe-plan').forEach(card => {
      const selected = card === node;
      card.classList.toggle('selected', selected);
      card.setAttribute('aria-pressed', String(selected));
    });
    const date = new Date(`${day.date}T12:00:00Z`);
    const weekday = date.getUTCDay();
    const errands = week.errands?.date === day.date ? week.errands : null;
    details.dataset.date = day.date;
    details.replaceChildren(
      h('div', {class: 'globe-details-label'}, 'YOUR DAY'),
      h('h3', {}, date.toLocaleDateString('en-GB', {weekday: 'long', day: 'numeric', month: 'long', timeZone: 'UTC'})),
      day.events ? h('div', {class: 'globe-detail-group week-timeline'}, h('h4', {}, 'Your schedule'),
        day.events.map(event => h('article', {class: 'week-event'},
          h('time', {}, `${clock(event.start)}–${clock(event.end)}`), h('strong', {}, eventTitle(event)),
          h('p', {}, `${(event.who || []).map(personLabel).join(', ')} · ${week.places?.[event.where] || event.where || ''}`),
          h('span', {class: 'mind-provenance'}, event.provenance === 'synthetic' ? 'Demo' : event.provenance === 'user' ? 'You' : 'Calculated'),
          event.docs_ready === false ? h('span', {class: 'week-conflict'}, 'Documents not ready') : null,
          event.needs_car ? h('span', {class: 'mind-provenance'}, 'Car needed') : null))) : null,
      day.issues?.length ? h('div', {class: 'globe-detail-group week-issues'}, h('h4', {}, 'Needs attention'),
        day.issues.map(issue => h('article', {class: 'week-issue'}, h('strong', {}, issueTitles[issue.kind] || issue.kind.replaceAll('_', ' ')),
          h('p', {}, (issue.evidence_event_ids || []).map(id => eventTitle(day.events.find(event => event.id === id) || {id})).join(' · ')),
          issue.people?.length ? h('p', {class: 'globe-detail-muted'}, issue.people.map(personLabel).join(', ')) : null))) : null,
      !day.events ? h('div', {class: 'globe-detail-group'}, h('h4', {}, 'Family time'), h('p', {}, formatWindows(day.family))) : null,
      !day.events ? h('div', {class: 'globe-detail-group'}, h('h4', {}, 'Outdoor windows by person'),
        h('dl', {}, Object.entries(day.people || {}).map(([person, windows]) =>
          h('div', {class: 'globe-person-window'}, h('dt', {}, person), h('dd', {}, formatWindows(windows)))))) : null,
      errands ? h('div', {class: 'globe-detail-group'}, h('h4', {}, 'Family errands'),
        h('p', {}, errands.order), h('p', {class: 'globe-detail-muted'}, `Start at ${errands.start_at}${errands.chain_min != null ? ` · ${Math.round(errands.chain_min)} min route` : ''}`)) : null,
      week.home && weekday > 0 && weekday < 6 ? h('div', {class: 'globe-detail-group'}, h('h4', {}, 'Home comfort'),
        h('p', {}, `Start cooling at ${week.home.start_at}`),
        h('p', {class: 'globe-detail-muted'}, `Ready at ${week.home.return_at} · target ${week.home.target_c}°C`)) : null);
    const screen = host.parentElement;
    const delta = details.getBoundingClientRect().top - screen.getBoundingClientRect().top - screen.clientHeight * .52;
    screen.scrollTo({top: Math.max(0, screen.scrollTop + delta), behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
  }
  const cards = (week.days || []).map((day, i) => {
    const date = new Date(`${day.date}T12:00:00Z`);
    const label = date.toLocaleDateString('en-GB', {weekday: 'short', day: 'numeric', month: 'short', timeZone: 'UTC'});
    const windows = day.family.map(w => `${w.from}–${w.to}`).join(' · ');
    const errands = week.errands?.date === day.date ? `${week.errands.start_at} · ${week.errands.order}` : null;
    const cooling = date.getUTCDay() > 0 && date.getUTCDay() < 6 && week.home ? `Home ready at ${week.home.return_at} · ${week.home.target_c}°C` : null;
    const node = h('article', {class: 'globe-plan', role: 'button', tabindex: '0', 'aria-pressed': 'false',
      'aria-label': `View plan for ${label}`, style: `--plan-color:${day.issues?.length || errands ? '#ffc086' : '#a3ceff'}`,
      onclick: () => selectDay(day, node),
      onkeydown: event => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectDay(day, node); }
      }},
      h('div', {class: 'globe-day'}, label),
      h('strong', {}, day.events ? `${day.events.length} plans` : errands ? 'Family errands' : 'Family time'),
      h('p', {}, day.events ? day.events.slice(0, 2).map(eventTitle).join(' · ') : errands || (windows ? `Outdoors ${windows}` : 'Indoor time · no shared outdoor window')),
      day.issues?.length ? h('small', {class: 'week-conflict'}, `${day.issues.length} conflicts to review`) : cooling ? h('small', {}, cooling) : null);
    host.append(node);
    const latitude = .8 - 1.6 * (i + .5) / week.days.length;
    const ringRadius = Math.sqrt(1 - latitude * latitude);
    const angle = i * 2.399963229728653;
    return {node, x: Math.cos(angle) * ringRadius, y: latitude, z: Math.sin(angle) * ringRadius};
  });
  if (!cards.length) host.append(h('div', {class: 'globe-empty', role: 'status'}, 'No weekly plan available for this profile yet.'));
  host.append(h('div', {class: 'globe-caption'}, 'FAMILY · WORK · WELLBEING', h('small', {}, week.data_mode === 'synthetic' ? 'Demo schedule · from your family memory' : 'Your family schedule')));
  const ctx = canvas.getContext('2d');
  const points = Array.from({length: 1050}, (_, i) => {
    const y = 1 - 2 * (i + .5) / 1050, radius = Math.sqrt(1 - y * y), angle = i * 2.399963229728653;
    return {x: Math.cos(angle) * radius, y, z: Math.sin(angle) * radius, warm: i % 13 === 0};
  });
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  let frame, disposed = false, width = 0, height = 0;
  const observer = new ResizeObserver(() => {
    width = host.clientWidth; height = host.clientHeight;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = width * dpr; canvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (reduced.matches) { cancelAnimationFrame(frame); frame = requestAnimationFrame(paint); }
  });
  observer.observe(host);
  const start = performance.now();
  function paint(now) {
    if (disposed) return;
    if (width && height) {
      const rotation = reduced.matches ? .4 : .4 + (now - start) * .00016;
      const radius = Math.min(width * .42, height * .29), cx = width / 2, cy = height * .49;
      const project = (x, y, z) => {
        const rx = x * Math.cos(rotation) + z * Math.sin(rotation);
        const rz = z * Math.cos(rotation) - x * Math.sin(rotation);
        const ry = y * .96 - rz * .28, depth = y * .28 + rz * .96;
        return {x: cx + rx * radius, y: cy + ry * radius, z: depth};
      };
      ctx.clearRect(0, 0, width, height);
      const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius * 1.4);
      glow.addColorStop(0, '#719dff50'); glow.addColorStop(.6, '#416bcb28'); glow.addColorStop(1, '#14204400');
      ctx.fillStyle = glow; ctx.fillRect(0, 0, width, height);
      for (const ring of [.18, -.24]) {
        ctx.beginPath();
        for (let i = 0; i <= 150; i++) {
          const a = i * Math.PI * 2 / 150, p = project(Math.cos(a) * 1.15, ring, Math.sin(a) * 1.15);
          if (i) ctx.lineTo(p.x, p.y); else ctx.moveTo(p.x, p.y);
        }
        ctx.strokeStyle = '#b3d5ff70'; ctx.lineWidth = .9; ctx.setLineDash([1, 4]); ctx.stroke(); ctx.setLineDash([]);
      }
      for (const point of points.map(p => ({...project(p.x, p.y, p.z), warm: p.warm})).sort((a, b) => a.z - b.z)) {
        ctx.globalAlpha = .38 + (point.z + 1) * .31;
        ctx.fillStyle = point.warm ? '#ffc392' : '#b0d7ff';
        ctx.beginPath(); ctx.arc(point.x, point.y, point.warm ? 1.6 : 1.1, 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;
      cards.forEach(({node, x, y, z}) => {
        // Share the sphere's rotation and tilt; keep labels facing the reader.
        const anchor = project(x, y, z);
        const depth = Math.max(0, Math.min(1, (anchor.z + 1) / 2));
        const scale = .72 + depth * .28;
        const px = cx + (anchor.x - cx) * .68;
        const cardHeight = node.offsetHeight;
        const py = Math.max(100 + cardHeight / 2, Math.min(height - 72 - cardHeight / 2, cy + (anchor.y - cy) * 1.38));
        node.style.transform = `translate(${px - 62}px, ${py - cardHeight / 2}px) scale(${scale})`;
        node.style.zIndex = String(2 + Math.round(depth * 8));
        node.style.opacity = String(.34 + Math.pow(depth, 1.5) * .66);
        node.style.filter = `blur(${(1 - depth) * .5}px)`;
        ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(anchor.x, anchor.y);
        ctx.strokeStyle = `rgba(116,157,255,${.06 + depth * .18})`; ctx.lineWidth = .7; ctx.stroke();
        ctx.beginPath(); ctx.arc(anchor.x, anchor.y, 2, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(144,185,255,${.2 + depth * .7})`; ctx.fill();
      });
    }
    if (!reduced.matches || !width) frame = requestAnimationFrame(paint);
  }
  frame = requestAnimationFrame(paint);
  const onMotionChange = () => { cancelAnimationFrame(frame); frame = requestAnimationFrame(paint); };
  reduced.addEventListener('change', onMotionChange);
  return () => { disposed = true; cancelAnimationFrame(frame); observer.disconnect(); reduced.removeEventListener('change', onMotionChange); details.remove(); };
}
