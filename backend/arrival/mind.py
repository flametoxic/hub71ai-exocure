"""Evidence and inference views of the existing encrypted resident Contour."""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from .config import Settings
from .person import sanitize_for_llm
from .language import GatewayError


WEEK_TASK_LINKS = {
    'mon-eid': 'emirates_id',
    'tue-med': 'medical',
    'tue-bank': 'bank',
    'wed-adgm': 'company_license',
    'thu-lease': 'tenancy',
    'thu-addc': 'utilities',
}


def _iso_date(value):
    return value.isoformat() if hasattr(value, 'isoformat') else str(value)


def _event_people(event, household_ids):
    people = []
    for who in event.get('who', []):
        for person in (household_ids if who == 'family' else [who]):
            if person in household_ids and person not in people:
                people.append(person)
    return people


def _hour(value):
    hours, minutes = value.split(':')
    return int(hours) + int(minutes) / 60


def _week_issues(session, events):
    """Find only conflicts proven by the schedule or existing CURE calculations."""
    household = [person['id'] for person in session.c.profile_doc.get('household', [])]
    issues = []

    def add(kind, evidence, **details):
        evidence = sorted(evidence)
        issues.append({'id': stable_id('week-issue', [kind, evidence, details]), 'kind': kind,
                       'evidence_event_ids': evidence, 'provenance': 'core', **details})

    expanded = {event['id']: _event_people(event, household) for event in events}
    for index, left in enumerate(events):
        for right in events[index + 1:]:
            if left['date'] != right['date'] or left['start'] >= right['end'] or right['start'] >= left['end']:
                continue
            shared = sorted(set(expanded[left['id']]) & set(expanded[right['id']]))
            if shared:
                add('person_overlap', [left['id'], right['id']], date=left['date'], people=shared)
            if left.get('needs_car') and right.get('needs_car'):
                add('car_conflict', [left['id'], right['id']], date=left['date'], resource='family_car')

    for event in events:
        if event.get('docs_ready') is False:
            add('docs_not_ready', [event['id']], date=event['date'])

    try:
        from . import life
        safe = life.outdoor_windows(profile=session.profile(), policy=session.P, week=session.S['week'])
        by_person = {person: {_iso_date(day['date']): day['windows'] for day in days}
                     for person, days in safe['people'].items()}
        for event in events:
            if not event.get('outdoor'):
                continue
            unsafe = []
            for person in expanded[event['id']]:
                windows = by_person.get(person, {}).get(event['date'], [])
                if not any(event['start'] >= _hour(window['from']) and event['end'] <= _hour(window['to'])
                           for window in windows):
                    unsafe.append(person)
            if unsafe:
                add('unsafe_outdoor', [event['id']], date=event['date'], people=sorted(unsafe),
                    calculation='life.outdoor_windows')
    except (KeyError, ValueError):
        # No issue is emitted when the existing climate/core model cannot prove it.
        pass
    return sorted(issues, key=lambda item: (item['date'], item['kind'], item['id']))


def sync_week_events(session):
    """Merge synthetic week observations into Leila's encrypted Contour by event.id."""
    if session.rid != 'leila' or 'week_events' not in session.S:
        return {'events': {}, 'places': {}, 'issues': []}
    source = session.S['week_events']
    events = {}
    for raw in source.raw('events'):
        event = {key: raw.get(key) for key in
                 ('id', 'date', 'start', 'end', 'who', 'where', 'kind', 'needs_car', 'outdoor',
                  'docs_ready', 'movable', 'en', 'ru')}
        event['date'] = _iso_date(event['date'])
        event['who'] = list(event.get('who') or [])
        event['provenance'] = 'synthetic'
        events[event['id']] = event
    places = {place_id: {'id': place_id, 'en': value.get('en', place_id), 'ru': value.get('ru', place_id),
                         'provenance': 'synthetic'}
              for place_id, value in source.raw('places').items()}
    state = {'events': events, 'places': places, 'issues': _week_issues(session, list(events.values())),
             'meta': {'data_mode': 'synthetic', 'source': source.raw('meta').get('source')}}
    if session.c.extra.get('week_mind') != state:
        session.c.extra['week_mind'] = state
        session.c.extra['rev'] = session.c.extra.get('rev', 0) + 1
        session.save()
    return session.c.extra['week_mind']


class Inference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["routine", "availability", "preference", "current_situation"]
    likely_activity: Literal["school_run", "work", "at_home", "outdoors", "unknown"]
    assessment: str = Field(max_length=600)
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(max_length=12)
    assumptions: list[str] = Field(max_length=8)


def stable_id(prefix, value):
    return prefix + ':' + hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _terms(text):
    text = text.casefold()
    for word, replacement in {"адам": "adam son", "пыл": "dust", "школ": "school", "лейл": "leila",
                              "сын": "son adam", "жар": "heat", "деньг": "money", "предпочтен": "preference"}.items():
        text = re.sub(word + r'\w*', replacement, text)
    words = set(re.findall(r'[\w]+', text))
    return words - {'and', 'the', 'with', 'what', 'was', 'there', 'about', 'и', 'что', 'там', 'было', 'с', 'про'}


class Mind:
    def __init__(self, session, settings=None, client=None):
        self.s = session
        self.settings = settings or Settings.from_env()
        self.client = client

    def state(self):
        return self.s.c.extra.setdefault('mind', {'version': 1, 'inferences': [], 'embedding_cache': {}, 'last_trace': {}})

    def _client(self):
        if self.settings.language_mode != 'openai' or not self.settings.openai_api_key:
            raise GatewayError('openai_required', 'Mind inference requires OpenAI mode and OPENAI_API_KEY')
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.settings.openai_api_key, timeout=60, max_retries=0)
        return self.client

    def memories(self):
        s, items = self.s, []
        def add(text, provenance, about=(), at=None, importance=0.5, original_source=None,
                item_id=None, kind='memory', metadata=None):
            clean = sanitize_for_llm(str(text), s.P)
            if not clean or clean == '[private note omitted]':
                return
            items.append({'id': item_id or stable_id('memory', [clean, provenance, at]), 'text': clean,
                          'provenance': provenance, 'original_source': original_source,
                          'about': list(about), 'at': at, 'importance': importance,
                          'kind': kind, 'metadata': metadata or {}})
        for person in s.c.profile_doc.get('household', []):
            add(f"{s.name(person['id'])} ({person['id']}): {person['role']}",
                'synthetic' if s.rid == 'leila' else 'user', [person['id']])
        for who, constraints in s.c.constraints.items():
            add(f"{s.name(who)} ({who}) dust/outdoor hard constraints: {json.dumps(constraints)}",
                'core', [who, 'dust', 'outdoors'], importance=1)
        for note in s.c.extra.get('notes', []):
            add(note.get('fact', ''), 'synthetic' if s.rid == 'leila' else 'user',
                [note.get('about', s.rid), note.get('category', 'note')], original_source=note.get('source'))
        for trip in s.c.profile_doc.get('trips', []):
            add(json.dumps(trip), 'synthetic', trip.get('who', []) + [trip['from'], trip['to'], 'routine'])
        for fact in s.c.facts[-80:]:
            if fact.get('kind') == 'sealed':
                continue
            source = fact.get('source', 'core')
            provenance = 'user' if source in {'you', 'user', 'chat'} else ('agent' if source.startswith('agent:') else 'core')
            provenance = fact.get('provenance', provenance)
            add(fact.get('text', ''), provenance, [s.rid, fact.get('kind', 'note')], fact.get('at'), original_source=source)
            if items and items[-1]['text'] == sanitize_for_llm(fact.get('text', ''), s.P):
                items[-1]['status'] = fact.get('status', 'active')
        for decision in s.c.model.history[-20:]:
            add(decision['text'], 'user', [s.rid, 'decision', 'preference'], decision.get('at'), importance=0.8)
        for inference in self.state()['inferences'][-12:]:
            add(inference['assessment'], 'inference', [s.rid, inference['kind']], inference['at'])
        week = sync_week_events(s)
        household_ids = [person['id'] for person in s.c.profile_doc.get('household', [])]
        for event in week['events'].values():
            people = _event_people(event, household_ids)
            add(event.get(s.lang) or event.get('en') or event['id'], 'synthetic',
                people + [event['where'], event['kind']], event['date'] + f"T{int(event['start']):02d}:00:00+00:00",
                importance=.65, original_source='week_events.yaml', item_id='event:' + event['id'], kind='event',
                metadata=event)
        return list({item['id']: item for item in items}.values())

    def snapshot(self):
        s = self.s
        memories = self.memories()
        week = sync_week_events(s)
        steps = [step for step in s.S['steps'].raw('steps')
                 if all(s.c.profile_doc.get('flags', {}).get(k) == v for k, v in step.get('applies_if', {}).items())]
        completed = s.c.extra.get('completed_steps', {})
        tasks = [{'id': x['id'], 'title': s.t(x), 'blockers': [b for b in x.get('blocks_on', []) if b not in completed],
                  'status': 'completed' if x['id'] in completed else ('blocked' if any(b not in completed for b in x.get('blocks_on', [])) else 'available'),
                  'provenance': 'user' if x['id'] in completed else 'synthetic',
                  'affects_calculations': True} for x in steps]
        agent_tasks = [dict(task) for task in s.c.extra.get('agent_tasks', {}).values()]
        tasks += agent_tasks
        household_nodes = [{'id': 'person:' + p['id'], 'kind': 'person', 'title': s.name(p['id']),
                            'provenance': 'synthetic' if s.rid == 'leila' else 'user'}
                           for p in s.c.profile_doc.get('household', [])]
        nodes = list(household_nodes)
        nodes += [{'id': m['id'], 'kind': m.get('kind', 'memory'), 'title': m['text'],
                   'provenance': m['provenance'], **({'event': m['metadata']} if m.get('kind') == 'event' else {})}
                  for m in memories]
        nodes += [{'id': 'task:' + t['id'], 'kind': 'task', 'title': t['title'], 'provenance': t['provenance']} for t in tasks]
        edges = [{'from': m['id'], 'to': 'person:' + who, 'kind': 'about', 'provenance': m['provenance']}
                 for m in memories for who in m['about'] if any(p['id'] == who for p in s.c.profile_doc.get('household', []))]
        edges += [{'from': 'task:' + t['id'], 'to': 'task:' + b, 'kind': 'blocked_by', 'provenance': 'synthetic'} for t in tasks for b in t['blockers']]
        nodes.append({'id':'goal:settle', 'kind':'goal', 'title':'Complete relocation and settle the family',
                      'provenance':'synthetic' if s.rid == 'leila' else 'assumption'})
        edges += [{'from':'goal:settle', 'to':'task:' + task['id'], 'kind':'requires', 'provenance':'synthetic'} for task in tasks]
        for person in s.c.profile_doc.get('household', [])[1:]:
            edges.append({'from':'person:' + s.c.profile_doc['household'][0]['id'], 'to':'person:' + person['id'],
                          'kind':'family_member', 'provenance':'synthetic' if s.rid == 'leila' else 'user'})
        places = list(s.c.extra.get('places', [])) + list(week['places'].values())
        for place in {place['id']: place for place in places}.values():
            nodes.append({'id':'place:' + place['id'], 'kind':'place', 'title':place.get(s.lang, place.get('en',place['id'])),
                          'provenance':place.get('provenance', 'synthetic')})
        household_ids = [person['id'] for person in s.c.profile_doc.get('household', [])]
        topic_ids = set()
        task_ids = {task['id'] for task in tasks}
        for event in week['events'].values():
            event_id = 'event:' + event['id']
            for person in _event_people(event, household_ids):
                edges.append({'from': 'person:' + person, 'to': event_id, 'kind': 'participates_in',
                              'provenance': 'synthetic'})
            edges.append({'from': event_id, 'to': 'place:' + event['where'], 'kind': 'at',
                          'provenance': 'synthetic'})
            topic_id = 'topic:' + event['kind']
            if topic_id not in topic_ids:
                topic_ids.add(topic_id)
                nodes.append({'id': topic_id, 'kind': 'topic', 'title': event['kind'], 'provenance': 'synthetic'})
            edges.append({'from': event_id, 'to': topic_id, 'kind': 'kind', 'provenance': 'synthetic'})
            linked_task = WEEK_TASK_LINKS.get(event['id'])
            if linked_task in task_ids:
                edges.append({'from': event_id, 'to': 'task:' + linked_task, 'kind': 'related_to',
                              'provenance': 'synthetic'})
        for issue in week['issues']:
            nodes.append({'id': issue['id'], 'kind': 'issue', 'title': issue['kind'], 'provenance': 'core',
                          'issue': issue})
            for event_id in issue['evidence_event_ids']:
                edges.append({'from': issue['id'], 'to': 'event:' + event_id, 'kind': 'evidence',
                              'provenance': 'core'})
        for agent in s.c.extra.get('dynamic_agents', {}).values():
            nodes.append({'id':agent['agent_id'], 'kind':'agent', 'title':agent['task'], 'provenance':'agent'})
            for memory in memories:
                if memory.get('original_source') == 'agent:' + agent['agent_id']:
                    edges.append({'from':agent['agent_id'], 'to':memory['id'], 'kind':'finding', 'provenance':'agent'})
        source_nodes = set()
        for task in agent_tasks:
            source_id = stable_id('source', task['source_url'])
            if source_id not in source_nodes:
                source_nodes.add(source_id)
                nodes.append({'id': source_id, 'kind': 'source', 'title': urlsplit(task['source_url']).hostname,
                              'url': task['source_url'], 'provenance': 'official'})
            edges.extend([
                {'from': task['agent_id'], 'to': 'task:' + task['id'], 'kind': 'proposed_task', 'provenance': 'agent'},
                {'from': 'task:' + task['id'], 'to': task['evidence_memory_id'], 'kind': 'evidence', 'provenance': 'agent'},
                {'from': 'task:' + task['id'], 'to': source_id, 'kind': 'supported_by', 'provenance': 'official'},
            ])
        graph = {'nodes': nodes, 'edges': edges}
        if self.state().get('graph') != graph:
            self.state()['graph'] = graph
            s.save()
        return {'identity': {'resident_id': s.rid, 'name': s.name(s.c.profile_doc['household'][0]['id'])},
                'household': household_nodes,
                'current_situation': {'scenario_clock': s.now.isoformat(), 'current_hour': s.now.hour,
                                      'scenario_date_text': s.now.strftime('%d %B %Y'),
                                      'weekday': s.now.strftime('%A'), 'arrival': s.c.extra.get('arrival'),
                                      'district': s.c.extra.get('district'), 'delays': s.c.extra.get('delays', {}),
                                      'completed_steps': completed, 'provenance': 'core'},
                'goals': [{'id': 'settle', 'title': 'Complete relocation and settle the family', 'provenance': 'synthetic' if s.rid == 'leila' else 'assumption'}],
                'tasks': tasks, 'constraints': s.c.constraints, 'preferences': s.c.model.summary(),
                'decisions': s.c.model.history[-20:], 'memories': memories,
                'recent_inferences': self.state()['inferences'][-8:],
                'agents': list(s.c.extra.get('dynamic_agents', {}).values()),
                'week': {'events': list(week['events'].values()), 'places': list(week['places'].values()),
                         'issues': week['issues'], 'event_count': len(week['events'])},
                'graph': graph, 'last_trace': self.state()['last_trace'],
                'mode': self.settings.language_mode}

    def recall(self, query, *, embeddings=False, limit=8):
        memories = self.memories()
        clean_query = sanitize_for_llm(query, self.s.P)
        terms = _terms(clean_query)
        graph_model = self.snapshot()['graph']
        adjacency = {}
        for edge in graph_model['edges']:
            adjacency.setdefault(edge['from'], set()).add(edge['to'])
            adjacency.setdefault(edge['to'], set()).add(edge['from'])
        matched_nodes = {node['id'] for node in graph_model['nodes']
                         if terms & _terms(' '.join([node.get('id', ''), node.get('title', '')]))}
        distances = {node_id: 0 for node_id in matched_nodes}
        frontier = set(matched_nodes)
        for distance in (1, 2):
            following = {target for node_id in frontier for target in adjacency.get(node_id, ())
                         if target not in distances}
            distances.update({node_id: distance for node_id in following})
            frontier = following
        semantic = {}
        mode, error = 'local_graph_lexical', None
        if embeddings and memories:
            try:
                model = self.settings.openai_embed_model
                cache = self.state()['embedding_cache']
                missing = [m for m in memories if model + ':' + m['id'] not in cache]
                response = self._client().embeddings.create(model=model, input=[clean_query] + [m['text'] for m in missing])
                vectors = [x.embedding for x in sorted(response.data, key=lambda x: x.index)]
                for m, vector in zip(missing, vectors[1:]):
                    cache[model + ':' + m['id']] = vector
                q = vectors[0]
                for m in memories:
                    v = cache[model + ':' + m['id']]
                    semantic[m['id']] = sum(a*b for a,b in zip(q,v)) / (math.sqrt(sum(a*a for a in q)*sum(a*a for a in v)) or 1)
                valid_keys = {model + ':' + m['id'] for m in memories}
                self.state()['embedding_cache'] = {k:v for k,v in cache.items() if k in valid_keys}
                mode = 'openai_embeddings_graph'
            except Exception as exc:
                error = type(exc).__name__
        now = self.s.now
        for m in memories:
            lexical = len(terms & _terms(m['text'])) / max(len(terms), 1)
            about_graph = min(1, len(terms & _terms(' '.join(m['about']))) / 2)
            distance = distances.get(m['id'])
            reachable_graph = {0: 1.0, 1: .75, 2: .5}.get(distance, 0)
            graph = max(about_graph, reachable_graph)
            age = abs((now - datetime.fromisoformat(m['at'])).total_seconds()) / 86400 if m.get('at') else 90
            situation = 1 if self.s.c.extra.get('topic') in m['about'] else 0
            m.update(semantic_score=round(semantic.get(m['id'], lexical), 4), graph_relevance=graph,
                     graph_distance=distance)
            m['score'] = round(.5*m['semantic_score'] + .25*graph + .1*m['importance'] + .1/(1+age) + .05*situation, 4)
        matches = sorted(memories, key=lambda m: m['score'], reverse=True)[:limit]
        self.state()['last_trace'] = {'operation': 'recall', 'mode': mode, 'query': clean_query,
                                    'memory_ids': [m['id'] for m in matches], 'embedding_error': error,
                                    'graph_hops': 2, 'matched_node_ids': sorted(matched_nodes),
                                    'reachable_node_ids': sorted(distances)}
        self.s.save()
        return {'mode': mode, 'memories': matches, 'embedding_error': error,
                'graph': {'hops': 2, 'matched_node_ids': sorted(matched_nodes),
                          'reachable_node_ids': sorted(distances)}}

    def context(self, query='', *, embeddings=False):
        if 'arrival_planning' in self.s.c.revoked:
            raise GatewayError('consent_required', 'Resident planning consent is revoked')
        snapshot = self.snapshot()
        recall = self.recall(query, embeddings=embeddings)
        relevant = set(recall['graph']['reachable_node_ids']) | {memory['id'] for memory in recall['memories']}
        relevant_nodes = [node for node in snapshot['graph']['nodes'] if node['id'] in relevant][:40]
        relevant_ids = {node['id'] for node in relevant_nodes}
        relevant_edges = [edge for edge in snapshot['graph']['edges']
                          if edge['from'] in relevant_ids and edge['to'] in relevant_ids][:60]
        evidence_events = {node_id.removeprefix('event:') for node_id in relevant_ids if node_id.startswith('event:')}
        issues = [issue for issue in snapshot['week']['issues']
                  if set(issue['evidence_event_ids']) & evidence_events][:12]
        return sanitize_for_llm({'identity': snapshot['identity'], 'current_situation': snapshot['current_situation'],
                                'goals': snapshot['goals'], 'hard_constraints': snapshot['constraints'],
                                'preferences': snapshot['preferences'],
                                'memories': recall['memories'], 'week_issues': issues,
                                'relevant_subgraph': {'nodes': relevant_nodes, 'edges': relevant_edges}}, self.s.P)

    def infer(self, kind='routine', query='Current family schedule and previous decisions'):
        context = self.context(query, embeddings=True)
        try:
            response = self._client().responses.create(model=self.settings.openai_model, store=False,
                reasoning={'effort': self.settings.reasoning_effort},
                instructions='Infer only from the supplied evidence. This is an inference, never a user fact. '
                             'Respect the scenario clock. Return unknown with confidence 0 and no evidence if insufficient. '
                             'Never invent measurements, money, dates, permissions or relax hard constraints. '
                             'Respect each memory about/person binding: do not transfer a husband or child routine to Leila. '
                             'Use exactly the requested kind and valid memory IDs. Explain in the resident language.',
                input=json.dumps({'kind': kind, 'language': self.s.lang, 'context': context}, ensure_ascii=False),
                text={'format': {'type': 'json_schema', 'name': 'mind_inference', 'strict': True, 'schema': Inference.model_json_schema()}})
            candidate = Inference.model_validate_json(response.output_text)
            ids = {m['id'] for m in context['memories']}
            if candidate.kind != kind or not set(candidate.evidence_ids) <= ids:
                raise ValueError('invalid inference evidence or kind')
            if not candidate.evidence_ids:
                candidate.likely_activity, candidate.confidence = 'unknown', 0
            record = sanitize_for_llm(candidate.model_dump(), self.s.P)
            from .agent_runtime import _validate_draft_text
            _validate_draft_text(record['assessment'], context)
            record.update(id=stable_id('inference', [response.id, record]), at=self.s.now.isoformat(),
                          provenance='inference', provider='openai', model=self.settings.openai_model, response_id=response.id)
            self.state()['inferences'] = (self.state()['inferences'] + [record])[-40:]
            self.state()['last_trace'] = {'operation': 'inference', 'response_id': response.id,
                                         'context': context, 'memory_ids': candidate.evidence_ids}
            self.s.save()
            return record
        except GatewayError:
            raise
        except Exception as error:
            raise GatewayError('mind_inference_failed', type(error).__name__) from error
