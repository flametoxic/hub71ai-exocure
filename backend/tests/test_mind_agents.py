import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from arrival.config import Settings
from arrival.service import Runtime, leila_contour
from arrival.mind import Mind
from arrival.person import sanitize_for_llm
from arrival.dynamic_agents import FORBIDDEN, SpecProposal, validate_spec, start, _prepare_action_proposals
from arrival.gateway import allowed_source
from arrival.language import GatewayError
from arrival.language import QueryPlan, ToolCall
from arrival.modeling import ModelRegistry, PlanValidator
from arrival.agents import sign
from arrival import dynamic_agents


@pytest.fixture
def session(tmp_path, monkeypatch):
    for k,v in {'ARRIVAL_CORE_DIR':str(tmp_path), 'ARRIVAL_CORE_KEY':'test-core-key-long-enough',
                'ARRIVAL_VAULT_KEY':'test-vault-key-long-enough', 'ARRIVAL_TEMPLATE_KEY':'demo-template-key-change-me',
                'CURE_LANGUAGE_MODE':'local'}.items():
        monkeypatch.setenv(k,v)
    rt = Runtime()
    return leila_contour(rt, rt.S['leila'].get('voice_example'), 'en', 'phone')


def test_real_runtime_restart_retains_mind_and_forget_removes_it(session):
    mind = Mind(session)
    mind.state()['inferences'].append({'assessment':'test', 'kind':'routine', 'at':session.now.isoformat()})
    session.save()
    restarted = Runtime().session('leila')
    assert restarted.c.extra['mind']['inferences'][0]['assessment'] == 'test'
    restarted.delete()
    assert Runtime().session('leila') is None


def test_outbound_boundary_removes_nested_health_conversation_and_secrets(session):
    payload = {'sealed':{'son':'asthma'}, 'nested':{'diagnosis':'private', 'note':'my son has asthma',
               'api_key':'sk-private', 'other':'sk-secret-value'}, 'conversation':[{'text':'private'}],
               'hard_constraints':session.c.constraints}
    clean = sanitize_for_llm(payload, session.P)
    assert 'asthma' not in json.dumps(clean)
    assert 'sk-' not in json.dumps(clean)
    assert 'sealed' not in clean and 'conversation' not in clean
    assert clean['hard_constraints']['son']['pm10_ugm3_max'] == 50


def test_adam_dust_recall_has_graph_and_provenance(session):
    mind = Mind(session)
    recalled = mind.recall('Что там было с Адамом и пылью?')
    assert recalled['memories'][0]['about'] == ['son','dust','outdoors']
    assert recalled['memories'][0]['provenance'] == 'core'
    assert mind.snapshot()['graph']['edges']
    assert 'asthma' not in json.dumps(mind.context('Adam'))


def test_week_events_are_idempotent_persistent_and_graph_connected(session):
    mind = Mind(session)
    first = mind.snapshot()
    second = mind.snapshot()
    assert first['week']['event_count'] == second['week']['event_count'] == 46
    assert len(session.c.extra['week_mind']['events']) == 46
    assert all(event['provenance'] == 'synthetic' for event in first['week']['events'])
    nodes = {node['id']: node for node in first['graph']['nodes']}
    edges = first['graph']['edges']
    assert len([node for node in nodes.values() if node['kind'] == 'event']) == 46
    assert len([node for node in nodes.values() if node['kind'] == 'place']) >= 15
    assert any(edge['from'] == 'person:son' and edge['to'] == 'event:mon-sports'
               and edge['kind'] == 'participates_in' for edge in edges)
    assert any(edge['from'] == 'event:mon-sports' and edge['to'] == 'place:school'
               and edge['kind'] == 'at' for edge in edges)
    assert any(edge['from'] == 'event:mon-eid' and edge['to'] == 'task:emirates_id'
               and edge['kind'] == 'related_to' for edge in edges)
    session.rt.restart_process()
    restarted = session.rt.session('leila')
    assert len(restarted.c.extra['week_mind']['events']) == 46


def test_week_core_finds_schedule_car_docs_and_outdoor_issues(session):
    issues = Mind(session).snapshot()['week']['issues']
    assert any(issue['kind'] == 'person_overlap' and
               set(issue['evidence_event_ids']) == {'mon-inv', 'mon-eid'} for issue in issues)
    assert any(issue['kind'] == 'car_conflict' and issue['date'] == '2026-11-26' and
               set(issue['evidence_event_ids']) == {'thu-client', 'thu-inv'} for issue in issues)
    assert any(issue['kind'] == 'docs_not_ready' and issue['evidence_event_ids'] == ['mon-eid'] for issue in issues)
    assert any(issue['kind'] == 'unsafe_outdoor' and issue['evidence_event_ids'] == ['mon-sports'] and
               issue['calculation'] == 'life.outdoor_windows' for issue in issues)
    assert all('requirement' not in issue for issue in issues if issue['kind'] == 'docs_not_ready')


def test_graph_expansion_recalls_event_without_literal_dust_match(session):
    recalled = Mind(session).recall('Adam and dust', limit=20)
    event = next(item for item in recalled['memories'] if item['id'] == 'event:mon-sports')
    assert 'dust' not in event['text'].casefold()
    assert event['graph_relevance'] > 0
    # Dust alone reaches the event through constraint memory -> Adam -> event in two hops.
    dust_only = Mind(session).recall('dust', limit=20)
    event = next(item for item in dust_only['memories'] if item['id'] == 'event:mon-sports')
    assert event['semantic_score'] == 0
    assert event['graph_distance'] == 2
    assert 'event:mon-sports' in dust_only['graph']['reachable_node_ids']


def test_dynamic_action_suggestions_wait_for_executor_review(session):
    run = {'agent_id': 'specialist-actions', 'spec': {'scope': 'family'}, 'response_id': 'response-test',
           'review': {'items': [{'finding_id': 'finding-1'}]},
           'result': {'proposed_actions': [
               {'action': 'checklist', 'title': 'Review checklist', 'when': None, 'duration_min': None},
               {'action': 'reminder', 'title': 'Check cat documents',
                'when': '2026-11-25T18:00:00+04:00', 'duration_min': None},
               {'action': 'calendar_event', 'title': 'Vet appointment',
                'when': '2026-11-26T10:00:00+04:00', 'duration_min': 30}]}}
    prepared = _prepare_action_proposals(session, run)
    assert prepared[0]['status'] == 'review_required'
    pending = [item for item in prepared if item['status'] == 'proposal_pending']
    assert {item['action'] for item in pending} == {'reminder', 'calendar_event'}
    assert len(session.c.pending) == 2 and session.c.effects == []
    calendar = next(item for item in pending if item['action'] == 'calendar_event')
    session.decide(calendar['proposal_id'], True)
    assert session.c.effects[-1]['action'] == 'calendar_event'
    assert 'BEGIN:VCALENDAR' in session.c.effects[-1]['ics']


def test_dynamic_action_without_schedule_is_honestly_blocked(session):
    run = {'agent_id': 'specialist-actions', 'spec': {'scope': 'family'}, 'review': {'items': []},
           'result': {'proposed_actions': [{'action': 'reminder', 'title': 'Follow up',
                                            'when': None, 'duration_min': None}]}}
    assert _prepare_action_proposals(session, run) == [
        {'action': 'reminder', 'status': 'blocked', 'reason': 'missing_or_invalid_schedule',
         'executor_supported': True}]
    assert session.c.pending == {} and session.c.effects == []


def test_completion_unlocks_dependencies_and_recalculates(session):
    before = session._plan()['value']['ready_days']['mid']
    session.complete_step('emirates_id')
    after = session._plan()['value']['ready_days']['mid']
    assert after < before
    tasks = {t['id']:t for t in Mind(session).snapshot()['tasks']}
    assert tasks['emirates_id']['status'] == 'completed'
    assert tasks['bank']['status'] == 'available'
    assert tasks['tenancy']['status'] == 'available'
    assert Runtime().session('leila').c.extra['completed_steps']['emirates_id']['provenance'] == 'user'


def test_visa_delay_moves_dependent_steps(session):
    before = session._plan()['value']
    after = session._plan({'entry_permit':14})['value']
    before_steps = {s['id']:s['finish_day']['mid'] for s in before['steps']}
    after_steps = {s['id']:s['finish_day']['mid'] for s in after['steps']}
    assert after_steps['emirates_id'] - before_steps['emirates_id'] == pytest.approx(14)


def test_spec_permissions_and_sources_cannot_expand(session):
    context = Mind(session).context('unknown task')
    spec = {'allowed_tools':['search_public'], 'scope':'family', 'allowed_sources':['u.ae'],
            'context_memory_ids':[], 'writeback_permissions':['pending_findings'], 'forbidden_actions':FORBIDDEN}
    validate_spec(session, spec, context)
    for field,value in [('allowed_tools',['transfer_money']), ('scope','admin'), ('allowed_sources',['evil.example']),
                        ('context_memory_ids',['invented']), ('writeback_permissions',['profile'])]:
        altered = copy.deepcopy(spec)
        altered[field] = value
        with pytest.raises(ValueError):
            validate_spec(session, altered, context)
    assert not allowed_source('https://u.ae.evil.example/page', ['u.ae'])
    assert not allowed_source('https://evil.example/?source=u.ae', ['u.ae'])
    assert allowed_source('https://www.u.ae/page', ['u.ae'])


def test_unknown_task_local_mode_is_honest(session):
    with pytest.raises(GatewayError, match='live OpenAI'):
        start(session, 'An unprepared task')


def test_inference_rejects_unknown_evidence_without_mutating_constraints(session):
    constraints = copy.deepcopy(session.c.constraints)
    response = SimpleNamespace(output_text=json.dumps({'kind':'routine', 'likely_activity':'school_run',
            'assessment':'School run', 'confidence':.8, 'evidence_ids':['invented'], 'assumptions':[]}), id='test')
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs:response))
    mind = Mind(session, Settings(language_mode='openai', openai_api_key='placeholder'), client)
    with pytest.raises(GatewayError):
        mind.infer()
    assert session.c.constraints == constraints
    assert mind.state()['inferences'] == []


def test_restart_marks_interrupted_agent_failed(session):
    session.c.extra['dynamic_agents'] = {'x':{'status':'RUNNING','history':[]}}
    session.save()
    assert Runtime().session('leila').c.extra['dynamic_agents']['x']['status'] == 'FAILED'


def test_model_cannot_store_an_invented_user_budget(session):
    plan = QueryPlan(source='openai', operations=[ToolCall(name='record_event', arguments={
        'variable_id':'housing_budget_aed_month', 'observations':{'value':99999}})])
    outcome = PlanValidator(ModelRegistry.load_default()).validate(plan, session, user_text='Show my plan')
    assert outcome.status == 'clarification_required'
    assert session.c.profile_doc['params']['budget_aed_month']['value'] == 18000


def test_agent_result_requires_actual_search_source_binding(session):
    # A structured model result alone does not constitute public research.
    proposal = {'goal':'Research an unknown task', 'context_memory_ids':[], 'allowed_tools':['search_public'],
                'allowed_sources':['u.ae'], 'scope':'family', 'success_criteria':['Sourced checklist'],
                'max_steps':2}
    research = {'status':'completed', 'summary':'Checklist',
                'findings':[{'text':'Check official requirements', 'source_url':'https://u.ae/page'}],
                'checklist':['Check requirements'], 'questions':[], 'proposed_actions':['checklist']}
    replies = iter([SimpleNamespace(output_text=json.dumps(proposal), id='spec'),
                    SimpleNamespace(output_text=json.dumps(research), id='result', model_dump=lambda:{'output':[]})])
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs:next(replies)))
    run = start(session, 'Research an unknown task', settings=Settings(language_mode='openai', openai_api_key='placeholder'),
                client=client, background=False)
    assert run['status'] == 'FAILED'
    assert run['error'] == 'no completed official-source search'
    assert not any(f['kind'] == 'agent_finding' for f in session.c.facts)
    spec = run['spec']
    assert spec['signature'] == sign(spec, b'demo-template-key-change-me')
    spec['allowed_tools'].append('transfer_money')
    assert spec['signature'] != sign(spec, b'demo-template-key-change-me')


def test_consent_revocation_blocks_context_and_agent(session):
    session.c.revoked.append('arrival_planning')
    with pytest.raises(GatewayError, match='consent'):
        Mind(session).context('family')
    with pytest.raises(GatewayError, match='consent'):
        start(session, 'unknown task', settings=Settings(language_mode='openai', openai_api_key='placeholder'))


def _pending_review_run(session, *, agent_id='specialist-review-test'):
    spec = {'agent_id': agent_id, 'task': 'Research an unknown task', 'goal': 'Build a sourced checklist',
            'allowed_tools': ['search_public', 'draft_text'], 'allowed_sources': ['u.ae'], 'scope': 'family',
            'context_memory_ids': [], 'context_slice': {}, 'hard_constraints': {},
            'forbidden_actions': FORBIDDEN, 'success_criteria': ['Sourced checklist'], 'max_steps': 2,
            'timeout': 180, 'output_schema': {}, 'writeback_permissions': ['pending_findings']}
    spec['signature'] = sign(spec, b'demo-template-key-change-me')
    finding_id = 'finding-review-test'
    source_url = 'https://u.ae/information-and-services'
    fact = session.c.remember(at=session.now, kind='agent_finding', text='Check the official import requirements',
                              source='agent:' + agent_id, status='pending')
    fact.update(finding_id=finding_id, agent_id=agent_id, source_url=source_url, provenance='agent')
    run = {'agent_id': agent_id, 'task': spec['task'], 'status': 'WAITING_USER', 'history': [], 'spec': spec,
           'result': {'status': 'waiting_user', 'summary': 'Research result', 'checklist': [], 'questions': [],
                      'proposed_actions': ['checklist'], 'findings': [
                          {'finding_id': finding_id, 'text': fact['text'], 'source_url': source_url}]},
           'sources': [{'type': 'url', 'url': source_url}], 'writeback_status': 'pending_findings',
           'review': {'status': 'pending', 'items': [{'finding_id': finding_id, 'text': fact['text'],
                      'source_url': source_url, 'decision': 'pending'}]},
           'provenance': 'agent'}
    session.c.extra.setdefault('dynamic_agents', {})[agent_id] = run
    session.save()
    return run, finding_id


def test_reviewed_agent_finding_becomes_sourced_mind_task_without_changing_calculation(session):
    run, finding_id = _pending_review_run(session)
    ready_before = session._plan()['value']['ready_days']['mid']
    dynamic_agents.review(session, run['agent_id'], [{'finding_id': finding_id, 'decision': 'approve'}])
    # Repeating the same review is safe and does not duplicate the task or decision memory.
    dynamic_agents.review(session, run['agent_id'], [{'finding_id': finding_id, 'decision': 'approve'}])

    snapshot = Mind(session).snapshot()
    task = next(task for task in snapshot['tasks'] if task.get('finding_id') == finding_id)
    assert task['provenance'] == 'agent'
    assert task['reviewed_by'] == 'user'
    assert task['source_url'].startswith('https://u.ae/')
    assert task['affects_calculations'] is False
    assert session._plan()['value']['ready_days']['mid'] == ready_before
    assert sum(task.get('finding_id') == finding_id for task in session.c.extra['agent_tasks'].values()) == 1
    assert next(f for f in session.c.facts if f.get('finding_id') == finding_id)['status'] == 'reviewed'
    assert any(edge['from'] == 'task:' + task['id'] and edge['kind'] == 'supported_by'
               for edge in snapshot['graph']['edges'])
    assert run['writeback_status'] == 'reviewed'
    restarted = Runtime().session('leila')
    assert any(task.get('finding_id') == finding_id for task in restarted.c.extra['agent_tasks'].values())


def test_review_rechecks_signature_and_reject_does_not_create_task(session):
    run, finding_id = _pending_review_run(session)
    run['spec']['allowed_tools'].append('transfer_money')
    with pytest.raises(ValueError, match='signature'):
        dynamic_agents.review(session, run['agent_id'], [{'finding_id': finding_id, 'decision': 'approve'}])
    assert session.c.extra.get('agent_tasks', {}) == {}

    run['spec']['allowed_tools'].pop()
    run['spec']['signature'] = sign(run['spec'], b'demo-template-key-change-me')
    dynamic_agents.review(session, run['agent_id'], [{'finding_id': finding_id, 'decision': 'reject'}])
    assert session.c.extra.get('agent_tasks', {}) == {}
    assert next(f for f in session.c.facts if f.get('finding_id') == finding_id)['status'] == 'rejected'
