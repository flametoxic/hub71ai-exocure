"""Explicit live acceptance check. Uses real OpenAI; isolated encrypted storage.

Run: backend/.venv/Scripts/python.exe backend/tools/verify_live.py
Writes a secret-free JSON report under artifacts/. Does not touch demo residents.
"""
from pathlib import Path
import json
import os
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arrival.config import Settings, load_environment


def main():
    load_environment()
    os.environ['CURE_LANGUAGE_MODE'] = 'openai'
    for name, value in {'ARRIVAL_TEMPLATE_KEY': 'demo-template-key-change-me',
                        'ARRIVAL_VAULT_KEY': 'demo-vault-key-change-me-0001',
                        'ARRIVAL_CORE_KEY': 'demo-core-key-change-me-00001'}.items():
        os.environ.setdefault(name, value)
    from fastapi.testclient import TestClient
    from openai import OpenAI
    from arrival import api
    from arrival.service import Runtime
    from arrival.mind import Mind
    from demo_app import create_app
    settings = Settings.from_env()
    report = {'model': settings.openai_model, 'live': True, 'checks': {}}
    # Observe actual SDK requests, forwarding every call to the real provider.
    import openai
    real_openai = openai.OpenAI
    provider_calls = []
    class AuditedOpenAI(real_openai):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for resource_name in ('responses', 'embeddings'):
                resource = getattr(self, resource_name)
                original = resource.create
                def observe(*args, _original=original, _kind=resource_name, **kwargs):
                    payload = json.dumps(kwargs.get('input'), ensure_ascii=False, default=str)
                    assert settings.openai_api_key not in payload
                    assert 'asthma' not in payload.casefold() and 'астма' not in payload.casefold()
                    assert '"sealed"' not in payload.casefold() and '"diagnosis"' not in payload.casefold()
                    try:
                        response = _original(*args, **kwargs)
                    except Exception as error:
                        provider_calls.append({'kind':_kind, 'model':kwargs.get('model'), 'error':type(error).__name__,
                            'code':getattr(error,'code',None), 'message':str(error).replace(settings.openai_api_key,'[redacted]')[:700]})
                        raise
                    provider_calls.append({'kind':_kind, 'model':kwargs.get('model'),
                                           'response_id':getattr(response,'id',None), 'privacy_checked':True})
                    return response
                resource.create = observe
    openai.OpenAI = AuditedOpenAI
    destination = ROOT.parent / 'artifacts' / 'live-verification.json'
    destination.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cure-live-') as directory:
        os.environ['ARRIVAL_CORE_DIR'] = directory
        api._RT = Runtime()
        http = TestClient(create_app())
        def request(route, body=None):
            result = http.get(route) if body is None else http.post(route, json=body)
            if result.status_code != 200:
                raise RuntimeError(f'{route}: {result.status_code} {result.text[:400]}')
            return result.json()
        request('/twin/state?rid=leila')
        s = api._RT.session('leila')
        os.environ['CURE_LANGUAGE_MODE'] = 'local'
        s.set_arrival(__import__('datetime').date.fromisoformat(s.P.get('scenario.move_in')))
        s.choose('A')
        os.environ['CURE_LANGUAGE_MODE'] = 'openai'
        try:
            print('CORE: week sync / deterministic issues / graph', flush=True)
            mind = Mind(s)
            snapshot = mind.snapshot()
            assert snapshot['week']['event_count'] == 46
            assert len(s.c.extra['week_mind']['events']) == 46
            issues = snapshot['week']['issues']
            assert any(issue['kind'] == 'person_overlap' for issue in issues)
            assert any(issue['kind'] == 'car_conflict' and
                       set(issue['evidence_event_ids']) == {'thu-client', 'thu-inv'} for issue in issues)
            assert any(issue['kind'] == 'docs_not_ready' for issue in issues)
            assert any(issue['kind'] == 'unsafe_outdoor' and issue.get('calculation') == 'life.outdoor_windows'
                       for issue in issues)
            local_recall = mind.recall('dust', embeddings=False, limit=20)
            graph_event = next(item for item in local_recall['memories'] if item['id'] == 'event:mon-sports')
            assert graph_event['semantic_score'] == 0 and graph_event['graph_distance'] == 2
            report['checks']['week'] = {'event_count': 46, 'issue_counts': {
                kind: sum(issue['kind'] == kind for issue in issues)
                for kind in sorted({issue['kind'] for issue in issues})},
                'graph_nodes': len(snapshot['graph']['nodes']), 'graph_edges': len(snapshot['graph']['edges'])}
            client = OpenAI(api_key=settings.openai_api_key, timeout=60, max_retries=0)
            available = {m.id for m in client.models.list().data}
            assert settings.openai_model in available
            assert (settings.openai_agent_model or settings.openai_model) in available
            report['checks']['model_available'] = True
            print('LIVE: ordinary answer / registered tool', flush=True)
            answer = request('/twin/ask', {'rid':'leila', 'text':'Show my relocation plan.'})
            report['checks']['answer'] = answer
            assert answer['meta']['language_source'] == 'openai'
            assert answer['meta']['operation'] in {'simulate','read_state'}
            assert any(card['type'] == 'plan' for card in answer['cards'])
            print('LIVE: embeddings recall / inference', flush=True)
            report['checks']['recall'] = request('/twin/mind/recall', {'query':'Что там было с Адамом и пылью?', 'embeddings':True})
            assert report['checks']['recall']['mode'] == 'openai_embeddings_graph'
            report['checks']['inference'] = request('/twin/mind/infer', {'kind':'routine'})
            assert report['checks']['inference']['provenance'] == 'inference'
            print('LIVE: unknown-task agent', flush=True)
            task = 'I need to move my cat from Cairo to Abu Dhabi. Figure out what I need.'
            agent_reply = request('/twin/ask', {'rid':'leila', 'text':task})
            assert agent_reply['route'] == 'dynamic_agent', agent_reply
            aid = agent_reply['meta']['agent_id']
            deadline = time.monotonic() + 300
            previous = None
            while time.monotonic() < deadline:
                run = request('/twin/agents/dynamic/' + aid)
                if previous != run['status']:
                    print('AGENT:', run['status'], flush=True)
                    previous = run['status']
                if run['status'] in {'COMPLETED', 'WAITING_USER', 'FAILED'}:
                    break
                time.sleep(1)
            report['checks']['agent'] = run
            assert run['status'] in {'COMPLETED','WAITING_USER'}, run.get('error')
            assert set(run['spec']['allowed_tools']) <= {'search_public','draft_text'}
            assert run['search_trace']['approved_urls'] and any(call['status'] == 'completed' for call in run['search_calls'])
            assert run['review']['status'] == 'pending' and run['writeback_status'] == 'pending_findings'
            assert all(next(fact for fact in s.c.facts if fact.get('finding_id') == item['finding_id'])['status'] == 'pending'
                       for item in run['review']['items'])
            assert not s.c.effects
            ready_before = s._plan()['value']['ready_days']['mid']
            first_finding = run['review']['items'][0]['finding_id']
            reviewed = request('/twin/agents/dynamic/' + aid + '/review',
                               {'rid':'leila', 'decisions':[{'finding_id':first_finding, 'decision':'approve'}]})
            assert reviewed['review']['items'][0]['decision'] == 'approve'
            assert s._plan()['value']['ready_days']['mid'] == ready_before
            report['checks']['review'] = {'approved_finding': first_finding,
                                          'relocation_ready_days_unchanged': True,
                                          'action_proposals': run.get('action_proposals', [])}
            print('LIVE: provider error with invalid model', flush=True)
            old_model = os.environ.get('OPENAI_MODEL')
            os.environ['OPENAI_MODEL'] = 'cure-invalid-model-for-acceptance'
            try:
                error = http.post('/twin/ask', json={'text':'Show my relocation plan.'})
                report['checks']['error'] = {'status':error.status_code, 'body':error.json()}
                assert error.status_code == 503
            finally:
                if old_model is None:
                    os.environ.pop('OPENAI_MODEL',None)
                else:
                    os.environ['OPENAI_MODEL'] = old_model
            api._RT = Runtime()
            restarted = api._RT.session('leila')
            assert restarted.c.extra['mind']['inferences']
            assert restarted.c.extra['dynamic_agents'][aid]['result']
            assert len(restarted.c.extra['week_mind']['events']) == 46
            assert any(task.get('finding_id') == first_finding
                       for task in restarted.c.extra.get('agent_tasks', {}).values())
            report['checks']['restart'] = True
            report['checks']['outbound_privacy'] = provider_calls
            report['passed'] = True
        except Exception as error:
            report['passed'] = False
            report['failure'] = {'type':type(error).__name__, 'message':str(error)[:700]}
            print('FAIL:', type(error).__name__, str(error)[:700], flush=True)
        finally:
            report['provider_calls'] = provider_calls
            destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
            openai.OpenAI = real_openai
    print('Report:', destination, flush=True)
    return 0 if report.get('passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
