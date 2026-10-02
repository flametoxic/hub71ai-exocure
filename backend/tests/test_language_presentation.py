import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from arrival.config import Settings, load_environment
from arrival.language.contracts import LanguageRequest
from arrival.language.narration import NarrationService, NarrationGenerationError
from arrival.language.openai_gateway import OpenAILanguageGateway
from arrival.modeling.results import BackendResult


class NarrationTests(unittest.TestCase):
    def test_request_language_and_retry_feedback_reach_model(self):
        calls = []
        texts = iter([
            'Синтетические данные: бюджет 999 AED.',
            'На синтетических данных бюджет составляет 12000 AED. С этим ориентиром можно продолжить подбор.',
        ])
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text=next(texts), id='response')
        gateway = OpenAILanguageGateway(Settings(openai_model='configured-model'),
            client=SimpleNamespace(responses=SimpleNamespace(create=create)))
        request = LanguageRequest(text='Расскажи о бюджете', lang='ru', resident_id='test')
        result = BackendResult(status='computed', operation='budget', facts={'budget_aed': 12000})
        answer = NarrationService(gateway).render(request, result)
        self.assertIn('12000 AED', answer.text)
        self.assertEqual(len(calls), 2)
        self.assertIn('"lang":"ru"', calls[0]['input'][-1]['content'])
        self.assertIn('unsupported_number:999', calls[1]['input'][-1]['content'])
        self.assertEqual(calls[0]['model'], 'configured-model')

    def test_model_cannot_turn_pending_proposal_into_completed_action(self):
        client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs:
            SimpleNamespace(output_text='Synthetic data: I booked the home.', id='response')))
        gateway = OpenAILanguageGateway(Settings(), client=client)
        result = BackendResult(status='approval_required', operation='home', pending=[{'id': 'proposal'}])
        with self.assertRaises(NarrationGenerationError):
            NarrationService(gateway).render(LanguageRequest(text='Prepare home', resident_id='test'), result)

    def test_env_file_and_explicit_mode(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            location = Path(directory) / '.env'
            location.write_text('CURE_LANGUAGE_MODE=auto\nOPENAI_API_KEY="test-placeholder"\n', encoding='utf-8')
            load_environment(location)
            with patch('arrival.config.load_environment'):
                self.assertEqual(Settings.from_env().language_mode, 'openai')
                os.environ['CURE_LANGUAGE_MODE'] = 'local'
                load_environment(location)
                self.assertEqual(Settings.from_env().language_mode, 'local')


if __name__ == '__main__':
    unittest.main()
