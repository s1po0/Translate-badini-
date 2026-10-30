import unittest
from train_adapter import select_examples
from train import split_pairs


class AdapterDataTest(unittest.TestCase):
    def test_balanced_selection_and_determinism(self):
        rows = [(lang, f'text {i}', 'badini') for lang in ('en', 'ar', 'sorani') for i in range(6)]
        selected = select_examples(rows, 2, 42)
        self.assertEqual(selected, select_examples(rows, 2, 42))
        self.assertEqual(len(selected), 6)
        self.assertEqual({lang: sum(row[0] == lang for row in selected) for lang in ('en', 'ar', 'sorani')},
                         {'en': 2, 'ar': 2, 'sorani': 2})

    def test_source_split_disjoint(self):
        rows = [('en', 'Same', 'one'), ('en', 'same', 'two'), ('ar', 'نص', 'بادینی')]
        train, test = split_pairs(rows)
        self.assertFalse({(a, b.casefold()) for a, b, _ in train} & {(a, b.casefold()) for a, b, _ in test})


class AdapterServingTest(unittest.TestCase):
    def test_supported_directions_route_to_trained_adapter(self):
        from unittest.mock import MagicMock, patch
        import app
        server = app.AdapterTranslator('/tmp/test-adapter')
        server.models[server.adapter_path] = (MagicMock(), MagicMock())
        with patch.object(server, '_run', return_value='بادینی') as run:
            self.assertEqual(server.translate('hello', 'en', 'badini'), 'بادینی')
            run.assert_called_once_with(server.adapter_path, 'hello', 'en', 'badini')
        with self.assertRaises(ValueError):
            server.translate('بادینی', 'badini', 'en')


class AdapterPageTest(unittest.TestCase):
    def test_public_demo_page_labels_and_limits(self):
        import os
        from unittest.mock import patch
        from http.client import HTTPConnection
        from http.server import ThreadingHTTPServer
        import threading
        import app
        server = ThreadingHTTPServer(('127.0.0.1', 0), app.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        with patch.object(app, 'translator', app.AdapterTranslator('/tmp/adapter')), patch.dict(os.environ, {'PUBLIC_DEMO': '1'}):
            thread.start()
            try:
                conn = HTTPConnection('127.0.0.1', server.server_port)
                conn.request('GET', '/')
                response = conn.getresponse()
                body = response.read().decode()
                self.assertEqual(response.status, 200)
                self.assertIn('AIـی بادینیی ڕاهێنراو', body)
                self.assertIn('زانیاریی تایبەت مەنووسە', body)
                self.assertIn('id="swap" hidden', body)
                conn.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

class SyntheticSelectionTest(unittest.TestCase):
    def test_limit_is_deterministic_and_preserves_default(self):
        from train_adapter import limit_synthetic
        rows = list(range(100))
        self.assertEqual(limit_synthetic(rows, 0, 1), rows)
        self.assertEqual(limit_synthetic(rows, 10, 1), limit_synthetic(rows, 10, 1))
        self.assertEqual(len(set(limit_synthetic(rows, 10, 1))), 10)
