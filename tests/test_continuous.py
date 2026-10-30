import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import continuous_training as continuous
import synthetic_batch


class ContinuousTrainingTests(unittest.TestCase):
    def test_teacher_batch_pivots_and_filters(self):
        class FakeRunner:
            def __init__(self):
                self.calls, self.unloads = [], 0
            def translate(self, model, text, source, target):
                self.calls.append((model, text, source, target))
                return 'English pivot' if target == 'en' else 'سلاڤ'
            def unload(self):
                self.unloads += 1
        runner = FakeRunner()
        rows = synthetic_batch.make_batch([('en', 'hello'), ('sorani', 'سڵاو')], runner, 'teacher', 'pivot')
        self.assertEqual(len(rows), 2)
        self.assertEqual(runner.calls[0], ('pivot', 'سڵاو', 'sorani', 'en'))
        self.assertEqual(runner.calls[1], ('teacher', 'hello', 'en', 'badini'))
        self.assertEqual(runner.calls[2], ('teacher', 'English pivot', 'en', 'badini'))
        self.assertEqual(runner.unloads, 2)
        self.assertFalse(synthetic_batch.valid_badini('hello'))

    def test_sources_never_use_heldout(self):
        pools = continuous.source_pool()
        from train import read_pairs, split_pairs
        from pathlib import Path
        _, heldout = split_pairs(read_pairs(Path('training_data/ferman_approximate_sorani_badini.csv')))
        selected = continuous.select_sources(pools, 1, 5)
        self.assertFalse(set(selected) & {(l, s) for l, s, _ in heldout})
        self.assertEqual(len(selected), 15)
        self.assertNotEqual(selected, continuous.select_sources(pools, 2, 5))

    def test_guard_rejects_regression_and_nonfinite(self):
        def report(before, after):
            return {'teacher_forced_loss_before': {'overall': before, 'by_language': {'en': before}},
                    'teacher_forced_loss_after': {'overall': after, 'by_language': {'en': after}}}
        self.assertTrue(continuous.accepted(report(3, 2), .03))
        self.assertFalse(continuous.accepted(report(3, 4), .03))
        self.assertFalse(continuous.accepted(report(3, float('nan')), .03))

    def test_checkpoint_commit_and_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'initial-adapter').mkdir()
            state = {'cycle': 0, 'adapter': 'initial-adapter', 'base': 'base'}
            args = SimpleNamespace(output=tmp, teachers=['teacher'], batch_per_language=1, allow_download=False,
                pivot='pivot', base='base', steps=1, replay_per_language=1, eval_per_language=1, max_regression=.03)
            pools = {'en': ['hello'], 'ar': ['مرحبا'], 'sorani': ['سڵاو']}
            def run(command, **kwargs):
                if command[1].endswith('train_adapter.py'):
                    out = Path(command[command.index('--output') + 1]); (out / 'adapter').mkdir(parents=True)
                    (out / 'adapter/adapter_config.json').write_text('{}')
                    metrics = {'teacher_forced_loss_before': {'overall': 3, 'by_language': {'en': 3}},
                               'teacher_forced_loss_after': {'overall': 2, 'by_language': {'en': 2}}}
                    (out / 'report.json').write_text(json.dumps(metrics))
            new = continuous.run_cycle(args, state, pools, run=run)
            self.assertTrue(new['last_accepted'])
            self.assertTrue((root / new['adapter'] / 'adapter_config.json').exists())
            self.assertEqual(json.loads((root / 'state.json').read_text()), new)
            def broken(command, **kwargs):
                raise RuntimeError('failure')
            with self.assertRaises(RuntimeError):
                continuous.run_cycle(args, new, pools, run=broken)
            self.assertEqual(json.loads((root / 'state.json').read_text()), new)
