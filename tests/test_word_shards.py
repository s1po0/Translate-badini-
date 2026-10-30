import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.prepare_word_training import prepare
from tools.word_shard_stage import decide, plan
from train import split_pairs


class WordShardTests(unittest.TestCase):
    def test_shards_cover_training_once_and_exclude_heldout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'synthetic.csv'
            with path.open('w', encoding='utf-8', newline='') as f:
                w = csv.writer(f)
                w.writerow(['id', 'source_lang', 'source_text', 'badini'])
                for i in range(100):
                    w.writerow([str(i), 'en', f'word{i}', 'بادینی'])
            manifest = prepare(path, Path(tmp)/'shards', 12)
            shards = []
            for item in manifest['shards']:
                with (Path(tmp)/'shards'/item['file']).open(encoding='utf-8') as f:
                    shards.extend(list(csv.DictReader(f)))
            train, heldout = split_pairs([('en', f'word{i}', 'بادینی') for i in range(100)])
            self.assertEqual(len(shards), len(train))
            self.assertEqual(len({r['id'] for r in shards}), len(shards))
            self.assertFalse({r['source_text'] for r in shards} & {r[1] for r in heldout})
            self.assertEqual(manifest['heldout_rows'], len(heldout))

    def test_stage_steps_cover_full_training_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp)/'shard.csv'
            src, _ = split_pairs([('en', f'word{i}', 'بادینی') for i in range(10)])
            with shard.open('w', newline='', encoding='utf-8') as f:
                w = csv.writer(f); w.writerow(['id','source_lang','source_text','badini'])
                for i, row in enumerate(src): w.writerow([i, *row])
            with patch('tools.word_shard_stage.read_pairs', return_value=[('en', 'replay', 'بادینی')]):
                count, steps = plan(shard, 1, 1)
            self.assertEqual(count, len(src))
            self.assertGreaterEqual(steps, count)

    def test_guards_reject_regression(self):
        report = {'teacher_forced_loss_before': {'overall': 2., 'by_language': {'en': 2.}}, 'teacher_forced_loss_after': {'overall': 2.1, 'by_language': {'en': 2.1}}}
        loss, words = decide(report, {'before_exact': 5, 'after_exact': 6, 'examples': 100})
        self.assertFalse(loss)
        self.assertTrue(words)
