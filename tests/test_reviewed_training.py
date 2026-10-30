import json
import tempfile
import unittest
from pathlib import Path

from train_adapter import read_reviewed_examples, remove_reviewed_sources
from tools.train_reviewed import decide, training_steps

class ReviewedTrainingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'private.json'
        self.rows = [{'language': 'en', 'source': 'Water', 'target': 'ئاڤ'},
                     {'language': 'sorani', 'source': 'سڵاو', 'target': 'سلاڤ'}]
        self.write(self.rows)
    def tearDown(self): self.temp.cleanup()
    def write(self, rows): self.path.write_text(json.dumps({'format': 'badini-trainer-v1', 'pairs': rows}), encoding='utf-8')
    def test_all_reviewed_rows_and_no_conflicting_replay(self):
        rows = read_reviewed_examples(self.path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(remove_reviewed_sources([('en', 'water', 'old'), ('ar', 'ماء', 'ئاڤ')], rows), [('ar', 'ماء', 'ئاڤ')])
        self.assertGreaterEqual(training_steps(self.path, 2, 3), 6)
    def test_duplicate_and_invalid_rejected(self):
        self.write(self.rows + [dict(self.rows[0], source='water')])
        with self.assertRaises(ValueError): read_reviewed_examples(self.path)
        self.write([{'language':'badini','source':'a','target':'b'}])
        with self.assertRaises(ValueError): read_reviewed_examples(self.path)
    def test_rejected_candidate_does_not_pass_guard(self):
        report = {'teacher_forced_loss_before': {'overall': 1, 'by_language': {'en': 1}},
                  'teacher_forced_loss_after': {'overall': 1.1, 'by_language': {'en': 1}}}
        self.assertEqual(decide(report, {'before_exact':2, 'after_exact':3}), (False, True))
        report['teacher_forced_loss_after']['overall'] = .9
        self.assertEqual(decide(report, {'before_exact':2, 'after_exact':1}), (True, False))

if __name__ == '__main__': unittest.main()
