import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import distill_badini as distill
import teacher


class DistillTests(unittest.TestCase):
    def test_english_direct(self):
        run = Mock(return_value='سلاڤ')
        self.assertEqual(distill.translate_example(run, 'Hello', 'en', 'badini-model', 'pivot'), 'سلاڤ')
        run.assert_called_once_with('badini-model', 'Hello', 'en', 'badini')

    def test_pivot_languages(self):
        for language in ('ar', 'sorani'):
            run = Mock(side_effect=['Hello', 'سلاڤ'])
            self.assertEqual(distill.translate_example(run, 'source', language, 'badini-model', 'pivot'), 'سلاڤ')
            self.assertEqual(run.call_args_list[0].args, ('pivot', 'source', language, 'en'))
            self.assertEqual(run.call_args_list[1].args, ('badini-model', 'Hello', 'en', 'badini'))

    def test_output_is_unapproved_and_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, review, output = (Path(tmp) / p for p in ('input.csv', 'review.csv', 'approved.csv'))
            with source.open('w', newline='', encoding='utf-8') as f:
                w = csv.writer(f); w.writerow(['source_lang', 'source_text']); w.writerow(['en', 'Hello'])
            distill.generate(source, review, runner=Mock(return_value='سلاڤ'))
            with review.open(encoding='utf-8', newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]['approved'], '')
            self.assertEqual(rows[0]['teacher_model'], distill.BADINI_MODEL)
            with self.assertRaises(ValueError):
                teacher.export_approved(review, output)
            with self.assertRaises(ValueError):
                distill.generate(source, review, runner=Mock())

    def test_invalid_input_and_empty_pivot(self):
        with self.assertRaises(ValueError):
            distill.translate_example(Mock(), 'test', 'fa', 'teacher', 'pivot')
        with self.assertRaises(ValueError):
            distill.translate_example(Mock(return_value=''), 'test', 'ar', 'teacher', 'pivot')
