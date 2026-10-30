import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app
import train


class TrainingDataTests(unittest.TestCase):
    def test_included_data_and_clean_split(self):
        pairs = train.read_pairs()
        self.assertTrue(any(row[0] == 'en' for row in pairs))
        self.assertTrue(any(row[0] == 'ar' for row in pairs))
        training, heldout = train.split_pairs(pairs)
        self.assertTrue(training and heldout)
        self.assertFalse({(l, s.casefold()) for l, s, _ in training} &
                         {(l, s.casefold()) for l, s, _ in heldout})

    def test_custom_sorani_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pairs.csv'
            with path.open('w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['source_lang', 'source_text', 'badini'])
                writer.writerow(['sorani', 'سڵاو', 'سلاڤ'])
            self.assertIn(('sorani', 'سڵاو', 'سلاڤ'), train.read_pairs(path))

    def test_personal_model_routing(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'badini_model.json').write_text(json.dumps({'source_languages': ['en', 'ar']}))
            with patch.dict('os.environ', {'BADINI_MODEL_DIR': directory}):
                translator = app.Translator()
            with patch.object(translator, '_run', return_value='سلاڤ') as run:
                self.assertEqual(translator.translate('hello', 'en', 'badini'), 'سلاڤ')
                run.assert_called_once_with(directory, 'hello', 'en', 'badini')
                with self.assertRaises(ValueError):
                    translator.translate('سڵاو', 'sorani', 'badini')
                with self.assertRaises(ValueError):
                    translator.translate('سلاڤ', 'badini', 'en')


class SplitStabilityTests(unittest.TestCase):
    def test_split_is_stable_when_new_examples_are_added(self):
        pairs = [('en', 'hello', 'سلاڤ'), ('en', 'water', 'ئاڤ'), ('ar', 'ماء', 'ئاڤ')]
        a_train, a_test = train.split_pairs(pairs)
        b_train, b_test = train.split_pairs(pairs + [('sorani', 'سڵاو', 'سلاڤ')])
        self.assertEqual(set(a_train), set(b_train) & set(pairs))
        self.assertEqual(set(a_test), set(b_test) & set(pairs))

    def test_card_examples_included(self):
        self.assertIn(('en', 'How are you? Are you doing well?', 'تو چەوانی باشی'), train.read_pairs())
