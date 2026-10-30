import csv
import tempfile
import unittest
from pathlib import Path
from tools.scan_word_translations import scan


class WordScanTest(unittest.TestCase):
    def test_counts_and_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'attempts.csv'
            with file.open('w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['id', 'english', 'badini', 'script_valid'])
                w.writerows([['1', 'hello', 'سلاڤ', '1'], ['2', 'house', 'house', '0'], ['3', 'bird', '', '0']])
            data = scan(file)
            self.assertEqual((data['attempts'], data['script_pass'], data['blank_output'], data['unchanged']), (3, 1, 1, 1))
            self.assertEqual(data['samples']['hello']['id'], '1')

class DictionaryAgreementTest(unittest.TestCase):
    def test_agreement_is_strict_and_retains_id(self):
        from tools.filter_dictionary_agreement import filter_pairs, normalize
        self.assertEqual(normalize('كِتاب'), normalize('کتاب'))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'attempts.csv'
            with path.open('w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['id', 'english', 'badini', 'script_valid'])
                w.writerows([['01', 'book', 'كِتاب', 1], ['02', 'cat', 'سگ', 1], ['03', 'unknown', 'نازانم', 1]])
            output = Path(tmp) / 'filtered.csv'
            counts = filter_pairs(path, output, [('en', 'book', 'کتاب'), ('en', 'cat', 'پشیک')])
            self.assertEqual(counts['reference_agree'], 1)
            self.assertEqual(counts['reference_disagree'], 1)
            with output.open(encoding='utf-8') as f:
                self.assertEqual(list(csv.DictReader(f))[0]['id'], '01')

class WordEvaluationTest(unittest.TestCase):
    def test_fixed_sources_group_variants_and_filter_non_words(self):
        from tools.evaluate_word_quality import heldout_words
        rows = [('en', 'Cat', 'پشیک'), ('en', 'cat', 'پشی'), ('en', 'two words', 'دو'), ('ar', 'نص', 'بادینی')]
        words = heldout_words(rows, 100)
        self.assertEqual(words, [('cat', {'پشیک', 'پشی'})])
