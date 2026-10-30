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
