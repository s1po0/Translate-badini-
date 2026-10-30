import csv
import tempfile
import unittest
from pathlib import Path

from translate_english_words import read_words, write_results


class EnglishWordsTests(unittest.TestCase):
    def test_ranges_and_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'index.tsv'
            path.write_text('000001\thello\n000002\tworld\n000003\tfriend\n')
            self.assertEqual(read_words(path, 2, 2), [('000002', 'world'), ('000003', 'friend')])
            with self.assertRaises(ValueError):
                read_words(path, 0, 1)
            with self.assertRaises(ValueError):
                read_words(path, 4, 1)

    def test_outputs_keep_ids_and_filter(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = [('000001', 'hello'), ('000002', 'world')]
            def translate(rows):
                yield [(rows[0], 'سلاڤ'), (rows[1], 'world')]
            write_results(rows, translate, folder)
            with (Path(folder) / 'attempts.csv').open() as f:
                attempts = list(csv.DictReader(f))
            with (Path(folder) / 'synthetic.csv').open() as f:
                candidates = list(csv.DictReader(f))
            self.assertEqual([x['id'] for x in attempts], ['000001', '000002'])
            self.assertEqual([x['id'] for x in candidates], ['000001'])


if __name__ == '__main__':
    unittest.main()

class MergeWordsTests(unittest.TestCase):
    def test_merge_detects_incomplete_or_mismatched_ids(self):
        from tools.merge_translated_words import merge
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = root / 'index.tsv'
            index.write_text('000001\thello\n000002\tworld\n')
            batch = root / 'batches/part'
            batch.mkdir(parents=True)
            (batch / 'attempts.csv').write_text('id,english,badini,script_valid\n000001,hello,سلاڤ,1\n')
            with self.assertRaises(ValueError):
                merge(root / 'batches', root / 'out', index)
            (batch / 'attempts.csv').write_text('id,english,badini,script_valid\n000001,hello,سلاڤ,1\n000002,world,جیهان,1\n')
            merge(root / 'batches', root / 'out', index)
            with (root / 'out/synthetic.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 2)
