import csv
import gzip
import tempfile
import unittest
import zipfile
from pathlib import Path
from tools.dasinya_vocabulary import extract, tokens
from tools.backtranslate_badini_vocab import choose, write


class CorpusTests(unittest.TestCase):
    def test_tokenizer_preserves_joiners_and_skips_digits_and_latin(self):
        self.assertEqual(list(tokens('سلاڤ، ئه‌ڤ ١٢٣ test سلاڤ')), ['سلاڤ', 'ئه‌ڤ', 'سلاڤ'])

    def test_extract_and_backtranslate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with zipfile.ZipFile(root/'corpus.zip', 'w') as z:
                z.writestr('corpus\\3_segmented\\part_stage6_structured.txt', '<P>\n<S> سلاڤ سلاڤ ماموستا <S>\n<T> ماموستا <T>')
                z.writestr('corpus\\2_preprocessed\\ignored.txt', 'دروست')
            result = extract(root/'corpus.zip', root/'output')
            self.assertEqual((result['segmented_files'], result['word_tokens'], result['distinct_observed_forms']), (1, 4, 2))
            rows = choose(root/'output/badini-vocabulary.tsv.gz', 2)
            self.assertEqual({r['badini_word'] for r in rows}, {'سلاڤ', 'ماموستا'})
            def batches(items):
                yield [(items[0], 'hello'), (items[1], 'ئاڤ')]
            self.assertEqual(write(rows, batches, root/'back'), (2, 1))
            with (root/'back/synthetic.csv').open(encoding='utf-8') as f:
                self.assertEqual(len(list(csv.DictReader(f))), 1)

class MergeBacktranslationsTests(unittest.TestCase):
    def test_full_coverage_and_mismatch(self):
        from tools.merge_badini_backtranslations import merge
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vocab = root/'vocab.gz'
            with gzip.open(vocab, 'wt', encoding='utf-8', newline='') as f:
                w=csv.writer(f, delimiter='\t'); w.writerow(['id','badini_word','frequency']); w.writerows([['1','سلاڤ','2'], ['2','ماموستا','1']])
            batch = root/'batches/one'; batch.mkdir(parents=True)
            fpath = batch/'backtranslation-attempts.csv'
            with fpath.open('w', encoding='utf-8', newline='') as f:
                w=csv.writer(f); w.writerow(['id','badini','frequency','english','passes_filter']); w.writerow(['1','سلاڤ','2','hello','1'])
            with self.assertRaises(ValueError): merge(vocab, root/'batches', root/'missing')
            with fpath.open('a', encoding='utf-8', newline='') as f:
                csv.writer(f).writerow(['2','ماموستا','1','teacher','1'])
            result=merge(vocab, root/'batches', root/'complete')
            self.assertEqual((result['attempted'], result['script_pass']), (2,2))

    def test_start_is_one_based(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'vocab.gz'
            with gzip.open(p, 'wt', encoding='utf-8', newline='') as f:
                w=csv.writer(f,delimiter='\t');w.writerow(['id','badini_word','frequency']);w.writerows([['1','سلاڤ','2'],['2','ئاڤ','1']])
            self.assertEqual([r['id'] for r in choose(p,1,start=2)], ['2'])
