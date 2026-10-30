import csv
import json
import tempfile
import unittest
from pathlib import Path
from tools.merge_all_training_data import merge
from tools.prepare_word_training import prepare
from train import split_pairs

class AllDataTest(unittest.TestCase):
    def setUp(self): self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def fixture(self):
        data=self.root/'teacher.csv'
        with data.open('w',encoding='utf-8',newline='') as f:
            w=csv.writer(f);w.writerow(['id','source_lang','source_text','badini']);w.writerows([['001','en','water','ئاڤ'],['002','ar','بيت','خانی'],['003','sorani','سڵاو','سلاڤ']])
        return data
    def test_merge_dedup_and_source_disjoint(self):
        data=self.fixture(); seed=[('en','water','ئاڤ'),('en','home','خانی')]
        result=merge(seed,{'teacher':[data]},self.root/'merged')
        self.assertEqual(result['raw_rows'],5);self.assertEqual(result['unique_pairs'],4);self.assertEqual(result['exact_duplicate_rows'],1)
        report=prepare(self.root/'merged/synthetic.csv',self.root/'shards',shard_size=10)
        rows=[]
        for shard in report['shards']:
            with (self.root/'shards'/shard['file']).open(encoding='utf8') as f:rows+=list(csv.DictReader(f))
        _, heldout=split_pairs([(r['source_lang'],r['source_text'],r['badini']) for r in rows])
        self.assertEqual(heldout,[])
    def test_bad_rows_fail_not_silently_discarded(self):
        with self.assertRaises(ValueError):merge([('en','test','english')],{},self.root/'merged')
    def test_missing_source_fails(self):
        with self.assertRaises(ValueError):merge([('en','water','ئاڤ')],{'missing':[]},self.root/'merged')

if __name__=='__main__':unittest.main()
