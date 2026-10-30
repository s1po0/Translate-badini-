import csv
import tempfile
import unittest
from pathlib import Path
from tools.teacher_consensus import compare, combine

class TeacherConsensusTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def attempts(self,name,rows):
        p=self.root/name
        with p.open('w',encoding='utf8',newline='') as f:
            w=csv.writer(f);w.writerow(['id','english','badini','script_valid']);w.writerows(rows)
        return p
    def test_agreement_not_script_only(self):
        a=self.attempts('a.csv',[['000001','water','ئاڤ','1'],['000002','house','ماڵ','1'],['000003','bad','bad','0']])
        b=self.attempts('b.csv',[['000001','water','ئاڤ','1'],['000002','house','خانی','1'],['000003','bad','bad','0']])
        d=self.root/'batch';d.mkdir(); report=compare(a,b,d/'agreement.csv')
        self.assertEqual(report['attempted'],3);self.assertEqual(report['normalized_agreement'],1)
        combine(self.root,self.root/'result.csv',self.root/'report.json')
        with (self.root/'result.csv').open(encoding='utf8') as f:
            self.assertEqual(len(list(csv.DictReader(f))),1)
    def test_mismatch_fails(self):
        a=self.attempts('a.csv',[['000001','water','ئاڤ','1']]); b=self.attempts('b.csv',[['000001','wrong','ئاڤ','1']])
        with self.assertRaises(ValueError):compare(a,b,self.root/'out.csv')
    def test_duplicate_attempts_rejected(self):
        a=self.attempts('a.csv',[['000001','water','ئاڤ','1'],['000001','water','ئاڤ','1']])
        with self.assertRaises(ValueError):compare(a,a,self.root/'out.csv')

if __name__=='__main__':unittest.main()
