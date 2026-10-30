import json
import tempfile
import unittest
from pathlib import Path
from tools.finalize_all_training import combined_decision

class FinalizeAllTest(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();root=Path(self.t.name);self.first=root/'first';self.final=root/'final';self.first.mkdir();self.final.mkdir()
        before={'overall':3.0,'by_language':{'en':3.0}}; after={'overall':2.9,'by_language':{'en':2.9}}
        (self.first/'training-report.json').write_text(json.dumps({'teacher_forced_loss_before':before}))
        (self.final/'training-report.json').write_text(json.dumps({'teacher_forced_loss_before':before,'teacher_forced_loss_after':after}))
        self.progress={'stages':[{'stage':1,'candidate_accepted':True,'optimizer_steps':2},{'stage':2,'candidate_accepted':False,'optimizer_steps':2}], 'examples_seen_in_attempts':4,'planned_examples':4,'examples_in_accepted_stages':2}
        self.write();self.words={'examples':100,'before_exact':6,'after_exact':6}
    def tearDown(self):self.t.cleanup()
    def write(self):(self.final/'progress.json').write_text(json.dumps(self.progress))
    def test_rejected_last_uses_previous_checkpoint_loss(self):
        result=combined_decision(self.first,self.final,self.words,2)
        self.assertEqual(result['heldout_loss_after'],3.0);self.assertTrue(result['candidate_accepted'])
        self.assertEqual(result['accepted_training_pairs'],2)
    def test_incomplete_and_no_accepted(self):
        self.progress['examples_seen_in_attempts']=3;self.write()
        with self.assertRaises(ValueError):combined_decision(self.first,self.final,self.words,2)
        self.progress['examples_seen_in_attempts']=4;self.progress['stages'][0]['candidate_accepted']=False;self.progress['examples_in_accepted_stages']=0;self.write()
        self.assertFalse(combined_decision(self.first,self.final,self.words,2)['candidate_accepted'])

if __name__=='__main__':unittest.main()
