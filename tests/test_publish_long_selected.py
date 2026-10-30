import json
import tempfile
import unittest
from pathlib import Path
from tools.hf_training_round import FILES
from tools.publish_hf_adapter import BASE
from tools.publish_long_selected import prepare_decision, WINNER

class PublishLongSelectedTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.baseline=self.root/'baseline'; self.original=self.root/'original'; self.selected=self.root/'selected'; self.candidate=self.root/'candidate'
        for path, weight in [(self.baseline,b'old'),(self.original/'accepted-adapter',b'old'),(self.selected/'accepted-adapter',b'new'),(self.candidate/'adapter',b'new')]:
            path.mkdir(parents=True)
            for name in FILES:
                content = json.dumps({'base_model_name_or_path':BASE}).encode() if name == 'adapter_config.json' else (weight if name == 'adapter_model.safetensors' else b'tokenizer')
                (path/name).write_bytes(content)
        row={'name': WINNER,'loss_pass':True,'word_pass':True,'overall_loss_before':3.0,'overall_loss_after':2.9,'word_exact_before':6,'word_exact_after':8}
        (self.selected/'selection.json').write_text(json.dumps({'selected':WINNER,'candidates':[row]}))
        report={'teacher_forced_loss_before':{'overall':3.0,'by_language':{'en':3}}, 'teacher_forced_loss_after':{'overall':2.9,'by_language':{'en':2.9}},'optimizer_steps':9000,'synthetic_unique_rows':300}
        (self.candidate/'report.json').write_text(json.dumps(report))
        (self.candidate/'word-evaluation.json').write_text(json.dumps({'examples':100,'before_exact':6,'after_exact':8}))
    def run_guard(self):return prepare_decision(self.baseline,self.original,self.selected,self.candidate)
    def test_accepts_verified_winner(self):
        self.assertTrue(self.run_guard()['candidate_accepted'])
    def test_refuses_changed_hub(self):
        (self.baseline/'adapter_model.safetensors').write_bytes(b'other')
        with self.assertRaises(ValueError):self.run_guard()
    def test_refuses_mismatched_selected_weights(self):
        (self.selected/'accepted-adapter'/'adapter_model.safetensors').write_bytes(b'other')
        with self.assertRaises(ValueError):self.run_guard()
    def test_refuses_failed_word_guard(self):
        (self.candidate/'word-evaluation.json').write_text(json.dumps({'examples':100,'before_exact':6,'after_exact':5}))
        with self.assertRaises(ValueError):self.run_guard()

if __name__ == '__main__':unittest.main()
