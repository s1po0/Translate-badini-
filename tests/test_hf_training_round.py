import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from tools.hf_training_round import decide, update_card, publish, FILES, REPO, START, END
from tools.publish_hf_adapter import CARD

class HFTrainingRoundTest(unittest.TestCase):
    def setUp(self):
        self.report={'teacher_forced_loss_before':{'overall':3.0,'by_language':{'en':3.0}},
                     'teacher_forced_loss_after':{'overall':2.9,'by_language':{'en':3.0}},
                     'optimizer_steps':2000,'synthetic_unique_rows':100}
        self.words={'examples':100,'before_exact':6,'after_exact':6}
    def test_guard_requires_both_metrics(self):
        self.assertTrue(decide(self.report,self.words)['candidate_accepted'])
        self.words['after_exact']=5
        self.assertFalse(decide(self.report,self.words)['candidate_accepted'])
        self.words['after_exact']=6; self.report['teacher_forced_loss_after']['overall']=3.1
        self.assertFalse(decide(self.report,self.words)['candidate_accepted'])
    def test_card_preserves_existing_sections_and_updates(self):
        decision=decide(self.report,self.words)
        output=update_card(CARD,decision,'12345')
        self.assertIn(START,output);self.assertIn(END,output)
        self.assertIn('41047507896',CARD);self.assertIn('12345',output)
        self.assertIn('CC BY-NC',output)
        self.assertIn('67890',update_card(output,decision,'67890'))
    def test_publish_checks_revision_and_never_publishes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); candidate=root/'candidate';baseline=root/'baseline'
            candidate.mkdir();baseline.mkdir()
            from tools.publish_hf_adapter import BASE
            for name in FILES:
                data=b'{"base_model_name_or_path": "'+BASE.encode()+b'"}' if name=='adapter_config.json' else b'test'
                (candidate/name).write_bytes(data)
                (baseline/name).write_bytes(b'old-weights' if name=='adapter_model.safetensors' else data)
            (baseline/'README.md').write_text(CARD)
            ref={'repo':REPO,'revision':'a'*40}; decision=decide(self.report,self.words)
            class API:
                token='fake-token'
                def __init__(self,sha): self.sha=sha;self.uploaded=False
                def repo_info(self,**kw):return SimpleNamespace(sha=self.sha)
                def list_repo_files(self,**kw):return ['README.md']
                def upload_folder(self,**kw):
                    assert kw['parent_commit']=='a'*40
                    self.uploaded=True
                    return 'ok'
            api=API('b'*40)
            with self.assertRaises(ValueError):publish(api,candidate,baseline,ref,decision,'123',root/'out')
            self.assertFalse(api.uploaded)
            api=API('a'*40); denied=dict(decision,candidate_accepted=False)
            with self.assertRaises(ValueError):publish(api,candidate,baseline,ref,denied,'123',root/'out')
            self.assertFalse(api.uploaded)
            fake=SimpleNamespace(hf_hub_download=lambda **kw: str(candidate/'adapter_model.safetensors'))
            with patch.dict('sys.modules', {'huggingface_hub':fake}):
                self.assertEqual(publish(api,candidate,baseline,ref,decision,'123',root/'out'),'ok')
            self.assertTrue(api.uploaded)

    def test_modified_card_not_overwritten(self):
        with self.assertRaises(ValueError):update_card('A different user card',decide(self.report,self.words),'12345')
        with self.assertRaises(ValueError):update_card(START+'broken',decide(self.report,self.words),'12345')

if __name__=='__main__':unittest.main()
