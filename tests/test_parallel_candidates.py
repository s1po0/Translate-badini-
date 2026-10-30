import json
import tempfile
import unittest
from pathlib import Path
from tools.select_parallel_candidate import select


class ParallelSelectionTest(unittest.TestCase):
    def test_picks_passing_candidate_and_keeps_baseline_if_none_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'baseline/accepted-adapter').mkdir(parents=True)
            (root/'baseline/accepted-adapter/adapter_config.json').write_text('{}')
            def candidate(name, loss, exact):
                p=root/'candidates'/name; (p/'adapter').mkdir(parents=True)
                (p/'adapter/adapter_config.json').write_text('{}')
                report={'teacher_forced_loss_before':{'overall':3.,'by_language':{'en':3.}}, 'teacher_forced_loss_after':{'overall':loss,'by_language':{'en':loss}}}
                (p/'report.json').write_text(json.dumps(report))
                (p/'word-evaluation.json').write_text(json.dumps({'before_exact':6,'after_exact':exact,'examples':100}))
            candidate('a',2.8,5)
            candidate('b',2.9,6)
            chosen=select(root/'candidates',root/'baseline',root/'out')
            self.assertEqual(chosen['selected'],'b')
            self.assertTrue((root/'out/accepted-adapter/adapter_config.json').exists())
            (root/'candidates/b/word-evaluation.json').write_text(json.dumps({'before_exact':6,'after_exact':5,'examples':100}))
            self.assertEqual(select(root/'candidates',root/'baseline',root/'fallback')['selected'],'baseline')

    def test_rejects_inconsistent_baselines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'baseline/accepted-adapter').mkdir(parents=True)
            (root/'baseline/accepted-adapter/adapter_config.json').write_text('{}')
            for name, before in [('a',3.0),('b',4.0)]:
                p=root/'candidates'/name; (p/'adapter').mkdir(parents=True)
                (p/'adapter/adapter_config.json').write_text('{}')
                (p/'report.json').write_text(json.dumps({'teacher_forced_loss_before':{'overall':before,'by_language':{'en':before}},'teacher_forced_loss_after':{'overall':2.,'by_language':{'en':2.}}}))
                (p/'word-evaluation.json').write_text(json.dumps({'before_exact':6,'after_exact':6,'examples':100}))
            with self.assertRaises(ValueError): select(root/'candidates',root/'baseline',root/'out')
