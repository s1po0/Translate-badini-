import json
import tempfile
import unittest
from pathlib import Path
from tools.publish_hf_adapter import BASE, CARD, package_adapter

class HfPublishTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source'
        adapter = self.source / 'accepted-adapter'
        adapter.mkdir(parents=True)
        (adapter / 'adapter_config.json').write_text(json.dumps({'base_model_name_or_path':BASE}))
        (adapter / 'adapter_model.safetensors').write_bytes(b'test')
        (adapter / 'tokenizer_config.json').write_text('{}')
        (adapter / 'README.md').write_text('auto card')
        (self.source / 'selection.json').write_text(json.dumps({'selected':'parallel-candidate-balanced'}))
    def tearDown(self): self.tmp.cleanup()
    def test_new_repo_receives_card_and_weights(self):
        target=self.root / 'staged'; result=package_adapter(self.source,target,['.gitattributes'])
        self.assertIn('adapter_model.safetensors',result)
        self.assertEqual((target/'README.md').read_text(),CARD)
        self.assertIn('selection.json',result)
    def test_existing_readme_preserved(self):
        target=self.root/'staged'; result=package_adapter(self.source,target,['README.md'])
        self.assertNotIn('README.md',result); self.assertIn('BADINI_MODEL_INFO.md',result)
    def test_existing_weights_refuse_overwrite(self):
        with self.assertRaises(ValueError): package_adapter(self.source,self.root/'staged',['adapter_model.safetensors'])
    def test_unexpected_base_refused(self):
        (self.source/'accepted-adapter'/'adapter_config.json').write_text(json.dumps({'base_model_name_or_path':'wrong'}))
        with self.assertRaises(ValueError): package_adapter(self.source,self.root/'staged',[])

if __name__ == '__main__': unittest.main()
