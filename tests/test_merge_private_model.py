import unittest
from types import SimpleNamespace
from tools.merge_private_model import check_destination, SUBFOLDER

class MergeDestinationTest(unittest.TestCase):
    def test_private_stable_empty_destination(self):
        check_destination(SimpleNamespace(private=True, sha='a'*40), ['adapter_config.json'])
    def test_public_refused(self):
        with self.assertRaises(ValueError):
            check_destination(SimpleNamespace(private=False, sha='a'*40), [])
    def test_existing_model_not_overwritten(self):
        with self.assertRaises(ValueError):
            check_destination(SimpleNamespace(private=True, sha='a'*40), [SUBFOLDER+'/config.json'])
    def test_missing_revision_refused(self):
        with self.assertRaises(ValueError):
            check_destination(SimpleNamespace(private=True, sha=''), [])
