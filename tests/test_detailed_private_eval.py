import csv
import tempfile
import unittest
from pathlib import Path
from tools.detailed_private_eval import synthetic_sources, collect

class DetailedEvaluationTest(unittest.TestCase):
    def test_synthetic_identifiers_are_resolved_across_duplicate_origins(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/'provenance.csv').write_text('id,dataset,original_id\na,reference,\na,teacher,1\nb,reference,\n')
            (p/'synthetic.csv').write_text('id,source_lang,source_text,badini\na,en,Car,ماشین\nb,en,Tree,دار\n')
            sources,targets=synthetic_sources(p)
            self.assertEqual(sources['en'],{'car'})
            self.assertEqual(targets,{'ماشین'})
    def test_missing_provenance_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/'provenance.csv').write_text('id,dataset,original_id\na,teacher,1\n')
            (p/'synthetic.csv').write_text('id,source_lang,source_text,badini\n')
            with self.assertRaises(ValueError):synthetic_sources(p)
