import unittest
from tools.compare_private_model import sample_rows

class Samples(unittest.TestCase):
    def test_groups_reference_variants_and_rejects_phrases_for_words(self):
        rows=[('en','Car','کار'),('en','Car','ماشین'),('en','long phrase','وشە'),
              ('ar','سيارة','ترومبێل'),('ar','كلمة اثنان','وشە'),
              ('sorani','چۆنی','تو چەوانی')]
        result=sample_rows(rows)
        self.assertEqual(len(result['en']),1)
        self.assertEqual(result['en'][0][1],{'کار','ماشین'})
        self.assertEqual(len(result['ar']),1)
        self.assertEqual(len(result['sorani']),1)
    def test_requires_all_languages(self):
        with self.assertRaises(ValueError):sample_rows([('en','a','ا')])
