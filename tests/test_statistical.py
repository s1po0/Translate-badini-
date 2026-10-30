import unittest
from statistical import StatisticalTranslator, train_ibm1


class StatisticalTest(unittest.TestCase):
    def test_training_learns(self):
        model = train_ibm1([('hello', 'سلاڤ'), ('hello', 'سلاڤ'), ('water', 'ئاڤ')], iterations=2)
        self.assertEqual(model['phrases']['hello'], 'سلاڤ')
        self.assertEqual(model['lexicon']['water'], 'ئاڤ')

    def test_bundled_model_and_unsupported(self):
        translator = StatisticalTranslator()
        self.assertTrue(translator.translate('water', 'en', 'badini'))
        self.assertIn('وەرنەگێڕدراو', translator.translate('zzzzunlikelywordzzzz', 'en', 'badini'))
        self.assertTrue(translator.translate('کەشوهەوای بەیجین چۆنە', 'sorani', 'badini'))
        with self.assertRaises(ValueError):
            translator.translate('سلاڤ', 'badini', 'en')

    def test_unknown_model_not_remote(self):
        translator = StatisticalTranslator()
        translator.translate('hello', 'en', 'badini')
        self.assertIn('en', translator.data['languages'])
