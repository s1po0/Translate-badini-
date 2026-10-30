import unittest
from train_adapter import select_examples
from train import split_pairs


class AdapterDataTest(unittest.TestCase):
    def test_balanced_selection_and_determinism(self):
        rows = [(lang, f'text {i}', 'badini') for lang in ('en', 'ar', 'sorani') for i in range(6)]
        selected = select_examples(rows, 2, 42)
        self.assertEqual(selected, select_examples(rows, 2, 42))
        self.assertEqual(len(selected), 6)
        self.assertEqual({lang: sum(row[0] == lang for row in selected) for lang in ('en', 'ar', 'sorani')},
                         {'en': 2, 'ar': 2, 'sorani': 2})

    def test_source_split_disjoint(self):
        rows = [('en', 'Same', 'one'), ('en', 'same', 'two'), ('ar', 'نص', 'بادینی')]
        train, test = split_pairs(rows)
        self.assertFalse({(a, b.casefold()) for a, b, _ in train} & {(a, b.casefold()) for a, b, _ in test})
