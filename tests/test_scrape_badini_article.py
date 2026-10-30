import unittest
from scrapling.parser import Selector
from tools.scrape_badini_article import extract


class TestArticleExtraction(unittest.TestCase):
    def test_only_bilingual_word_tables_and_preserves_glosses(self):
        page = Selector('''<html><table><tr><th>Not words</th><th>English Translation</th><th>Tags</th></tr>
        <tr><td>ئازاد</td><td>free</td><td>JJ</td></tr></table>
        <table><tr><th>Words</th><th>English Translation</th><th>Tags</th></tr>
        <tr><td>ل</td><td>On/At</td><td>IN</td></tr><tr><td>١٠٠</td><td>100</td><td>NUM</td></tr>
        <tr><td>و</td><td>And</td><td>CC</td></tr><tr><td>.</td><td></td><td>SENT</td></tr>
        </table><table><tr><th>Words</th><th>English Translation</th><th>POS Tagging</th></tr>
        <tr><td>و</td><td>And</td><td>CC</td></tr><tr><td>خۆ</td><td>Self (himself/herself)</td><td>PRPR</td></tr></table></html>''')
        rows = extract(page)
        self.assertEqual([(r['badini'],r['english_gloss']) for r in rows],
                         [('ل','On/At'),('و','And'),('خۆ','Self (himself/herself)')])
        self.assertTrue(all(r['review_status'] == 'unreviewed_contextual_gloss' for r in rows))

    def test_fail_closed_on_unexpected_page(self):
        with self.assertRaises(ValueError):
            extract(Selector('<html><body>Denied</body></html>'))


if __name__ == '__main__':
    unittest.main()
