import json
import unittest
try:
    from scrapling.parser import Selector
    from tools.scrape_badini_github import extract
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith("scrapling"):
        raise
    Selector = None
    extract = None


def page(data):
    # Script content is JSON text, not HTML attributes.
    return Selector('<html><script data-target="react-app.embeddedData">' + json.dumps(data, ensure_ascii=False) + '</script></html>')


@unittest.skipIf(Selector is None, "optional Scrapling dependency is not installed")
class ScrapingTests(unittest.TestCase):
    def test_extract_and_preserve_original_row_numbers(self):
        data = {'payload': {'codeViewBlobRoute': {'csv': [['English', 'Kurdish Arabic'], ['Hello', 'سلاڤ'], ['', ''] ]}}}
        self.assertEqual(extract(page(data), 'Kurdish Arabic'), [('en', 'Hello', 'سلاڤ', 2)])

    def test_fail_closed_if_github_truncates_data(self):
        data = {'payload': {'codeViewBlobRoute': {'csvError': 'too large'}}}
        with self.assertRaises(ValueError):
            extract(page(data), 'Kurdish Arabic')

    def test_missing_payload(self):
        with self.assertRaises(ValueError):
            extract(Selector('<html>Denied</html>'), 'Kurdish Arabic')
