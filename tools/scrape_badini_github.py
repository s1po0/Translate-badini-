"""Use Scrapling to collect two attributed Badini CSVs rendered on GitHub.

GitHub provides both as structured CSV data inside its HTML. The source and
revision are pinned. These data are already bundled in training_data; report
coverage rather than claiming scraping created new examples or trained a model.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from urllib.robotparser import RobotFileParser

from scrapling.fetchers import Fetcher

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import read_pairs

REVISION = 'd03e6f7fe0e92c92e3cbf240928c5b0ba7674851'
BASE = 'https://github.com/Jad-deeb/Kurdish-behdini-language-resources'
ROBOTS = 'https://github.com/robots.txt'
AGENT = 'BadiniResearchCollector/1.0'
FILES = {
    'Expressions_BehdiniArabic-BehdiniLatin-English.csv': 'Kurdish (Arabic Script)',
    'Proverbs_BehdiniArabic-BehdiniLatini-English.csv': 'Kurdish Arabic',
}


def extract(page, target_column):
    nodes = page.css('script[data-target="react-app.embeddedData"]::text').getall()
    matches = []
    for text in nodes:
        payload = json.loads(text).get('payload', {}).get('codeViewBlobRoute', {})
        if payload.get('csvError'):
            raise ValueError('GitHub CSV display is incomplete: ' + str(payload['csvError']))
        if payload.get('csv'):
            matches.append(payload['csv'])
    if len(matches) != 1:
        raise ValueError('Expected exactly one rendered CSV payload')
    table = matches[0]
    if not table or 'English' not in table[0] or target_column not in table[0]:
        raise ValueError('Unexpected CSV header')
    header = table[0]
    if len(set(header)) != len(header):
        raise ValueError('Duplicate CSV header')
    results = []
    for number, values in enumerate(table[1:], 2):
        if len(values) != len(header):
            raise ValueError(f'Unexpected number of fields in row {number}')
        entry = dict(zip(header, values))
        source, target = entry['English'].strip(), entry[target_column].strip()
        if source and target:
            results.append(('en', source, target, number))
    if not results:
        raise ValueError('No parallel examples found')
    return results


def request(url):
    page = Fetcher.get(url, headers={'User-Agent': AGENT}, impersonate=None,
                       stealthy_headers=False, timeout=30)
    if page.status != 200:
        raise RuntimeError(f'Fetch failed (HTTP {page.status}) for {url}; no bypass attempted')
    return page


def collect(output):
    robots = request(ROBOTS)
    rules = RobotFileParser()
    rules.parse(robots.body.decode('utf-8').splitlines())
    pairs, counts, hashes = [], {}, {}
    for name, target_column in FILES.items():
        url = f'{BASE}/blob/{REVISION}/{name}'
        if not rules.can_fetch(AGENT, url):
            raise RuntimeError(f'robots.txt disallows {url}')
        page = request(url)
        rows = extract(page, target_column)
        counts[name] = len(rows)
        hashes[name] = hashlib.sha256(page.body).hexdigest()
        pairs.extend((lang, source, target, url, number) for lang, source, target, number in rows)
    old = set(read_pairs('training_data/ferman_approximate_sorani_badini.csv'))
    identities = {(lang, source, target) for lang, source, target, _, _ in pairs}
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    with (root / 'scraped-pairs.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source_lang', 'source_text', 'badini', 'source_url', 'source_row'])
        writer.writerows(pairs)
    (root / 'ATTRIBUTION.txt').write_text(
        'Jad-deeb, Kurdish-behdini-language-resources, ' + BASE + '\n'
        'Revision: ' + REVISION + '\nMIT license: ' + BASE + '/blob/' + REVISION + '/LICENSE\n'
        'README describes educational/research use. Translation quality is not independently verified.\n', encoding='utf-8')
    report = {'raw_pairs': len(pairs), 'distinct_pairs': len(identities), 'already_bundled_distinct': len(identities & old),
              'not_in_bundled_data': len(identities - old), 'source_counts': counts,
              'source_html_sha256': hashes, 'revision': REVISION,
              'limitation': 'Previously bundled examples, not new model training. Counts do not prove learning.'}
    (root / 'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('::notice title=Scrapling collection::' + json.dumps(report), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    collect(p.parse_args().output)

if __name__ == '__main__':
    main()
