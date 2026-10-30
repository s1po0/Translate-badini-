"""Collect the published Badini/English examples, not an autonomous web crawl.

The source's words are glosses in sentence context, not necessarily standalone
translations. Output is deliberately review-only and never published as a model.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from urllib.robotparser import RobotFileParser

from scrapling.fetchers import Fetcher

URL = 'https://www.mdpi.com/2078-2489/15/9/578'
ROBOTS = 'https://www.mdpi.com/robots.txt'
AGENT = 'BadiniResearchCollector/1.0'
ATTRIBUTION = ('Media Azzat, Karwan Jacksi, Ismael Ali, A Hybrid Approach to Ontology '
               'Construction for the Badini Kurdish Language, Information 15(9), 578, 2024, '
               'https://doi.org/10.3390/info15090578; CC BY 4.0: '
               'https://creativecommons.org/licenses/by/4.0/. '
               'Extracted table rows; context and tags retained; not independently verified.')


def cells(row):
    return [' '.join(' '.join(c.css('::text').getall()).split()) for c in row.css('th, td')]


def extract(page):
    found = []
    seen = set()
    for table in page.css('table'):
        rows = [cells(row) for row in table.css('tr')]
        header_index = next((i for i, row in enumerate(rows)
                             if len(row) >= 3 and row[0].lower() == 'words'
                             and row[1].lower() == 'english translation'), None)
        if header_index is None:
            continue
        for row in rows[header_index + 1:]:
            if len(row) < 3:
                continue
            badini, english, tag = row[:3]
            # Keep contextual glosses intact (including alternatives), never
            # turn POS labels, numbers or punctuation into translated words.
            if not badini or not english or not any('\u0600' <= c <= '\u06ff' and c.isalpha() for c in badini):
                continue
            key = (badini, english, tag)
            if key in seen:
                continue
            seen.add(key)
            found.append({'badini': badini, 'english_gloss': english, 'pos': tag,
                          'source': URL, 'review_status': 'unreviewed_contextual_gloss'})
    if not found:
        raise ValueError('No expected bilingual tables found; source may have changed or denied access')
    return found


def fetch():
    robot = Fetcher.get(ROBOTS, headers={'User-Agent': AGENT}, impersonate=None, stealthy_headers=False, timeout=30)
    if robot.status != 200:
        raise RuntimeError(f'Cannot verify robots.txt (HTTP {robot.status})')
    rules = RobotFileParser()
    rules.set_url(ROBOTS)
    rules.parse(robot.body.decode('utf-8').splitlines())
    if not rules.can_fetch(AGENT, URL):
        raise RuntimeError('Source robots.txt does not allow collection')
    page = Fetcher.get(URL, headers={'User-Agent': AGENT}, impersonate=None, stealthy_headers=False, timeout=30)
    if page.status != 200:
        raise RuntimeError(f'Article fetch failed (HTTP {page.status}); no bypass attempted')
    if '10.3390/info15090578' not in page.body.decode('utf-8', errors='replace'):
        raise ValueError('Unexpected article content')
    return page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    page = fetch()
    rows = extract(page)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    with (root / 'review-only.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (root / 'ATTRIBUTION.txt').write_text(ATTRIBUTION + '\n', encoding='utf-8')
    report = {'count': len(rows), 'source': URL, 'source_sha256': hashlib.sha256(page.body).hexdigest(),
              'license': 'CC BY 4.0', 'status': 'review-only contextual glosses; no model trained or published'}
    (root / 'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('::notice title=Scrapling collection::' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
