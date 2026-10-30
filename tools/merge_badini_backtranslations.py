"""Verify every Badini vocabulary ID was attempted, then combine usable labels."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import re


def merge(vocabulary, input_dir, output):
    with gzip.open(vocabulary, 'rt', encoding='utf-8', newline='') as f:
        expected = {r['id']: r for r in csv.DictReader(f, delimiter='\t')}
    found = {}
    for file in sorted(Path(input_dir).glob('*/backtranslation-attempts.csv')):
        with file.open(encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                word = expected.get(row['id'])
                if not word or row['id'] in found or word['badini_word'] != row['badini'] or word['frequency'] != row['frequency']:
                    raise ValueError(f"Invalid, duplicate or mismatched vocabulary ID {row['id']}")
                found[row['id']] = row
    if set(expected) != set(found):
        raise ValueError(f'Incomplete coverage: {len(found)}/{len(expected)}')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    valid_count = 0
    with (root/'attempts.csv').open('x', encoding='utf-8', newline='') as a, (root/'synthetic.csv').open('x', encoding='utf-8', newline='') as s:
        wa, ws = csv.writer(a), csv.writer(s)
        wa.writerow(['id', 'badini', 'frequency', 'english', 'passes_filter'])
        ws.writerow(['id', 'source_lang', 'source_text', 'badini'])
        for identifier in expected:
            row = found[identifier]
            english = row['english'].strip()
            valid = bool(english and len(english) <= 160 and re.search('[A-Za-z]', english) and not re.search(r'[\u0600-\u06ff]', english))
            wa.writerow([identifier, row['badini'], row['frequency'], english, int(valid)])
            if valid:
                ws.writerow([identifier, 'en', english, row['badini']])
                valid_count += 1
    if not valid_count:
        raise ValueError('No usable candidates')
    report = {'attempted': len(found), 'script_pass': valid_count, 'script_fail': len(found)-valid_count,
              'limitation': 'Backtranslations and corpus tokens are unverified; script checks do not prove accuracy.'}
    (root/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('::notice title=Badini vocabulary backtranslation coverage::' + json.dumps(report), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vocabulary', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    merge(a.vocabulary, a.input, a.output)


if __name__ == '__main__':
    main()
