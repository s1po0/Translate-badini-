"""Combine numbered teacher outputs, refusing missing, duplicate or mismatched IDs."""
import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from translate_english_words import INDEX
from synthetic_batch import valid_badini


def merge(input_dir, output, index=INDEX):
    entries = {}
    with Path(index).open(encoding='utf-8') as f:
        for line in f:
            i, word = line.rstrip('\n').split('\t', 1)
            entries[i] = word
    attempts = {}
    for path in sorted(Path(input_dir).glob('*/attempts.csv')):
        with path.open(encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                i = row['id']
                if i in attempts or entries.get(i) != row['english']:
                    raise ValueError(f'Duplicate or unknown ID or mismatched English: {i}')
                attempts[i] = row['badini']
    if set(attempts) != set(entries):
        raise ValueError(f'Incomplete translations: {len(attempts)}/{len(entries)}')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'attempts.csv').open('x', encoding='utf-8', newline='') as a, (root / 'synthetic.csv').open('x', encoding='utf-8', newline='') as s:
        wa, ws = csv.writer(a), csv.writer(s)
        wa.writerow(['id', 'english', 'badini', 'script_valid'])
        ws.writerow(['id', 'source_lang', 'source_text', 'badini'])
        valid_count = 0
        for i, word in entries.items():
            result = attempts[i].strip()
            valid = valid_badini(result)
            wa.writerow([i, word, result, int(valid)])
            if valid:
                ws.writerow([i, 'en', word, result])
                valid_count += 1
    if not valid_count:
        raise ValueError('No results passed the script check')
    print(f'Complete numbered attempts: {len(entries)}; script-filtered unverified pairs: {valid_count}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--index', default=INDEX)
    args = parser.parse_args()
    merge(args.input, args.output, args.index)


if __name__ == '__main__':
    main()
