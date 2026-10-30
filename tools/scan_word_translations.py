"""Audit unverified numbered teacher output; never treat script checks as correctness."""
import argparse
import collections
import csv
import json
import re
from pathlib import Path

COMMON = ('hello', 'water', 'house', 'mother', 'father', 'love', 'book', 'school', 'dog', 'cat', 'good', 'bad', 'eat', 'go', 'beautiful', 'friend', 'one', 'two', 'sun', 'moon')


def scan(path):
    total = 0
    valid = 0
    blank = 0
    unchanged = 0
    suspicious_latin = 0
    targets = collections.Counter()
    samples = {}
    with Path(path).open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if not {'id', 'english', 'badini', 'script_valid'}.issubset(reader.fieldnames or []):
            raise ValueError('Unexpected attempts CSV columns')
        for row in reader:
            total += 1
            source = row['english'].strip()
            output = row['badini'].strip()
            flag = row['script_valid'] == '1'
            valid += flag
            blank += not bool(output)
            unchanged += source.casefold() == output.casefold()
            suspicious_latin += bool(re.search(r'[A-Za-z]', output))
            if output:
                targets[output] += 1
            if source in COMMON:
                samples[source] = {'id': row['id'], 'output': output, 'script_valid': flag}
    if not total:
        raise ValueError('No attempts')
    return {'attempts': total, 'script_pass': valid, 'script_fail': total-valid,
            'blank_output': blank, 'unchanged': unchanged, 'contains_latin': suspicious_latin,
            'distinct_nonempty_outputs': len(targets), 'most_common_outputs': targets.most_common(10),
            'samples': samples,
            'caveat': 'Script checks and repetition statistics do not verify Badini dialect or translation accuracy.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = scan(args.input)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('::notice title=Word translation audit::' + json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
