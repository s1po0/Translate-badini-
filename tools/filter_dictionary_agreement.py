"""Retain teacher word outputs matching an existing attributed Badini reference.

Matching script spelling does not prove quality or correctness, nor create new
independent translations; the seed dictionary may itself contain errors.
"""
import argparse
import collections
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import read_pairs, split_pairs


def normalize(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    text = text.translate(str.maketrans({'ك': 'ک', 'ي': 'ی', 'ى': 'ی', 'ۀ': 'ە', 'ة': 'ە', 'ـ': ''}))
    text = ''.join(c for c in text if not unicodedata.combining(c) and c not in '\u200c\u200d\ufeff')
    return re.sub(r'\s+', ' ', text).strip()


def filter_pairs(attempts, output, references):
    by_source = collections.defaultdict(set)
    for lang, source, target in references:
        if lang == 'en':
            by_source[normalize(source)].add(normalize(target))
    counts = collections.Counter()
    with Path(attempts).open(encoding='utf-8', newline='') as source, Path(output).open('w', encoding='utf-8', newline='') as dest:
        reader = csv.DictReader(source)
        writer = csv.writer(dest)
        writer.writerow(['id', 'source_lang', 'source_text', 'badini'])
        for row in reader:
            counts['attempts'] += 1
            word = row['english'].strip()
            out = row['badini'].strip()
            refs = by_source.get(normalize(word), set())
            if refs:
                counts['source_has_reference'] += 1
                if normalize(out) and normalize(out) in refs:
                    counts['reference_agree'] += 1
                    writer.writerow([row['id'], 'en', word, out])
                else:
                    counts['reference_disagree'] += 1
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempts')
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    # Only source-disjoint training references; never match against held-out seed sources.
    references, _ = split_pairs(read_pairs())
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    report = filter_pairs(args.attempts, args.output, references)
    report['limitation'] = 'Normalized spelling agreement with existing references is not independent verification or new ground truth.'
    Path(args.report).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('::notice title=Dictionary agreement::' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
