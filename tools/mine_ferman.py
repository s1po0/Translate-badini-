"""Mine approximate Sorani–Badini sentence pairs from the Ferman v2 NLU CSV.

Ferman has intent-labeled utterances, not translation-pair IDs. These are
heuristic matches, not verified parallel text. They require expert review before relying on their translations.
"""
import argparse
import collections
import csv
import re
import unicodedata
from pathlib import Path


def trigrams(text):
    text = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()
    return {text[i:i + 3] for i in range(len(text) - 2)}


def mine(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    # Keep original test/validation partitions untouched.
    source = [r for r in rows if r['language'] == 'sorani' and r['partition'] == 'train']
    target = [r for r in rows if r['language'] == 'badini' and r['partition'] == 'train']
    target_grams = [trigrams(r['utt']) for r in target]
    index = collections.defaultdict(set)
    for i, row in enumerate(target):
        for gram in target_grams[i]:
            index[(row['intent'], gram)].add(i)
    proposals = []
    for row in source:
        grams = trigrams(row['utt'])
        candidates = set()
        for gram in grams:
            candidates.update(index[(row['intent'], gram)])
        scores = []
        for i in candidates:
            other = target_grams[i]
            score = 2 * len(grams & other) / (len(grams) + len(other))
            if score >= 0.70:
                scores.append((score, i))
        scores.sort(reverse=True)
        if not scores or scores[0][0] >= .99:
            continue
        if len(scores) > 1 and scores[0][0] - scores[1][0] < .10:
            continue
        score, i = scores[0]
        proposals.append((score, row['utt'].strip(), target[i]['utt'].strip(), i))
    # Don't assign the same target to several unrelated sources.
    used_sources, used_targets, result = set(), set(), []
    for score, src, tgt, idx in sorted(proposals, reverse=True):
        if src.casefold() in used_sources or idx in used_targets:
            continue
        used_sources.add(src.casefold())
        used_targets.add(idx)
        result.append(('sorani', src, tgt))
    return sorted(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Ferman XLMR_dataset_4lang_v2.csv')
    parser.add_argument('output', help='Output CSV of approximate pairs')
    args = parser.parse_args()
    pairs = mine(args.input)
    with Path(args.output).open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source_lang', 'source_text', 'badini'])
        writer.writerows(pairs)
    print(f'Mined {len(pairs)} unverified pairs; review before relying on them.')


if __name__ == '__main__':
    main()
