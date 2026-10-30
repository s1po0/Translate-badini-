"""Make deterministic, source-disjoint, numbered training shards.

These are *unverified* teacher outputs. Non-Arabic outputs and held-out
sources are deliberately not used as translation supervision.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import split_pairs
from synthetic_batch import valid_badini


def prepare(synthetic, output, shard_size, limit=0, shard_count=0):
    if shard_size < 1 or limit < 0 or shard_count < 0:
        raise ValueError('Invalid shard size, count or limit')
    with Path(synthetic).open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Empty teacher dataset')
    ids = set()
    by_tuple = {}
    for row in rows:
        identifier = row['id']
        key = (row['source_lang'], row['source_text'].strip(), row['badini'].strip())
        if identifier in ids or key in by_tuple or key[0] != 'en' or not key[1] or not valid_badini(key[2]):
            raise ValueError(f'Invalid or duplicate synthetic row {identifier}')
        ids.add(identifier)
        by_tuple[key] = row
    training, heldout = split_pairs(list(by_tuple))
    if limit:
        training = training[:limit]
    random.Random(218).shuffle(training)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    shards = []
    if shard_count:
        if len(training) < shard_count:
            raise ValueError('Too few rows for requested shard count')
        quotient, extra = divmod(len(training), shard_count)
        groups = []
        offset = 0
        for i in range(shard_count):
            length = quotient + (i < extra)
            groups.append(training[offset:offset + length])
            offset += length
    else:
        groups = [training[offset:offset + shard_size] for offset in range(0, len(training), shard_size)]
    for group in groups:
        name = f'{len(shards)+1:04}.csv'
        with (root / name).open('x', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'source_lang', 'source_text', 'badini'])
            writer.writeheader()
            for key in group:
                writer.writerow({k: by_tuple[key][k] for k in writer.fieldnames})
        shards.append({'number': len(shards)+1, 'file': name, 'count': len(group)})
    manifest = {'script_passed_rows': len(rows), 'heldout_rows': len(heldout),
                'training_rows': len(training), 'limit': limit, 'shards': shards,
                'warning': 'Teacher outputs are unverified; Arabic script is not proof of Badini or correctness.'}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--synthetic', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--shard-size', type=int, default=5000)
    p.add_argument('--limit', type=int, default=0, help='For small pipeline tests only')
    p.add_argument('--shard-count', type=int, default=0, help='Split training rows into exactly this many shards instead of using shard-size')
    args = p.parse_args()
    result = prepare(args.synthetic, args.output, args.shard_size, args.limit, args.shard_count)
    print('::notice title=Training shard manifest::' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
