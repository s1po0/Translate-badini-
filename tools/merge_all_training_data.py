"""Unify accessible public reference and earlier teacher-training pairs.

The foundation model's original pretraining data and user's local browser data
are not accessible. Synthetic outputs are unverified; this does not prove that
any translation is correct. Exact duplicate pairs are counted and deduplicated.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import read_pairs
from synthetic_batch import valid_badini


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not {'source_lang','source_text','badini'}.issubset(reader.fieldnames or []):
            raise ValueError(f'Invalid training CSV header in {path}')
        for row in reader:
            yield (row['source_lang'],row['source_text'],row['badini']), row.get('id','')


def one(root, pattern):
    files = list(Path(root).rglob(pattern))
    if len(files) != 1: raise ValueError(f'Expected one {pattern} in {root}; got {len(files)}')
    return files[0]


def merge(seed_rows, datasets, output):
    origins = defaultdict(list); counts = Counter()
    def add(name, row, identifier=''):
        lang, source, target = row
        lang, source, target = lang.strip(), source.strip(), target.strip()
        if lang not in ('en','ar','sorani') or not source or not valid_badini(target):
            raise ValueError(f'Invalid row in {name}; no training data was silently discarded')
        key=(lang,source,target);counts[name] += 1
        origins[key].append((name,identifier))
    for row in seed_rows: add('reference',row)
    if not counts['reference']: raise ValueError('Reference data is empty')
    for name, paths in datasets.items():
        if not paths: raise ValueError(f'Missing dataset {name}')
        for path in paths:
            before=counts[name]
            for row, identifier in read_csv(path): add(name,row,identifier)
            if counts[name] == before: raise ValueError(f'Empty dataset file: {path}')
    dest=Path(output); dest.mkdir(parents=True,exist_ok=False)
    with (dest/'synthetic.csv').open('w',encoding='utf-8',newline='') as f, (dest/'provenance.csv').open('w',encoding='utf-8',newline='') as g:
        out=csv.writer(f); prov=csv.writer(g)
        out.writerow(['id','source_lang','source_text','badini']);prov.writerow(['id','dataset','original_id'])
        for (lang,source,target), entries in sorted(origins.items()):
            ident=hashlib.sha256(json.dumps([lang,source,target],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
            out.writerow([ident,lang,source,target])
            for name, original_id in entries:prov.writerow([ident,name,original_id])
    report={'raw_rows_by_dataset':dict(sorted(counts.items())), 'raw_rows':sum(counts.values()),
            'unique_pairs':len(origins),'exact_duplicate_rows':sum(counts.values())-len(origins),
            'limitations':'Only accessible historical datasets; original base-model training data and private browser corrections are unavailable. Synthetic labels are unverified; duplication removal and exposure do not prove learning.'}
    (dest/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('::notice title=Accessible training data coverage::'+json.dumps(report),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for x in ('english','badini','agreement20','agreement8','agreement4','continuous','balanced','pilot','corpus_pilot','output'):
        p.add_argument('--'+x.replace('_','-'),required=True)
    a=p.parse_args()
    continuous=list(Path(a.continuous).rglob('synthetic.csv'))
    if len(continuous)!=2:raise ValueError(f'Expected two continuous-training datasets, found {len(continuous)}')
    datasets={
        'all_english_teacher':[one(a.english,'synthetic.csv')],
        'badini_corpus_teacher':[one(a.badini,'synthetic.csv')],
        'teacher_agreement_20':[one(a.agreement20,'synthetic.csv')],
        'teacher_agreement_8':[one(a.agreement8,'synthetic.csv')],
        'teacher_agreement_4':[one(a.agreement4,'synthetic.csv')],
        'continuous_teacher':sorted(continuous),
        'dictionary_checked_teacher':[one(a.balanced,'checked.csv')],
        'english_pilot_teacher':[one(a.pilot,'synthetic.csv')],
        'corpus_pilot_teacher':[one(a.corpus_pilot,'synthetic.csv')],
    }
    merge(read_pairs('training_data/ferman_approximate_sorani_badini.csv'),datasets,a.output)

if __name__=='__main__':main()
