"""Match outputs from two related Badini model versions, without claiming correctness."""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synthetic_batch import valid_badini
from tools.filter_dictionary_agreement import normalize


def load_attempts(path):
    by_id = {}
    with Path(path).open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if not {'id', 'english', 'badini', 'script_valid'}.issubset(reader.fieldnames or []):
            raise ValueError('Attempt file has an invalid header')
        for row in reader:
            if not row['id'].isdigit() or row['id'] in by_id:
                raise ValueError('Invalid or duplicate ID in attempt file')
            by_id[row['id']] = row
    return by_id


def compare(old_path, adapted_path, output):
    old, adapted = load_attempts(old_path), load_attempts(adapted_path)
    if not adapted:
        raise ValueError('No adapted-teacher attempts')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(attempted=len(adapted))
    with output.open('x', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'source_lang', 'source_text', 'badini'])
        for identifier, row in adapted.items():
            original = old.get(identifier)
            if original is None or original['english'] != row['english']:
                raise ValueError('The teachers do not cover the same numbered word')
            one, two = original['badini'].strip(), row['badini'].strip()
            if valid_badini(one) and valid_badini(two):
                counts['both_script_valid'] += 1
                if normalize(one) and normalize(one) == normalize(two):
                    counts['normalized_agreement'] += 1
                    writer.writerow([identifier, 'en', row['english'], two])
    report = dict(counts)
    report.setdefault('both_script_valid', 0)
    report.setdefault('normalized_agreement', 0)
    report['limitation'] = 'Both teachers are related model versions. Script and normalized agreement do not prove Badini dialect or translation correctness.'
    return report


def combine(inputs, output, report_path):
    files = sorted(Path(inputs).rglob('agreement.csv'))
    if not files:
        raise ValueError('No agreement files found')
    seen = set(); count = 0
    with Path(output).open('x', encoding='utf-8', newline='') as dest:
        writer = csv.writer(dest)
        writer.writerow(['id', 'source_lang', 'source_text', 'badini'])
        for file in files:
            with file.open(encoding='utf-8', newline='') as source:
                reader = csv.DictReader(source)
                if not {'id','source_lang','source_text','badini'}.issubset(reader.fieldnames or []):
                    raise ValueError('Invalid agreement file')
                for row in reader:
                    if row['id'] in seen or row['source_lang'] != 'en' or not valid_badini(row['badini']):
                        raise ValueError('Duplicate or invalid agreement')
                    seen.add(row['id']); count += 1
                    writer.writerow([row['id'], row['source_lang'], row['source_text'], row['badini']])
    if not count:
        raise ValueError('No agreed pairs available for training')
    report = {'batches':len(files),'agreement_pairs':count,'limitation':'Agreement between related teachers and script checks are not correctness verification; outputs remain synthetic.'}
    Path(report_path).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('::notice title=Teacher agreement dataset::' + json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    comp = sub.add_parser('compare'); comp.add_argument('--old',required=True); comp.add_argument('--adapted',required=True); comp.add_argument('--output',required=True); comp.add_argument('--report',required=True)
    merge = sub.add_parser('combine'); merge.add_argument('--input',required=True); merge.add_argument('--output',required=True); merge.add_argument('--report',required=True)
    args = parser.parse_args()
    if args.command == 'compare':
        report = compare(args.old,args.adapted,args.output)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('::notice title=Teacher version comparison::' + json.dumps(report), flush=True)
    else: combine(args.input,args.output,args.report)

if __name__ == '__main__': main()
