"""Guarded publication of the selected independent 20-candidate experiment.

Only the winning adapter is published; other independent candidates are not
merged. A separately running historical-data chain will retain its pinned
baseline and must be reconciled before a later publication.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.hf_training_round import FILES, REPO, check_adapter, decide, publish, snapshot

WINNER = 'long-candidate-careful-9000'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).digest()


def prepare_decision(baseline, original, selected, candidate):
    baseline, original, selected, candidate = map(Path, (baseline, original, selected, candidate))
    original = original / 'accepted-adapter'
    selected_adapter = selected / 'accepted-adapter'
    candidate_adapter = candidate / 'adapter'
    for folder in (baseline, original, selected_adapter, candidate_adapter):
        check_adapter(folder)
    for name in FILES:
        if digest(baseline / name) != digest(original / name):
            raise ValueError('Current Hub baseline differs from the model used to train candidates')
        if digest(selected_adapter / name) != digest(candidate_adapter / name):
            raise ValueError('Selected adapter differs from the reported winning candidate')
    selection = json.loads((selected / 'selection.json').read_text(encoding='utf-8'))
    if selection['selected'] != WINNER:
        raise ValueError('Unexpected selected candidate')
    matches = [x for x in selection['candidates'] if x['name'] == WINNER]
    if len(matches) != 1 or not matches[0]['loss_pass'] or not matches[0]['word_pass']:
        raise ValueError('Selected candidate did not pass both selection guards')
    report = json.loads((candidate / 'report.json').read_text(encoding='utf-8'))
    words = json.loads((candidate / 'word-evaluation.json').read_text(encoding='utf-8'))
    result = decide(report, words)
    if not result['candidate_accepted']:
        raise ValueError('Winning candidate failed independent guard recheck')
    row = matches[0]
    if (abs(row['overall_loss_before'] - result['heldout_loss_before']) > 1e-6 or
        abs(row['overall_loss_after'] - result['heldout_loss_after']) > 1e-6 or
        row['word_exact_before'] != result['word_exact_before'] or
        row['word_exact_after'] != result['word_exact_after']):
        raise ValueError('Selection and candidate reports disagree')
    result['dataset_description'] = ('unverified teacher-agreement pairs; this is the careful-9000 winner '
        'from [20 independent candidates](https://github.com/s1po0/Translate-badini-/actions/runs/41052181009), '
        'not a merge of all candidates')
    result['selected_experiment_run'] = '41052181009'
    result['selected_candidate'] = WINNER
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'selected', 'candidate', 'work'):
        p.add_argument('--'+name, required=True)
    args = p.parse_args()
    root = Path(args.work)
    root.mkdir(parents=True, exist_ok=False)
    snapshot(root/'baseline', root/'baseline-ref.json')
    decision = prepare_decision(root/'baseline', args.original, args.selected, args.candidate)
    (root/'decision.json').write_text(json.dumps(decision, indent=2)+'\n', encoding='utf-8')
    print('::notice title=Selected candidate publication decision::'+json.dumps(decision), flush=True)
    from huggingface_hub import HfApi
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError('HF_TOKEN is missing')
    publish(HfApi(token=token), Path(args.selected)/'accepted-adapter', root/'baseline',
            json.loads((root/'baseline-ref.json').read_text()), decision,
            os.environ['GITHUB_RUN_ID'], root/'upload')


if __name__ == '__main__':
    main()
