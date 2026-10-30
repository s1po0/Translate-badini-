"""Run and evaluate one word-training shard against the preceding checkpoint."""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from continuous_training import accepted
from train import read_pairs, split_pairs
from train_adapter import select_examples

ROOT = Path(__file__).resolve().parents[1]


def plan(shard, stage, replay):
    with Path(shard).open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    tuples = [(r['source_lang'], r['source_text'], r['badini']) for r in rows]
    train, heldout = split_pairs(tuples)
    if not train or heldout:
        raise ValueError('Shard empty or contains held-out sources')
    replay_train, _ = split_pairs(read_pairs(ROOT / 'training_data/ferman_approximate_sorani_badini.csv'))
    replay_count = len(select_examples(replay_train, replay, 42 + stage))
    return len(train), len(train) + replay_count


def decide(report, word):
    loss_pass = accepted(report, .03)
    word_pass = word['after_exact'] >= word['before_exact'] - max(1, int(word['examples'] * .02))
    return loss_pass, word_pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=int, required=True)
    p.add_argument('--shards', required=True)
    p.add_argument('--previous', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--replay-per-language', type=int, default=200)
    args = p.parse_args()
    if args.stage < 1 or args.replay_per_language < 1:
        p.error('stage and replay must be positive')
    root, previous, shards = Path(args.output), Path(args.previous), Path(args.shards)
    manifest = json.loads((shards / 'manifest.json').read_text())
    if args.stage > len(manifest['shards']):
        raise ValueError('Stage missing from shard manifest')
    entry = manifest['shards'][args.stage-1]
    if entry['number'] != args.stage:
        raise ValueError('Unexpected shard number')
    shard = shards / entry['file']
    candidates = list(previous.rglob('accepted-adapter/adapter_config.json'))
    if len(candidates) != 1:
        raise ValueError('Expected one accepted previous adapter')
    adapter = candidates[0].parent
    history_paths = list(previous.rglob('progress.json'))
    if args.stage == 1:
        history = []
    else:
        if len(history_paths) != 1:
            raise ValueError('Previous progress is missing')
        history = json.loads(history_paths[0].read_text())['stages']
        if len(history) != args.stage - 1 or history[-1]['stage'] != args.stage - 1:
            raise ValueError('Cannot skip or repeat a stage')
    count, steps = plan(shard, args.stage, args.replay_per_language)
    if count != entry['count']:
        raise ValueError('Shard count does not match manifest')
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / 'candidate'
    cmd = [sys.executable, str(ROOT / 'train_adapter.py'), '--initial-adapter', str(adapter),
           '--synthetic-data', str(shard), '--synthetic-repeats', '1', '--steps', str(steps),
           '--per-language', str(args.replay_per_language), '--eval-per-language', '20',
           '--learning-rate', '0.00002', '--seed', str(42+args.stage), '--output', str(candidate)]
    subprocess.run(cmd, check=True, cwd=ROOT)
    report = json.loads((candidate / 'report.json').read_text())
    if report['synthetic_unique_rows'] != count or report['optimizer_steps'] != report['training_pool_pairs']:
        raise ValueError('Candidate did not cover its full training pool')
    subprocess.run([sys.executable, str(ROOT / 'tools/evaluate_word_quality.py'), '--before', str(adapter),
                    '--after', str(candidate / 'adapter'), '--size', '100', '--output', str(root / 'word-evaluation.json')], check=True, cwd=ROOT)
    word = json.loads((root / 'word-evaluation.json').read_text())
    loss_pass, word_pass = decide(report, word)
    keep = loss_pass and word_pass
    shutil.copytree(candidate / 'adapter' if keep else adapter, root / 'accepted-adapter')
    (root / 'training-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    record = {'stage': args.stage, 'examples_seen': count, 'optimizer_steps': steps,
              'candidate_accepted': keep, 'loss_guard_pass': loss_pass, 'word_guard_pass': word_pass}
    history.append(record)
    progress = {'stages': history, 'examples_seen_in_attempts': sum(x['examples_seen'] for x in history),
                'examples_in_accepted_stages': sum(x['examples_seen'] for x in history if x['candidate_accepted']),
                'planned_examples': manifest['training_rows'], 'planned_stages': len(manifest['shards']),
                'limitation': 'Outputs are unverified, rejected stages are not retained; repeated gates and small evaluations do not establish translation accuracy.'}
    (root / 'progress.json').write_text(json.dumps(progress, indent=2), encoding='utf-8')
    print('::notice title=Word training stage::' + json.dumps(record), flush=True)
    print('::notice title=Word training progress::' + json.dumps(progress), flush=True)
    # Only accepted checkpoint and reports are uploaded by the workflow; candidates stay on the runner.


if __name__ == '__main__':
    main()
