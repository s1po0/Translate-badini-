"""Train a candidate LoRA adapter on a private trainer backup and guard promotion.

Runs locally on an appropriately provisioned trusted machine. Never upload user
backups or trained adapters to a public repository or public workflow artifacts:
model weights may memorize private examples. This is not an online service.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from continuous_training import accepted
from train import read_pairs, split_pairs
from train_adapter import read_reviewed_examples, remove_reviewed_sources, select_examples

ROOT = Path(__file__).resolve().parents[1]


def training_steps(backup, replay, repeats, seed=42):
    reviewed = read_reviewed_examples(backup)
    replay_rows, _ = split_pairs(read_pairs(ROOT / 'training_data/ferman_approximate_sorani_badini.csv'))
    replay_rows = select_examples(remove_reviewed_sources(replay_rows, reviewed), replay, seed)
    return len(replay_rows) + len(reviewed) * repeats


def decide(report, word):
    loss_pass = accepted(report, .03)
    # Do not accept a decline in exact matches on this independent sample.
    word_pass = word['after_exact'] >= word['before_exact']
    return loss_pass, word_pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup', required=True, help='Private JSON exported from the trainer')
    parser.add_argument('--initial-adapter', required=True)
    parser.add_argument('--output', required=True, help='New private output directory')
    parser.add_argument('--replay-per-language', type=int, default=200)
    parser.add_argument('--reviewed-repeats', type=int, default=5)
    parser.add_argument('--eval-per-language', type=int, default=20)
    parser.add_argument('--word-eval-size', type=int, default=100)
    args = parser.parse_args()
    if min(args.replay_per_language, args.reviewed_repeats, args.eval_per_language, args.word_eval_size) < 1:
        parser.error('All sample sizes must be positive')
    backup = Path(args.backup).resolve(strict=True)
    adapter = Path(args.initial_adapter).resolve(strict=True)
    if not (adapter / 'adapter_config.json').is_file():
        raise ValueError('Initial adapter is missing or incomplete')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    count = len(read_reviewed_examples(backup))
    steps = training_steps(backup, args.replay_per_language, args.reviewed_repeats)
    candidate = out / 'candidate'
    cmd = [sys.executable, str(ROOT / 'train_adapter.py'), '--initial-adapter', str(adapter),
           '--reviewed-data', str(backup), '--reviewed-repeats', str(args.reviewed_repeats),
           '--steps', str(steps), '--per-language', str(args.replay_per_language),
           '--eval-per-language', str(args.eval_per_language), '--learning-rate', '0.00002',
           '--seed', '42', '--output', str(candidate)]
    subprocess.run(cmd, check=True, cwd=ROOT)
    report = json.loads((candidate / 'report.json').read_text())
    if report['reviewed_unique_rows'] != count or report['optimizer_steps'] != report['training_pool_pairs']:
        raise ValueError('Training did not cover the reviewed training pool')
    word_file = out / 'word-evaluation.json'
    subprocess.run([sys.executable, str(ROOT / 'tools/evaluate_word_quality.py'),
                    '--before', str(adapter), '--after', str(candidate / 'adapter'),
                    '--size', str(args.word_eval_size), '--exclude-reviewed', str(backup),
                    '--output', str(word_file)], cwd=ROOT, check=True)
    word = json.loads(word_file.read_text())
    loss_pass, word_pass = decide(report, word)
    keep = loss_pass and word_pass
    shutil.copytree(candidate / 'adapter' if keep else adapter, out / 'accepted-adapter')
    result = {'reviewed_examples_in_attempt': count, 'optimizer_steps': steps,
              'candidate_accepted': keep, 'loss_guard_pass': loss_pass, 'word_guard_pass': word_pass,
              'limitation': 'A training pass does not prove the model learned each example; a rejected candidate leaves the previous adapter intact. Small tests do not establish real-world quality.'}
    (out / 'selection.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('Reviewed training finished: ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
