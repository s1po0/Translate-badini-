"""Continuously distill local teachers into a personal Badini LoRA adapter."""
import argparse
import csv
import fcntl
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from distill_badini import BADINI_MODEL, PIVOT_MODEL
from train import read_pairs, split_pairs

ROOT = Path(__file__).resolve().parent


def source_pool():
    train, _ = split_pairs(read_pairs(ROOT / 'training_data/ferman_approximate_sorani_badini.csv'))
    pools = {lang: sorted({source for language, source, _ in train
                            if language == lang and len(source) <= 180 and len(source.split()) <= 25})
             for lang in ('en', 'ar', 'sorani')}
    for lang, pool in pools.items():
        if not pool:
            raise ValueError(f'No training sources for {lang}')
        random.Random(42).shuffle(pool)
    return pools


def select_sources(pools, cycle, per_language):
    result = []
    for language, items in pools.items():
        for offset in range(min(per_language, len(items))):
            result.append((language, items[((cycle - 1) * per_language + offset) % len(items)]))
    return result


def accepted(report, tolerance):
    before, after = report['teacher_forced_loss_before'], report['teacher_forced_loss_after']
    for key in ('overall',):
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (before[key], after[key])):
            return False
        if after[key] > before[key]:
            return False
    for lang, old in before['by_language'].items():
        new = after['by_language'].get(lang)
        if old is None or new is None or not math.isfinite(old) or not math.isfinite(new):
            return False
        if new > old * (1 + tolerance):
            return False
    return True


def save_state(path, state):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def run_cycle(args, state, pools, run=subprocess.run):
    cycle = state['cycle'] + 1
    teacher = args.teachers[(cycle - 1) % len(args.teachers)]
    folder = Path(args.output).resolve() / 'rounds' / f'{cycle:06d}-{uuid.uuid4().hex[:8]}'
    folder.mkdir(parents=True, exist_ok=False)
    source_file = folder / 'sources.csv'
    with source_file.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source_lang', 'source_text'])
        writer.writerows(select_sources(pools, cycle, args.batch_per_language))
    env = os.environ.copy()
    if not args.allow_download:
        env['HF_HUB_OFFLINE'] = '1'
        env['TRANSFORMERS_OFFLINE'] = '1'
    synthetic = folder / 'synthetic.csv'
    run([sys.executable, str(ROOT / 'synthetic_batch.py'), '--input', str(source_file), '--output', str(synthetic),
         '--teacher', teacher, '--pivot', args.pivot], cwd=ROOT, env=env, check=True)
    command = [sys.executable, str(ROOT / 'train_adapter.py'), '--base', args.base,
               '--data', str(ROOT / 'training_data/ferman_approximate_sorani_badini.csv'),
               '--synthetic-data', str(synthetic), '--output', str(folder / 'model'),
               '--steps', str(args.steps), '--per-language', str(args.replay_per_language),
               '--eval-per-language', str(args.eval_per_language), '--seed', str(42 + cycle)]
    if state.get('adapter'):
        command.extend(['--initial-adapter', str(Path(args.output).resolve() / state['adapter'])])
    run(command, cwd=ROOT, env=env, check=True)
    report_path = folder / 'model/report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    keep = accepted(report, args.max_regression)
    adapter = folder / 'model/adapter'
    if not (adapter / 'adapter_config.json').is_file():
        raise ValueError('Training did not produce a complete adapter')
    relative_adapter = str(adapter.relative_to(Path(args.output).resolve()))
    new_state = dict(state, cycle=cycle, adapter=relative_adapter if keep else state.get('adapter'),
                     last_round=str(folder.relative_to(Path(args.output).resolve())), last_accepted=keep, last_teacher=teacher)
    save_state(Path(args.output).resolve() / 'state.json', new_state)
    print(f'Round {cycle}: {"accepted" if keep else "rejected; retained previous adapter"}; teacher={teacher}', flush=True)
    return new_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='models/continuous')
    parser.add_argument('--initial-adapter', help='Existing LoRA adapter, used only for a new run')
    parser.add_argument('--base', default=BADINI_MODEL)
    parser.add_argument('--teachers', nargs='+', default=[BADINI_MODEL], help='Compatible local NLLB Badini teacher models; rotates between rounds')
    parser.add_argument('--pivot', default=PIVOT_MODEL)
    parser.add_argument('--batch-per-language', type=int, default=2)
    parser.add_argument('--replay-per-language', type=int, default=4)
    parser.add_argument('--eval-per-language', type=int, default=8)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--max-cycles', type=int, default=0, help='0 keeps running until stopped')
    parser.add_argument('--max-regression', type=float, default=0.03)
    parser.add_argument('--allow-download', action='store_true', help='Allow downloading missing model weights')
    parser.add_argument('--retry-seconds', type=int, default=60)
    args = parser.parse_args()
    if min(args.batch_per_language, args.replay_per_language, args.eval_per_language, args.steps, args.retry_seconds) < 1 or args.max_cycles < 0 or not 0 <= args.max_regression <= 1:
        parser.error('Batch sizes and steps must be positive; max-cycles must be nonnegative and max-regression in [0,1].')
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / 'state.json'
    with (root / '.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another training process is using this directory.')
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding='utf-8'))
            if state.get('base') != args.base:
                raise ValueError('The saved run uses a different base model.')
            if state.get('adapter') and not (root / state['adapter']).is_dir():
                raise ValueError('The saved adapter directory is missing.')
        else:
            adapter = None
            if args.initial_adapter:
                initial = Path(args.initial_adapter).resolve()
                if not (initial / 'adapter_config.json').is_file():
                    raise ValueError('Initial adapter not found or incomplete.')
                adapter = 'initial-' + uuid.uuid4().hex[:8]
                shutil.copytree(initial, root / adapter)
            state = {'cycle': 0, 'adapter': adapter, 'base': args.base}
            save_state(state_path, state)
        pools = source_pool()
        count = 0
        while args.max_cycles == 0 or count < args.max_cycles:
            if shutil.disk_usage(root).free < 2 * 1024 ** 3:
                raise RuntimeError('Less than 2 GiB of disk space remains; stopped without altering the last accepted adapter.')
            try:
                state = run_cycle(args, state, pools)
            except subprocess.CalledProcessError as exc:
                if args.max_cycles:
                    raise
                print(f'Round failed ({exc}); retrying in {args.retry_seconds}s. The current adapter is unchanged.', flush=True)
                time.sleep(args.retry_seconds)
                continue
            count += 1
    print('Finished. Current adapter:', state.get('adapter') or 'none', flush=True)


if __name__ == '__main__':
    main()
