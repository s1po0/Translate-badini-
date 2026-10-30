"""Select at most one parallel candidate; independent LoRA weights are not merged."""
import argparse
import json
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from continuous_training import accepted


def select(candidates, baseline, output, tolerance=.03):
    directories = sorted(p for p in Path(candidates).iterdir() if p.is_dir())
    if not directories:
        raise ValueError('No parallel candidates')
    results = []
    baseline_exact = None
    baseline_loss = None
    for directory in directories:
        report_files = list(directory.rglob('report.json'))
        evaluation_files = list(directory.rglob('word-evaluation.json'))
        if len(report_files) != 1 or len(evaluation_files) != 1:
            raise ValueError(f'Incomplete candidate {directory.name}')
        report = json.loads(report_files[0].read_text(encoding='utf-8'))
        word = json.loads(evaluation_files[0].read_text(encoding='utf-8'))
        adapter = report_files[0].parent / 'adapter'
        if not (adapter / 'adapter_config.json').is_file():
            raise ValueError(f'Adapter missing for {directory.name}')
        before = report['teacher_forced_loss_before']['overall']
        if baseline_loss is None:
            baseline_loss, baseline_exact = before, word['before_exact']
        elif abs(before-baseline_loss) > 1e-4 or word['before_exact'] != baseline_exact:
            raise ValueError('Candidates do not share the same measured baseline')
        loss_pass = accepted(report, tolerance)
        word_pass = word['after_exact'] >= word['before_exact']
        results.append({'name': directory.name, 'loss_pass': loss_pass, 'word_pass': word_pass,
                        'overall_loss_before': before, 'overall_loss_after': report['teacher_forced_loss_after']['overall'],
                        'word_exact_before': word['before_exact'], 'word_exact_after': word['after_exact'],
                        '_adapter': adapter})
    eligible = [r for r in results if r['loss_pass'] and r['word_pass']]
    winner = min(eligible, key=lambda r: (r['overall_loss_after'], -r['word_exact_after'], r['name'])) if eligible else None
    baseline_paths = list(Path(baseline).rglob('accepted-adapter/adapter_config.json'))
    if len(baseline_paths) != 1:
        raise ValueError('Expected one accepted baseline adapter')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(winner['_adapter'] if winner else baseline_paths[0].parent, root/'accepted-adapter')
    payload = {'selected': winner['name'] if winner else 'baseline',
               'candidates': [{k: v for k, v in item.items() if k != '_adapter'} for item in results],
               'limitation': 'Small non-independent tests, exact matches and loss do not establish real-world quality; synthetic labels are unverified.'}
    (root/'selection.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print('::notice title=Parallel candidate selection::' + json.dumps(payload), flush=True)
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates', required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--output', required=True)
    a=p.parse_args()
    select(a.candidates,a.baseline,a.output)


if __name__ == '__main__':
    main()
