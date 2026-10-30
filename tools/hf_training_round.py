"""Train from a pinned Hugging Face adapter and safely publish a guarded update."""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from continuous_training import accepted
from tools.publish_hf_adapter import BASE, CARD

REPO = 'sipan69/Badini-ai'
FILES = ('adapter_config.json', 'adapter_model.safetensors', 'tokenizer.json',
         'tokenizer_config.json', 'special_tokens_map.json', 'sentencepiece.bpe.model')
START = '<!-- BADINI_TRAINING_START -->'
END = '<!-- BADINI_TRAINING_END -->'


def check_adapter(directory):
    path = Path(directory)
    if any(not (path / name).is_file() for name in FILES):
        raise ValueError('Adapter or tokenizer files are missing')
    if json.loads((path / 'adapter_config.json').read_text())['base_model_name_or_path'] != BASE:
        raise ValueError('Unexpected base model')


def decide(report, words):
    if not isinstance(words.get('examples'), int) or words['examples'] < 1:
        raise ValueError('Invalid word evaluation')
    before, after = words.get('before_exact'), words.get('after_exact')
    if not isinstance(before, int) or not isinstance(after, int) or not (0 <= before <= words['examples'] and 0 <= after <= words['examples']):
        raise ValueError('Invalid exact-match scores')
    return {'candidate_accepted': accepted(report, .03) and after >= before,
            'loss_guard_pass': accepted(report, .03), 'word_guard_pass': after >= before,
            'heldout_loss_before': report['teacher_forced_loss_before']['overall'],
            'heldout_loss_after': report['teacher_forced_loss_after']['overall'],
            'word_exact_before': before, 'word_exact_after': after, 'word_examples': words['examples'],
            'optimizer_steps': report['optimizer_steps'], 'teacher_pairs_in_training_pool': report['synthetic_unique_rows'],
            'limitation': 'Synthetic teacher agreement is not verified truth; loss and exact matches do not establish translation accuracy.'}


def update_card(original, decision, run_id):
    before = decision['heldout_loss_before']; after = decision['heldout_loss_after']
    line = (f'{START}\nThe current adapter is a guarded fine-tuning of the previous Hugging Face checkpoint. '
            f'[Training run {run_id}](https://github.com/s1po0/Translate-badini-/actions/runs/{run_id}) '
            f'used {decision["teacher_pairs_in_training_pool"]} unverified teacher-agreement pairs. '
            f'Held-out loss was {before:.4f} before and {after:.4f} after; exact dictionary matches '
            f'were {decision["word_exact_before"]}/{decision["word_examples"]} before and '
            f'{decision["word_exact_after"]}/{decision["word_examples"]} after. '
            'These small tests do not establish translation quality. The prior checkpoint is available in repository history.\n' + END)
    if START in original or END in original:
        if original.count(START) != 1 or original.count(END) != 1:
            raise ValueError('Model card markers are inconsistent; refusing to replace user text')
        start, end = original.index(START), original.index(END) + len(END)
        if end <= start:
            raise ValueError('Model card markers are out of order')
        return original[:start] + line + original[end:]
    old_summary = CARD.split('\n\n')[2]
    if original.count(old_summary) != 1:
        raise ValueError('Model card was changed; refusing to overwrite it')
    return original.replace(old_summary, line, 1)


def snapshot(directory, ref_file):
    from huggingface_hub import HfApi, snapshot_download
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError('HF_TOKEN is not configured')
    api = HfApi(token=token)
    info = api.repo_info(repo_id=REPO, repo_type='model')
    if not re.fullmatch('[0-9a-f]{40}', info.sha or ''):
        raise ValueError('Invalid Hugging Face baseline revision')
    snapshot_download(repo_id=REPO, repo_type='model', revision=info.sha, token=token,
                      local_dir=directory, allow_patterns=list(FILES) + ['README.md', 'selection.json'])
    check_adapter(directory)
    if not (Path(directory) / 'README.md').is_file():
        raise ValueError('Baseline model card is missing')
    Path(ref_file).write_text(json.dumps({'repo': REPO, 'revision': info.sha})+'\n', encoding='utf-8')
    print('Captured the current Hugging Face adapter revision.', flush=True)


def publish(api, candidate, baseline, reference, decision, run_id, output):
    if not decision.get('candidate_accepted'):
        raise ValueError('Cannot publish a rejected candidate')
    if not run_id.isdigit() or reference.get('repo') != REPO or not re.fullmatch('[0-9a-f]{40}', reference.get('revision', '')):
        raise ValueError('Invalid run or repository revision')
    check_adapter(candidate)
    check_adapter(baseline)
    old_weights = hashlib.sha256((Path(baseline) / 'adapter_model.safetensors').read_bytes()).digest()
    new_weights = hashlib.sha256((Path(candidate) / 'adapter_model.safetensors').read_bytes()).digest()
    if old_weights == new_weights:
        raise ValueError('Candidate weights did not change; refusing to publish')
    current = api.repo_info(repo_id=REPO, repo_type='model')
    if current.sha != reference['revision']:
        raise ValueError('Hugging Face changed during training; previous model was left untouched')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name in FILES:
        shutil.copy2(Path(candidate) / name, output / name)
    old_card = (Path(baseline) / 'README.md').read_text(encoding='utf-8')
    (output / 'README.md').write_text(update_card(old_card, decision, run_id), encoding='utf-8')
    history = output / 'training-history'
    history.mkdir()
    record = dict(decision, run_id=run_id, baseline_revision=reference['revision'], dataset='20-batch agreement of two related Badini model versions')
    (history / f'{run_id}.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    existing = api.list_repo_files(repo_id=REPO, repo_type='model', revision=reference['revision'])
    if f'training-history/{run_id}.json' in existing:
        raise ValueError('This training run already exists on the Hub')
    result = api.upload_folder(repo_id=REPO, repo_type='model', folder_path=str(output),
                               parent_commit=reference['revision'], commit_message='Update Badini adapter after guarded training')
    # Verify the new weight file against the local candidate, without printing credentials.
    from huggingface_hub import hf_hub_download
    downloaded = hf_hub_download(repo_id=REPO, repo_type='model', filename='adapter_model.safetensors', token=api.token, force_download=True)
    local_hash = hashlib.sha256((output / 'adapter_model.safetensors').read_bytes()).digest()
    remote_hash = hashlib.sha256(Path(downloaded).read_bytes()).digest()
    if remote_hash != local_hash:
        raise RuntimeError('Uploaded model did not match the candidate; inspect the Hub before retrying')
    print('Verified guarded model update on Hugging Face.', flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    s=sub.add_parser('snapshot'); s.add_argument('--output',required=True); s.add_argument('--ref',required=True)
    d=sub.add_parser('decide'); d.add_argument('--report',required=True); d.add_argument('--words',required=True); d.add_argument('--output',required=True)
    q=sub.add_parser('publish'); q.add_argument('--candidate',required=True); q.add_argument('--baseline',required=True); q.add_argument('--ref',required=True); q.add_argument('--decision',required=True); q.add_argument('--run-id',required=True); q.add_argument('--output',required=True)
    a=p.parse_args()
    if a.action == 'snapshot': snapshot(a.output,a.ref)
    elif a.action == 'decide':
        result=decide(json.loads(Path(a.report).read_text()),json.loads(Path(a.words).read_text()))
        Path(a.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print('::notice title=Guarded training decision::' + json.dumps(result),flush=True)
        if os.environ.get('GITHUB_OUTPUT'):
            with open(os.environ['GITHUB_OUTPUT'],'a') as f:f.write('accepted='+str(result['candidate_accepted']).lower()+'\n')
    else:
        from huggingface_hub import HfApi
        token=os.environ.get('HF_TOKEN')
        if not token: raise ValueError('HF_TOKEN is not configured')
        publish(HfApi(token=token),a.candidate,a.baseline,json.loads(Path(a.ref).read_text()),json.loads(Path(a.decision).read_text()),a.run_id,a.output)

if __name__=='__main__': main()
