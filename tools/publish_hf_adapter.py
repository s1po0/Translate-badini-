"""Publish one existing selected adapter to a specifically authorized HF model repo.

Never print or store HF_TOKEN. Existing remote files are not replaced; the Hub
commit uses the checked parent SHA to avoid a race with another writer.
"""
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

BASE = 'arkanit/badini-nllb-translator'
CARD = '''---
library_name: peft
base_model: arkanit/badini-nllb-translator
tags:
- translation
- badini
- lora
license: other
---
# Experimental Badini translation adapter

This is a LoRA adapter, **not a complete standalone model**. Load it using PEFT with the base model `arkanit/badini-nllb-translator`. It is intended for English, Arabic and Sorani to Badini. Reverse translation is not supported by this adapter.

The selected checkpoint comes from [run 41047507896](https://github.com/s1po0/Translate-badini-/actions/runs/41047507896), `parallel-candidate-balanced`. Held-out mean loss was 3.00359 before and 2.95043 after; exact word matches were 6/100 before and after. These small tests do not establish translation quality. Synthetic labels and dialect are unverified; native speaker review is needed. Later training is not included unless explicitly uploaded.

Sources include [Jad-deeb Kurdish Behdini resources](https://github.com/Jad-deeb/Kurdish-behdini-language-resources) (MIT), [Ferman data](https://huggingface.co/datasets/OmarSY11/ferman-nlu) (CC BY 4.0), and [Dasinya corpus](https://zenodo.org/records/20729060) (CC BY 4.0), plus synthetic labels. The base model derives from Meta NLLB; upstream [NLLB-200 distilled](https://huggingface.co/facebook/nllb-200-distilled-600M) is CC BY-NC 4.0 while its Badini derivative reports a different license. Review all underlying licenses and provenance before use, particularly commercial use. No rights are granted beyond the applicable underlying terms.
'''


def package_adapter(source, target, existing):
    configs = list(Path(source).rglob('accepted-adapter/adapter_config.json'))
    if len(configs) != 1:
        raise ValueError('Expected exactly one selected adapter')
    adapter = configs[0].parent
    config = json.loads(configs[0].read_text(encoding='utf-8'))
    if config.get('base_model_name_or_path') != BASE or not (adapter / 'adapter_model.safetensors').is_file():
        raise ValueError('Selected adapter is incomplete or has an unexpected base')
    selections = list(Path(source).rglob('selection.json'))
    if len(selections) != 1 or json.loads(selections[0].read_text())['selected'] != 'parallel-candidate-balanced':
        raise ValueError('Selected checkpoint report is missing or unexpected')
    shutil.copytree(adapter, target)
    target = Path(target)
    # Replace the auto-generated local card only; never overwrite a remote card.
    (target / 'README.md').unlink(missing_ok=True)
    card = 'README.md' if 'README.md' not in existing else 'BADINI_MODEL_INFO.md'
    (target / card).write_text(CARD, encoding='utf-8')
    shutil.copy2(selections[0], target / 'selection.json')
    files = sorted(p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file())
    clashes = sorted(set(files) & set(existing))
    if clashes:
        raise ValueError('Remote repository already contains model files; refusing to replace them: ' + ', '.join(clashes))
    return files


def publish(api, source, repo):
    info = api.repo_info(repo_id=repo, repo_type='model')
    if not info.sha:
        raise ValueError('Remote repository has no stable revision; refusing to publish')
    existing = api.list_repo_files(repo_id=repo, repo_type='model', revision=info.sha)
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / 'model'
        files = package_adapter(source, target, existing)
        api.upload_folder(repo_id=repo, repo_type='model', folder_path=str(target), parent_commit=info.sha,
                          commit_message='Add experimental Badini LoRA adapter')
        # Verify remote file inventory and adapter config after commit.
        remote = set(api.list_repo_files(repo_id=repo, repo_type='model'))
        if not set(files) <= remote:
            raise RuntimeError('Upload returned but some model files could not be verified')
        import hashlib
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(repo_id=repo, repo_type='model', filename='adapter_config.json', token=api.token)
        if hashlib.sha256(Path(downloaded).read_bytes()).digest() != hashlib.sha256((target / 'adapter_config.json').read_bytes()).digest():
            raise RuntimeError('Uploaded adapter config did not match local config')
    print('Verified adapter upload: https://huggingface.co/' + repo, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--repo', required=True)
    args = parser.parse_args()
    if args.repo != 'sipan69/Badini-ai':
        raise ValueError('This workflow is restricted to the user-provided repository')
    token = os.environ.get('HF_TOKEN', '')
    if not token:
        raise ValueError('HF_TOKEN GitHub secret is missing')
    from huggingface_hub import HfApi
    publish(HfApi(token=token), args.source, args.repo)

if __name__ == '__main__':
    main()
