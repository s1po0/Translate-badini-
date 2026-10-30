"""Merge the current private LoRA adapter into a standalone seq2seq model.

The private source adapter and resulting weights never become GitHub artifacts.
Publishing adds only a new subfolder and uses the source commit as parent so a
concurrent Hub change cannot be silently overwritten. This is derived from the
original base, not training from scratch or a claim of improved accuracy.
"""
import gc
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import BASE_MODEL
from tools.hf_training_round import REPO, FILES, check_adapter

SUBFOLDER = 'merged-full-model'


def check_destination(info, remote_files):
    if not info.private:
        raise ValueError('The model repository must remain private')
    if not re.fullmatch('[0-9a-f]{40}', info.sha or ''):
        raise ValueError('Missing stable repository revision')
    if any(f == SUBFOLDER or f.startswith(SUBFOLDER + '/') for f in remote_files):
        raise ValueError('The full-model directory already exists; refusing to overwrite it')


def compare_logits(torch, before, after):
    if len(before) != len(after):
        raise ValueError('Merge verification case count changed')
    for i, (a, b) in enumerate(zip(before, after)):
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f'Invalid verification output for case {i}')
        if not torch.allclose(a, b, rtol=2e-4, atol=2e-4):
            raise ValueError(f'Merged model logits differ from the adapter for case {i}')


def checks(model, tokenizer, torch):
    # Public fixture sentences only; never log user text or generated text.
    cases = [('eng_Latn', 'How are you?'), ('arb_Arab', 'كيف حالك؟'), ('ckb_Arab', 'چۆنی؟')]
    result = []
    with torch.inference_mode():
        for lang, text in cases:
            tokenizer.src_lang = lang
            tokenizer.tgt_lang = 'ckb_Arab'
            inputs = tokenizer(text, return_tensors='pt')
            decoder = torch.tensor([[model.config.decoder_start_token_id]], dtype=torch.long)
            logits = model(**inputs, decoder_input_ids=decoder).logits.detach().cpu()
            result.append(logits)
    return result


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from huggingface_hub import HfApi, snapshot_download

    token = os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError('HF_TOKEN is missing')
    api = HfApi(token=token)
    info = api.repo_info(repo_id=REPO, repo_type='model')
    existing = api.list_repo_files(repo_id=REPO, repo_type='model', revision=info.sha)
    check_destination(info, existing)
    base_info = api.repo_info(repo_id=BASE_MODEL, repo_type='model')
    if not re.fullmatch('[0-9a-f]{40}', base_info.sha or ''):
        raise ValueError('Missing stable base-model revision')
    root = Path('private-merge-work')
    if root.exists():
        raise ValueError('Work directory already exists')
    root.mkdir()
    adapter = root / 'adapter'
    snapshot_download(repo_id=REPO, revision=info.sha, local_dir=str(adapter), token=token,
                      allow_patterns=list(FILES))
    check_adapter(adapter)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter))
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, revision=base_info.sha,
            torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False).eval()
    before = checks(model, tokenizer, torch)
    model = model.merge_and_unload(safe_merge=True).eval()
    after = checks(model, tokenizer, torch)
    compare_logits(torch, before, after)
    output = root / 'merged'
    model.save_pretrained(str(output), safe_serialization=True, max_shard_size='2GB')
    tokenizer.save_pretrained(str(output))
    del model, base
    gc.collect()
    reloaded = AutoModelForSeq2SeqLM.from_pretrained(str(output), torch_dtype=torch.float32,
                                                     low_cpu_mem_usage=True).eval()
    compare_logits(torch, before, checks(reloaded, tokenizer, torch))
    del reloaded, before, after
    gc.collect()
    if not (output/'config.json').is_file() or not list(output.glob('*.safetensors')):
        raise ValueError('Standalone model files are missing')
    (output/'README.md').write_text(
        '# Experimental merged Badini model\n\n'
        'Standalone merged model: the user adapter was merged into the pretrained base model '
        f'[{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) at revision `{base_info.sha}`. '
        'The base derives from Meta NLLB-200 distilled 600M. This model was not trained from scratch.\n\n'
        f'The source adapter revision was `{info.sha}`. Merge outputs were checked against '
        'the unmerged adapter on three fixed inputs; this is not a translation-quality evaluation. '
        'The added adapter was trained for English, Arabic and Sorani to Badini. '
        'Reverse-direction behavior and dialect accuracy require independent testing.\n\n'
        'Attribution and licenses: Arkan Musa Zubair, Badini NLLB Translator (model card says MIT); '
        'Meta NLLB-200 distilled 600M, CC BY-NC 4.0. Dataset sources and limitations are '
        'documented in the repository root model card. Review upstream rights and restrictions '\
        'before any use or redistribution.\n', encoding='utf-8')
    metadata = {'adapter_revision': info.sha, 'base_model': BASE_MODEL, 'base_revision': base_info.sha,
                'verification': 'FP32 first-step logits for three fixed inputs before merge, after merge, and after local reload; not a quality evaluation',
                'private_at_upload': True}
    (output/'merge-provenance.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    latest = api.repo_info(repo_id=REPO, repo_type='model')
    if not latest.private or latest.sha != info.sha:
        raise ValueError('Visibility or repository revision changed during merging; not publishing')
    api.upload_folder(repo_id=REPO, repo_type='model', folder_path=str(output), path_in_repo=SUBFOLDER,
                      parent_commit=info.sha, commit_message='Add private standalone merged model')
    verified = api.repo_info(repo_id=REPO, repo_type='model')
    remote = set(api.list_repo_files(repo_id=REPO, repo_type='model', revision=verified.sha))
    expected = {f'{SUBFOLDER}/{p.relative_to(output).as_posix()}' for p in output.rglob('*') if p.is_file()}
    if not verified.private or not expected.issubset(remote):
        raise RuntimeError('Upload was made but privacy or complete file inventory could not be verified')
    print('::notice title=Private standalone model::Merged weights verified and uploaded privately; source adapter retained.', flush=True)
    shutil.rmtree(root)


if __name__ == '__main__':
    main()
