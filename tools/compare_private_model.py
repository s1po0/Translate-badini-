"""Compare the private standalone model to its pinned starting base.

Uses the same source-grouped reference samples and decoding for both models.
Samples are held out of the seed reference split, but may have appeared in the
base pretraining or synthetic teacher data; this is not an independent test.
Results contain aggregate numbers only and are encrypted by the workflow.
"""
import gc
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import LANGUAGES, read_pairs, split_pairs
from tools.filter_dictionary_agreement import normalize


def sample_rows(rows, count=100, seed=619):
    grouped = {lang: {} for lang in ('en','ar','sorani')}
    for lang, source, target in rows:
        if lang not in grouped or not normalize(target):
            continue
        if lang == 'en' and not re.fullmatch('[A-Za-z]+', source):
            continue
        if lang == 'ar' and (len(source.split()) != 1 or not any('\u0600' <= c <= '\u06ff' and c.isalpha() for c in source)):
            continue
        key = source.casefold()
        if key not in grouped[lang]:
            grouped[lang][key] = [source, set()]
        grouped[lang][key][1].add(normalize(target))
    result = {}
    for lang in ('en','ar','sorani'):
        keys = sorted(grouped[lang]); random.Random(seed).shuffle(keys)
        rows = [grouped[lang][x] for x in keys[:count]]
        if not rows:
            raise ValueError('No held-out seed references for '+lang)
        result[lang] = rows
    return result


def evaluate(model_path, tokenizer_path, rows, revision=None):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path, revision=revision)
    model=AutoModelForSeq2SeqLM.from_pretrained(model_path, revision=revision, torch_dtype=torch.float32,
                                                low_cpu_mem_usage=True).eval()
    results={}
    with torch.inference_mode():
        for lang, cases in rows.items():
            tokenizer.src_lang = LANGUAGES[lang]
            tokenizer.tgt_lang = LANGUAGES['badini']
            predictions=[]
            for start in range(0, len(cases), 8):
                batch=cases[start:start+8]
                encoded=tokenizer([x[0] for x in batch], return_tensors='pt', padding=True,
                                  truncation=True, max_length=128)
                output=model.generate(**encoded, forced_bos_token_id=tokenizer.convert_tokens_to_ids('ckb_Arab'),
                                      num_beams=1, max_new_tokens=48, do_sample=False)
                predictions.extend(normalize(x) for x in tokenizer.batch_decode(output, skip_special_tokens=True))
            results[lang]=predictions
    del model, tokenizer
    gc.collect()
    return results


def score(rows, before, after):
    from sacrebleu.metrics import CHRF
    chrf=CHRF(word_order=2)
    result={}
    for lang, cases in rows.items():
        a,b=before[lang],after[lang]
        if len(a)!=len(cases) or len(b)!=len(cases):
            raise ValueError('Prediction count mismatch')
        exact_a=[p in refs for p,(_,refs) in zip(a,cases)]
        exact_b=[p in refs for p,(_,refs) in zip(b,cases)]
        def avg_chrf(predictions):
            return round(sum(max(chrf.sentence_score(p,[r]).score for r in refs)
                             for p,(_,refs) in zip(predictions,cases))/len(cases),2)
        # n, base exact, private exact, base chrF++, private chrF++,
        # private-only exact, base-only exact.
        result[lang]=[len(cases),sum(exact_a),sum(exact_b),avg_chrf(a),avg_chrf(b),
                      sum(y and not x for x,y in zip(exact_a,exact_b)),
                      sum(x and not y for x,y in zip(exact_a,exact_b))]
    return result


def main():
    import os
    from huggingface_hub import HfApi, snapshot_download
    from tools.hf_training_round import REPO
    from tools.merge_private_model import SUBFOLDER
    token=os.environ.get('HF_TOKEN')
    if not token:raise ValueError('HF_TOKEN missing')
    api=HfApi(token=token)
    info=api.repo_info(repo_id=REPO,repo_type='model')
    if not info.private:raise ValueError('Model must be private')
    folder=Path('private-eval-snapshot')
    snapshot_download(REPO,revision=info.sha,token=token,local_dir=str(folder),allow_patterns=[SUBFOLDER+'/*'])
    merged=folder/SUBFOLDER
    provenance=json.loads((merged/'merge-provenance.json').read_text())
    base,revision=provenance['base_model'],provenance['base_revision']
    from train import BASE_MODEL
    if base!=BASE_MODEL or not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('Invalid base provenance')
    _,heldout=split_pairs(read_pairs('training_data/ferman_approximate_sorani_badini.csv'))
    rows=sample_rows(heldout)
    a=evaluate(base,base,rows,revision=revision)
    b=evaluate(str(merged),str(merged),rows)
    Path('private-scores.json').write_text(json.dumps(score(rows,a,b),separators=(',',':')))

if __name__=='__main__':main()
