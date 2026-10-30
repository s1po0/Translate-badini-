"""More detailed paired private-model evaluation, with contamination checks.

This checks accessible historical synthetic sources, but the original base's
training data are unknown. The Sorani examples have machine-generated reference
translations and were used by earlier evaluations. Scores are text similarity,
not a replacement for an independent native-speaker assessment.
"""
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import LANGUAGES, read_pairs, split_pairs, BASE_MODEL
from tools.compare_private_model import sample_rows
from tools.evaluate_word_quality import heldout_words
from tools.filter_dictionary_agreement import normalize


def synthetic_sources(root):
    root=Path(root)
    synthetic_ids=set()
    with (root/'provenance.csv').open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            if row['dataset'] != 'reference':
                synthetic_ids.add(row['id'])
    sources=defaultdict(set); badini=set(); seen=set()
    with (root/'synthetic.csv').open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            if row['id'] in synthetic_ids:
                sources[row['source_lang']].add(normalize(row['source_text']))
                badini.add(normalize(row['badini']));seen.add(row['id'])
    if seen!=synthetic_ids:
        raise ValueError('Provenance contains missing synthetic IDs')
    return sources, badini


def collect(rows, training, exclusions, synthetic_targets, count=300):
    previous = sample_rows(rows)
    old_words = {normalize(x[0]) for x in previous['en']}
    old_words |= {normalize(x[0]) for x in heldout_words(rows, 100)}
    old_ar = {normalize(x[0]) for x in previous['ar']}
    old={'en':old_words, 'ar':old_ar}
    groups={x:{} for x in ('en','ar')}
    for lang,source,target in rows:
        if lang not in groups:continue
        key=normalize(source)
        if key in old[lang] or key in exclusions.get(lang,set()):continue
        if lang=='en' and not re.fullmatch('[A-Za-z]+',source):continue
        if lang=='ar' and (len(source.split())!=1 or not any('\u0600'<=c<='\u06ff' and c.isalpha() for c in source)):continue
        if not normalize(target):continue
        groups[lang].setdefault(key,[source,set()])[1].add(normalize(target))
    output={}
    for lang in ('en','ar'):
        keys=sorted(groups[lang]);random.Random(421).shuffle(keys)
        output[lang]=[groups[lang][x] for x in keys[:count]]
        if not output[lang]:raise ValueError('No eligible held-out examples for '+lang)
    # Use new held-out English entries only; deduplicate the Badini source, and
    # remove Badini texts appearing in reference training or historical synthetic output.
    target_in_train={normalize(t) for _,_,t in training} | synthetic_targets
    inverse={}
    for source,refs in output['en']:
        for target in refs:
            if target not in target_in_train:
                inverse.setdefault(target,[target,set()])[1].add(normalize(source))
    keys=sorted(inverse);random.Random(421).shuffle(keys)
    output['reverse']=[inverse[x] for x in keys[:count]]
    # Reused small, machine-generated benchmark: exploratory, not independent.
    output['sorani']=previous['sorani']
    return output


def evaluate(path, tokenizer_path, datasets, revision=None):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,revision=revision)
    model=AutoModelForSeq2SeqLM.from_pretrained(path,revision=revision,torch_dtype=torch.float32,low_cpu_mem_usage=True).eval()
    predictions={}
    with torch.inference_mode():
        for lang, rows in datasets.items():
            if not rows:
                predictions[lang]=[];continue
            tokenizer.src_lang=LANGUAGES['badini'] if lang=='reverse' else LANGUAGES[lang]
            tokenizer.tgt_lang=LANGUAGES['en'] if lang=='reverse' else LANGUAGES['badini']
            parts=[]
            for start in range(0,len(rows),8):
                encoded=tokenizer([x[0] for x in rows[start:start+8]],return_tensors='pt',padding=True,truncation=True,max_length=128)
                out=model.generate(**encoded, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tokenizer.tgt_lang),
                                   num_beams=1,max_new_tokens=48,do_sample=False)
                parts.extend(tokenizer.batch_decode(out,skip_special_tokens=True))
            predictions[lang]=parts
    del model,tokenizer
    import gc;gc.collect()
    return predictions


def aggregate(rows, a, b):
    from sacrebleu.metrics import CHRF
    chrf=CHRF(word_order=2)
    result={}
    for lang,cases in rows.items():
        if len(cases)!=len(a[lang]) or len(cases)!=len(b[lang]):raise ValueError('Mismatched prediction lengths')
        if not cases:
            result[lang]={'n':0};continue
        series=[]
        for (source,refs), old,new in zip(cases,a[lang],b[lang]):
            oldn,newn=normalize(old),normalize(new)
            sa=max(chrf.sentence_score(oldn,[ref]).score for ref in refs)
            sb=max(chrf.sentence_score(newn,[ref]).score for ref in refs)
            series.append((int(oldn in refs),int(newn in refs),sa,sb))
        n=len(series); rng=random.Random(2145)
        bootstrap=[]
        for _ in range(2000):
            ix=[rng.randrange(n) for _ in range(n)]
            bootstrap.append(sum(series[i][3]-series[i][2] for i in ix)/n)
        bootstrap.sort()
        result[lang]={'n':n,'base_exact':sum(x[0] for x in series),'new_exact':sum(x[1] for x in series),
                      'base_chrf':round(sum(x[2] for x in series)/n,2),
                      'new_chrf':round(sum(x[3] for x in series)/n,2),
                      'chrf_delta_interval':[round(bootstrap[49],2),round(bootstrap[1949],2)],
                      'new_only_exact':sum(not x[0] and x[1] for x in series),
                      'base_only_exact':sum(x[0] and not x[1] for x in series)}
        # Public reference sentences, plus private model outputs; kept encrypted.
        chosen=sorted(range(n),key=lambda i: abs(series[i][3]-series[i][2]),reverse=True)[:3]
        result[lang]['examples']=[{'source':cases[i][0],'reference':sorted(cases[i][1]),'base':a[lang][i],'new':b[lang][i]} for i in chosen]
    return result


def main():
    import os
    from huggingface_hub import HfApi,snapshot_download
    from tools.hf_training_round import REPO
    from tools.merge_private_model import SUBFOLDER
    token=os.environ.get('HF_TOKEN')
    if not token:raise ValueError('HF_TOKEN missing')
    api=HfApi(token=token);info=api.repo_info(repo_id=REPO,repo_type='model')
    if not info.private:raise ValueError('Model must be private')
    folder=Path('private-detailed-snapshot')
    snapshot_download(REPO,revision=info.sha,token=token,local_dir=str(folder),allow_patterns=[SUBFOLDER+'/*'])
    merged=folder/SUBFOLDER
    prov=json.loads((merged/'merge-provenance.json').read_text())
    if prov['base_model']!=BASE_MODEL or not re.fullmatch('[0-9a-f]{40}',prov['base_revision']):raise ValueError('Unexpected provenance')
    reference=read_pairs('training_data/ferman_approximate_sorani_badini.csv')
    train,heldout=split_pairs(reference)
    excluded,targets=synthetic_sources('historical')
    rows=collect(heldout,train,excluded,targets)
    a=evaluate(BASE_MODEL,BASE_MODEL,rows,revision=prov['base_revision'])
    b=evaluate(str(merged),str(merged),rows)
    Path('detailed-scores.json').write_text(json.dumps(aggregate(rows,a,b),ensure_ascii=False,separators=(',',':')))

if __name__=='__main__':main()
