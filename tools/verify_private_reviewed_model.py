"""Retest reviewed examples on the published private merged model."""
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from huggingface_hub import HfApi,snapshot_download,hf_hub_download
    from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
    from tools.hf_training_round import REPO
    from tools.merge_private_model import SUBFOLDER
    from tools.private_review_pipeline import user_match
    from train_adapter import read_reviewed_examples
    token=os.environ['HF_TOKEN'];api=HfApi(token=token)
    info=api.repo_info(repo_id=REPO,repo_type='model')
    if not info.private:raise ValueError('Model must remain private')
    root=Path('private-confirm-snapshot')
    snapshot_download(REPO,revision=info.sha,token=token,local_dir=str(root),allow_patterns=[SUBFOLDER+'/*'])
    reviewed=hf_hub_download(REPO,filename='private-training/reviewed/41354201548.json',revision=info.sha,token=token)
    pairs=read_reviewed_examples(reviewed); model_path=str(root/SUBFOLDER)
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(model_path,src_lang='eng_Latn')
    model=AutoModelForSeq2SeqLM.from_pretrained(model_path,torch_dtype=torch.float32,low_cpu_mem_usage=True).eval()
    inputs=tok([p[1] for p in pairs],return_tensors='pt',padding=True,truncation=True,max_length=64)
    with torch.inference_mode():
        out=model.generate(**inputs,forced_bos_token_id=tok.convert_tokens_to_ids('ckb_Arab'),num_beams=1,max_new_tokens=32,do_sample=False)
    preds=tok.batch_decode(out,skip_special_tokens=True)
    if len(preds)!=len(pairs):raise ValueError('Incorrect output count')
    result=[{'source':r[1],'target':r[2],'output':p,'matches':user_match(p,r[2])} for r,p in zip(pairs,preds)]
    Path('verification.json').write_text(json.dumps({'private':True,'revision':info.sha,'results':result},ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':main()
