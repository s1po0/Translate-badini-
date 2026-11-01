"""Generate private-model translations for a small, public English test set."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

SAMPLES = ['water', 'bread', 'mother', 'father', 'house', 'school', 'book',
           'friend', 'sun', 'apple', 'How are you?', 'I am going home.',
           'Where is the school?', 'Thank you very much.']


def main():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from huggingface_hub import HfApi, snapshot_download
    from tools.hf_training_round import REPO
    from tools.merge_private_model import SUBFOLDER
    token=os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError('HF_TOKEN missing')
    api=HfApi(token=token)
    info=api.repo_info(repo_id=REPO,repo_type='model')
    if not info.private:
        raise ValueError('Model repository must be private')
    folder=Path('private-test-snapshot')
    snapshot_download(REPO,revision=info.sha,token=token,local_dir=str(folder),
                      allow_patterns=[SUBFOLDER+'/*'])
    model_path=str(folder/SUBFOLDER)
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(model_path,src_lang='eng_Latn')
    model=AutoModelForSeq2SeqLM.from_pretrained(model_path,torch_dtype=torch.float32,
                                                low_cpu_mem_usage=True).eval()
    translations=[]
    for start in range(0,len(SAMPLES),8):
        inputs=tokenizer(SAMPLES[start:start+8],return_tensors='pt',padding=True,truncation=True,max_length=128)
        with torch.inference_mode():
            output=model.generate(**inputs,forced_bos_token_id=tokenizer.convert_tokens_to_ids('ckb_Arab'),
                                  num_beams=1,max_new_tokens=48,do_sample=False)
        translations.extend(tokenizer.batch_decode(output,skip_special_tokens=True))
    if len(translations)!=len(SAMPLES):
        raise ValueError('Wrong number of results')
    Path('private-translations.json').write_text(json.dumps(list(zip(SAMPLES,translations)),ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':
    main()
