"""Teach user-reviewed translations to a private model with guarded promotion.

Corrections and model weights stay private. Public workflow artifacts contain
only authenticated ciphertext. Three independent candidates are trained from one
pinned baseline; at most one is promoted together with its merged full model.
"""
import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train import BASE_MODEL
from continuous_training import accepted
from tools.hf_training_round import REPO, FILES, check_adapter, update_card
from tools.merge_private_model import SUBFOLDER, checks, compare_logits
from tools.private_review_channel import derive
from train_adapter import read_reviewed_examples
from tools.filter_dictionary_agreement import normalize

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def api_and_token():
    from huggingface_hub import HfApi
    token=os.environ.get('HF_TOKEN')
    if not token:raise ValueError('Missing token')
    return HfApi(token=token),token


def private_info(api):
    info=api.repo_info(repo_id=REPO,repo_type='model')
    if not info.private or not re.fullmatch('[0-9a-f]{40}',info.sha or ''):
        raise ValueError('Repository is not private or has no stable revision')
    return info


def prepare(backup,run_id):
    from huggingface_hub import CommitOperationAdd,hf_hub_download
    if not run_id.isdigit():raise ValueError('Invalid run ID')
    reviewed=read_reviewed_examples(backup)
    if len(reviewed)!=6:raise ValueError('Expected six reviewed examples')
    api,token=api_and_token(); info=private_info(api)
    p=hf_hub_download(REPO,filename=f'{SUBFOLDER}/merge-provenance.json',revision=info.sha,token=token)
    prov=json.loads(Path(p).read_text())
    base_info=api.repo_info(repo_id=BASE_MODEL,repo_type='model')
    if prov.get('base_model')!=BASE_MODEL or prov.get('base_revision')!=base_info.sha:
        raise ValueError('Base model changed; refusing to train against a different foundation')
    current=hf_hub_download(REPO,filename='adapter_model.safetensors',revision=info.sha,token=token)
    if 'adapter_model_sha256' in prov:
        if digest(current)!=prov['adapter_model_sha256']:raise ValueError('Adapter and full model are not synchronized')
    elif 'adapter_revision' in prov:
        original=hf_hub_download(REPO,filename='adapter_model.safetensors',revision=prov['adapter_revision'],token=token)
        if digest(current)!=digest(original):raise ValueError('Adapter and full model are not synchronized')
    else:raise ValueError('Merged model lacks adapter provenance')
    dest=f'private-training/reviewed/{run_id}.json'
    if dest in api.list_repo_files(REPO,revision=info.sha):raise ValueError('Reviewed examples already saved')
    api.create_commit(repo_id=REPO,repo_type='model',operations=[CommitOperationAdd(path_in_repo=dest,path_or_fileobj=str(backup))],
                      parent_commit=info.sha,commit_message='Save private reviewed translation examples')
    latest=private_info(api)
    if dest not in api.list_repo_files(REPO,revision=latest.sha):raise RuntimeError('Private examples were not saved')
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'],'a') as f:
            f.write('revision='+latest.sha+'\nbase_revision='+base_info.sha+'\n')
    print('Private reviewed examples saved and baseline pinned.',flush=True)


def predict(adapter,reviewed):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(str(adapter),src_lang='eng_Latn')
    base=AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL,low_cpu_mem_usage=True)
    model=PeftModel.from_pretrained(base,str(adapter)).eval()
    src=[x[1] for x in reviewed]
    batch=tok(src,return_tensors='pt',padding=True,truncation=True,max_length=64)
    with torch.inference_mode():
        out=model.generate(**batch,forced_bos_token_id=tok.convert_tokens_to_ids('ckb_Arab'),num_beams=1,max_new_tokens=32,do_sample=False)
    values=tok.batch_decode(out,skip_special_tokens=True)
    del model,base,tok; gc.collect()
    return values


def user_match(pred,target):
    # Invisible joiners and spacing differ in user-provided orthography; both
    # forms remain present in the encrypted report for human confirmation.
    def comparable(value):
        return normalize(value).replace(' ', '').strip('?.!؟،؛:')
    return comparable(pred)==comparable(target)


def gate(report,word,reviewed,before,after):
    if len(reviewed)!=len(before) or len(before)!=len(after):raise ValueError('Reviewed result mismatch')
    b=[user_match(p,row[2]) for p,row in zip(before,reviewed)]
    a=[user_match(p,row[2]) for p,row in zip(after,reviewed)]
    loss=accepted(report,.03)
    word_pass=word['after_exact']>=word['before_exact']
    correction_pass=sum(a)>sum(b) and all(not x or y for x,y in zip(b,a))
    return {'candidate_accepted':loss and word_pass and correction_pass,'loss_guard_pass':loss,'word_guard_pass':word_pass,
            'reviewed_guard_pass':correction_pass,'before_review_exact':sum(b),'after_review_exact':sum(a),
            'review_count':len(a),'before_review_matches':b,'after_review_matches':a,
            'heldout_loss_before':report['teacher_forced_loss_before']['overall'],
            'heldout_loss_after':report['teacher_forced_loss_after']['overall'],
            'word_exact_before':word['before_exact'],'word_exact_after':word['after_exact'],'word_examples':word['examples'],
            'optimizer_steps':report['optimizer_steps'],'teacher_pairs_in_training_pool':len(a),
            'dataset_description':'private user-reviewed examples with replay',
            'limitation':'Six training examples and small reference tests do not establish overall translation accuracy.'}


def seal_folder(folder,output,token):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    with tempfile.TemporaryDirectory() as tmp:
        archive=Path(tmp)/'private.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for path in Path(folder).rglob('*'):
                if path.is_file():z.write(path,path.relative_to(folder).as_posix())
        salt,nonce=os.urandom(32),os.urandom(12)
        encrypted=AESGCM(derive(token,salt)).encrypt(nonce,archive.read_bytes(),None)
        Path(output).write_bytes(salt+nonce+encrypted)


def open_folder(path,output,token):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    blob=Path(path).read_bytes()
    if len(blob)<61:raise ValueError('Invalid encrypted candidate')
    data=AESGCM(derive(token,blob[:32])).decrypt(blob[32:44],blob[44:],None)
    import io
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            p=Path(info.filename)
            if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe archive path')
        z.extractall(root)


def train_candidate(revision,base_revision,feedback_run,repeats,replay,lr,output):
    from huggingface_hub import snapshot_download,hf_hub_download
    from tools.train_reviewed import training_steps
    if not re.fullmatch('[0-9a-f]{40}',revision) or not re.fullmatch('[0-9a-f]{40}',base_revision) or not feedback_run.isdigit():raise ValueError('Invalid pinned inputs')
    api,token=api_and_token();private_info(api)
    if api.repo_info(repo_id=BASE_MODEL,repo_type='model').sha!=base_revision:raise ValueError('Base changed')
    root=Path('private-train-work');root.mkdir(exist_ok=False)
    baseline=root/'baseline'
    snapshot_download(REPO,revision=revision,token=token,local_dir=str(baseline),allow_patterns=list(FILES))
    check_adapter(baseline)
    feedback=hf_hub_download(REPO,filename=f'private-training/reviewed/{feedback_run}.json',revision=revision,token=token)
    reviewed=read_reviewed_examples(feedback)
    if len(reviewed)!=6:raise ValueError('Wrong reviewed example count')
    steps=training_steps(feedback,replay,repeats)
    cmd=[sys.executable,str(ROOT/'train_adapter.py'),'--initial-adapter',str(baseline),'--reviewed-data',feedback,
         '--reviewed-repeats',str(repeats),'--steps',str(steps),'--per-language',str(replay),
         '--eval-per-language','20','--learning-rate',str(lr),'--seed','42','--output',str(root/'trained')]
    subprocess.run(cmd,check=True,cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    candidate=root/'trained'/'adapter';check_adapter(candidate)
    report=json.loads((root/'trained'/'report.json').read_text())
    if report['reviewed_unique_rows']!=len(reviewed) or report['optimizer_steps']!=report['training_pool_pairs']:
        raise ValueError('Did not cover all reviewed examples in the attempted pool')
    wf=root/'words.json'
    cmd=[sys.executable,str(ROOT/'tools/evaluate_word_quality.py'),'--before',str(baseline),'--after',str(candidate),
         '--size','100','--exclude-reviewed',feedback,'--output',str(wf)]
    subprocess.run(cmd,check=True,cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    word=json.loads(wf.read_text());before=predict(baseline,reviewed);after=predict(candidate,reviewed)
    decision=gate(report,word,reviewed,before,after)
    bundle=root/'bundle';bundle.mkdir()
    shutil.copytree(candidate,bundle/'adapter')
    (bundle/'decision.json').write_text(json.dumps(decision,ensure_ascii=False))
    # This result, including predictions, is included only inside ciphertext.
    (bundle/'examples.json').write_text(json.dumps([{'source':r[1],'target':r[2],'before':a,'after':b} for r,a,b in zip(reviewed,before,after)],ensure_ascii=False))
    (bundle/'metadata.json').write_text(json.dumps({'baseline_revision':revision,'base_revision':base_revision,'feedback_run':feedback_run}))
    if api.repo_info(repo_id=BASE_MODEL,repo_type='model').sha!=base_revision:raise ValueError('Base changed during training')
    seal_folder(bundle,output,token)
    shutil.rmtree(root)
    print('Private candidate trained and encrypted.',flush=True)


def finalize(revision,base_revision,feedback_run,artifacts,output):
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
    if not re.fullmatch('[0-9a-f]{40}',revision) or not re.fullmatch('[0-9a-f]{40}',base_revision) or not feedback_run.isdigit():raise ValueError('Invalid pinned inputs')
    api,token=api_and_token(); info=private_info(api)
    if info.sha!=revision or api.repo_info(repo_id=BASE_MODEL,repo_type='model').sha!=base_revision:
        raise ValueError('Model changed during training; no weights published')
    root=Path('private-final-work');root.mkdir(exist_ok=False)
    files=sorted(Path(artifacts).rglob('candidate.enc'))
    if len(files)!=3:raise ValueError('Expected exactly three encrypted candidates')
    candidates=[]
    for index,f in enumerate(files):
        dest=root/f'candidate-{index}';open_folder(f,dest,token)
        meta=json.loads((dest/'metadata.json').read_text());decision=json.loads((dest/'decision.json').read_text())
        if meta!={'baseline_revision':revision,'base_revision':base_revision,'feedback_run':feedback_run}:raise ValueError('Candidate baseline mismatch')
        check_adapter(dest/'adapter');candidates.append((decision,dest))
    valid=[x for x in candidates if x[0]['candidate_accepted']]
    # Encrypting the report happens outside this function. Do not print examples.
    result={'candidate_count':len(candidates),'accepted_count':len(valid),
            'candidate_results':[x[0] for x in candidates], 'published':False}
    if not valid:
        Path(output).write_text(json.dumps(result,ensure_ascii=False));return
    decision,winner=max(valid,key=lambda x:(x[0]['after_review_exact'],-x[0]['heldout_loss_after']))
    result['selected']=decision; result['examples']=json.loads((winner/'examples.json').read_text())
    baseline=root/'baseline'
    snapshot_download(REPO,revision=revision,token=token,local_dir=str(baseline),allow_patterns=list(FILES)+['README.md'])
    check_adapter(baseline);adapter=winner/'adapter'
    if digest(adapter/'adapter_model.safetensors')==digest(baseline/'adapter_model.safetensors'):
        raise ValueError('Candidate adapter has unchanged weights')
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(str(adapter))
    base=AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL,revision=base_revision,torch_dtype=torch.float32,low_cpu_mem_usage=True)
    model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False).eval()
    before=checks(model,tokenizer,torch)
    model=model.merge_and_unload(safe_merge=True).eval()
    compare_logits(torch,before,checks(model,tokenizer,torch))
    upload=root/'upload';full=upload/SUBFOLDER
    model.save_pretrained(str(full),safe_serialization=True,max_shard_size='2GB');tokenizer.save_pretrained(str(full))
    del model,base;gc.collect()
    loaded=AutoModelForSeq2SeqLM.from_pretrained(str(full),torch_dtype=torch.float32,low_cpu_mem_usage=True).eval()
    compare_logits(torch,before,checks(loaded,tokenizer,torch));del loaded,before;gc.collect()
    for name in FILES:shutil.copy2(adapter/name,upload/name)
    (upload/'README.md').write_text(update_card((baseline/'README.md').read_text(),decision,feedback_run),encoding='utf-8')
    (full/'README.md').write_text('# Private experimental full model\n\n'
        f'The user adapter was merged into [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) at revision `{base_revision}`. '
        'This is a derivative, not a model trained from scratch. The adapter includes private user-reviewed corrections. '
        'The base derives from Meta NLLB-200 distilled 600M (CC BY-NC 4.0). '
        'The derivative base model card states MIT; review all upstream rights and attribution requirements before redistribution. '
        'Merge fidelity was checked; translation accuracy requires independent evaluation.\n',encoding='utf-8')
    prov={'base_model':BASE_MODEL,'base_revision':base_revision,'adapter_parent_revision':revision,
          'adapter_model_sha256':digest(adapter/'adapter_model.safetensors'),
          'training_run':feedback_run,'verification':'FP32 logits before and after merge and reload'}
    (full/'merge-provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
    history=upload/'training-history';history.mkdir()
    (history/f'private-{feedback_run}.json').write_text(json.dumps(decision,indent=2)+'\n')
    old=set(api.list_repo_files(REPO,revision=revision));new={p.relative_to(upload).as_posix() for p in upload.rglob('*') if p.is_file()}
    old_merged={p for p in old if p.startswith(SUBFOLDER+'/')}
    if not old_merged.issubset(new):raise ValueError('Refusing to remove existing merged-model files')
    if f'training-history/private-{feedback_run}.json' in old:raise ValueError('Training history already exists')
    if private_info(api).sha!=revision:raise ValueError('Model changed before publication')
    api.upload_folder(repo_id=REPO,repo_type='model',folder_path=str(upload),parent_commit=revision,
                      commit_message='Apply guarded private corrections to adapter and full model')
    final=private_info(api)
    from huggingface_hub import hf_hub_download
    check=hf_hub_download(REPO,filename='adapter_model.safetensors',revision=final.sha,token=token,force_download=True)
    if digest(check)!=digest(adapter/'adapter_model.safetensors'):
        raise RuntimeError('Published adapter hash did not match')
    final_files=set(api.list_repo_files(REPO,revision=final.sha))
    if not new.issubset(final_files):raise RuntimeError('Published file inventory incomplete')
    result['published']=True;result['new_revision']=final.sha
    Path(output).write_text(json.dumps(result,ensure_ascii=False))
    shutil.rmtree(root)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    a=s.add_parser('prepare');a.add_argument('--backup',required=True);a.add_argument('--run-id',required=True)
    a=s.add_parser('train');a.add_argument('--revision',required=True);a.add_argument('--base-revision',required=True);a.add_argument('--feedback-run',required=True);a.add_argument('--repeats',type=int,required=True);a.add_argument('--replay',type=int,required=True);a.add_argument('--lr',type=float,required=True);a.add_argument('--output',required=True)
    a=s.add_parser('finalize');a.add_argument('--revision',required=True);a.add_argument('--base-revision',required=True);a.add_argument('--feedback-run',required=True);a.add_argument('--artifacts',required=True);a.add_argument('--output',required=True)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.backup,a.run_id)
    elif a.action=='train':train_candidate(a.revision,a.base_revision,a.feedback_run,a.repeats,a.replay,a.lr,a.output)
    else:finalize(a.revision,a.base_revision,a.feedback_run,a.artifacts,a.output)

if __name__=='__main__':main()
