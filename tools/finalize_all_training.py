"""Compare the final accepted shard chain against its original HF checkpoint."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.hf_training_round import decide


def load_one(root, name):
    matches=list(Path(root).rglob(name))
    if len(matches)!=1:raise ValueError(f'Expected one {name} under {root}')
    return json.loads(matches[0].read_text(encoding='utf-8'))


def combined_decision(first, final, words, expected_stages):
    initial=load_one(first,'training-report.json')
    last=load_one(final,'training-report.json')
    progress=load_one(final,'progress.json')
    stages=progress['stages']
    if len(stages)!=expected_stages or [x['stage'] for x in stages]!=list(range(1,expected_stages+1)):
        raise ValueError('Incomplete or misordered stage history')
    if progress['examples_seen_in_attempts'] != progress['planned_examples']:
        raise ValueError('Not all planned training rows were attempted')
    after=last['teacher_forced_loss_after'] if stages[-1]['candidate_accepted'] else last['teacher_forced_loss_before']
    report={'teacher_forced_loss_before':initial['teacher_forced_loss_before'],
            'teacher_forced_loss_after':after,
            'optimizer_steps':sum(s['optimizer_steps'] for s in stages),
            'synthetic_unique_rows':progress['examples_in_accepted_stages']}
    result=decide(report,words)
    result['dataset_description']='accessible reference and unverified synthetic pairs in accepted stages'
    result['attempted_training_pairs']=progress['examples_seen_in_attempts']
    result['accepted_training_pairs']=progress['examples_in_accepted_stages']
    result['accepted_stages']=sum(bool(s['candidate_accepted']) for s in stages)
    result['total_stages']=len(stages)
    if not result['accepted_stages']:
        result['candidate_accepted']=False
        result['limitation'] += ' All stages were rejected; the original checkpoint is unchanged.'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first',required=True);p.add_argument('--final',required=True);p.add_argument('--words',required=True);p.add_argument('--stages',type=int,required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=combined_decision(a.first,a.final,json.loads(Path(a.words).read_text()),a.stages)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print('::notice title=Full-data final decision::'+json.dumps(result),flush=True)
    import os
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'],'a') as f:f.write('accepted='+str(result['candidate_accepted']).lower()+'\n')

if __name__=='__main__':main()
