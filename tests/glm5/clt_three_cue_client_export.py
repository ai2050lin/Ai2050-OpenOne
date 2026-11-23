"""Build the offline client asset from audited public confirmation data only."""
import json
from pathlib import Path
import sys

import numpy as np
import clt_b26_three_cues as B
from deploy.storage_policy import check_write
from source_binding_design import MODELS, read, sha

DEST = B.ROOT/'frontend/src/components/v8/threeCueEvidence.json'


def summary(folder, receipt, title, tokenizer):
    record=read(folder/receipt)
    files=record['files']
    names=('evaluation.json','confirm-collection.json','confirm-materials.json')
    for name in names:
        assert sha(folder/name)==files[name]
    evaluation=read(folder/names[0]); collection=read(folder/names[1]); rows=read(folder/names[2]); outcome=[]; masses=[]
    assert collection['passed'] and evaluation['collection_sha']==files['confirm-collection.json']
    for name, digest in sorted(collection['shards'].items()):
        meta=name.replace('.npz','.json')
        assert digest==files[name]==sha(folder/name) and sha(folder/meta)==files[meta]
        block=read(folder/meta);assert digest==block['sha']
        outcome.extend(block['scores'])
        with np.load(folder/name) as npz:masses.extend(npz['mass_actual'])
    assert len(outcome)==len(masses)==len(rows) and len(rows)%12==0
    assert evaluation['provenance']['model_revision']==record['provenance']['model_revision']
    family_data=[]; tokens={token:tokenizer.decode(token,clean_up_tokenization_spaces=False) for token in {x['top1'] for x in outcome}}
    for i in range(0,len(rows),12):
        family=rows[i:i+12];root=family[0]; cases=[]
        for r,s,m in zip(family,outcome[i:i+12],masses[i:i+12],strict=True):
            assert r['id']==s['id'] and r['family']==root['family'] and r['names']==root['names'] and r['codes']==root['codes']
            assert s['top1_correct']==(s['top1']==r['gold_id']) and len(set(r['cue_targets']))==3
            at=r['prompt'].find('Question:');end=r['prompt'].find('<|im_end|>',at)
            assert 0<=at<end
            cases.append(dict(id=r['id'],rule=r['rule'],labelSwap=bool(r['label_swap']),positionSwap=bool(r['position_swap']),order=r['physical_order'],
                              labels=r['printed'],targets=r['cue_targets'],gold=r['target_record'],mass=[round(float(v),8) for v in m],
                              output=tokens[s['top1']],tokenId=s['top1'],correct=s['top1_correct'],inputTokens=len(r['tokens']),question=r['prompt'][at+10:end].strip()))
        family_data.append(dict(id=root['family'],archive=root['title'],names=root['names'],codes=root['codes'],cases=cases))
    causal=None
    if 'intervention-causal.json' in files:
        assert sha(folder/'intervention-causal.json')==files['intervention-causal.json']
        assert record['causal_audit'] is True
        result=read(folder/'intervention-causal.json')
        causal=dict(n=result['n'],branches=result['branches'])
    calibration_failed='calibration-causal.json' in files and causal is None
    scalar_refs=record['native_audit'].get('scalar_examples',{})
    return dict(id=folder.name,title=title,snapshot=evaluation['provenance']['model_revision'],precision='BF16',model='Qwen3-14B',
                readout=dict(layer=26,head=36,position='最后输入 token'),
                scope='边界 BEGIN 下方的第一条记录；姓名、印刷标签和物理首行始终分别指向三条记录。',
                totals=collection['totals'],probe=evaluation['probe_accuracy'],natural=evaluation['diagnostic'],
                predictor={k:dict(**evaluation['prediction'][f'{k}_selected_512'],passed=evaluation['gates'][k]['passed']) for k in ('q','k')},
                swaps=record['native_audit']['swaps_by_rule'],coordinates=record['native_audit']['coordinates'],scalarRefs=scalar_refs,
                admission=evaluation['admitted'],causal=causal,calibrationFailed=calibration_failed,
                prior={k:evaluation['prediction'][f'{k}_prior_512']['five_way_agreement'] for k in ('q','k')} if 'q_prior_512' in evaluation['prediction'] else None,
                provenance={name:files[name] for name in (*names,'candidate.json')},families=family_data)


def main():
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODELS['qwen14'],local_files_only=True)
    base=B.HERE/'result'
    runs=[summary(base/'clt_b26_three_cues_v3','delivery.json','三线索独立交换',tokenizer)]
    if '--include-competition' in sys.argv:
        runs.append(summary(base/'clt_b26_competition_v1','verified.json','记录竞争导向 Q/K',tokenizer))
    payload=json.dumps({'schema':'clt_three_cue_client.v1','runs':runs},ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
    destination=check_write(DEST,len(payload)*2,operation='Verified client evidence')
    temp=destination.with_suffix('.json.partial')
    assert not temp.exists()
    temp.write_bytes(payload);temp.replace(destination)
    print(f'Exported {len(runs)} audited run(s); {sum(len(r["families"]) for r in runs)} families; {len(payload)} bytes')


if __name__=='__main__':main()
