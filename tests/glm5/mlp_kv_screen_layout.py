"""Recompute any screening mismatch in its original four-sample batch layout."""
from collections import defaultdict

import numpy as np
import torch

import mlp_kv_common as C
from mlp_kv_mediation import mediate, summarize
from mlp_functional_trace import load_model
from shared.dist_protocol.mechanism_trace import GpuLease


def main():
    root=C.OUT/'targeted';dest=root/'result-layout-v2.json'
    if dest.exists():return
    screen=C.read(root/'screen.json');inputs=C.read(root/'inputs.json');initial=C.read(root/'result.json')
    assert screen['inputs_sha256']==C.file_sha(root/'inputs.json')
    enrolled=screen['enrolled'];rows=initial['rows'];assert len(rows)==len(enrolled)
    def match(i,row):
        a=screen['rows'][i]['clean'];b=row['conditions']['clean']
        return (a['word'],a['gap'])==(b['word'],b['gap'])
    failed=defaultdict(list)
    for j,(i,row) in enumerate(zip(enrolled,rows)):
        if not match(i,row):failed[i//4*4].append((j,i))
    contract={**C.identity(),'schema':'mlp_kv_layout_correction_prereg.v2','source_sha256':C.file_sha(__file__),
              'initial_result_sha256':C.file_sha(root/'result.json'),'screen_sha256':C.file_sha(root/'screen.json'),
              'batches':{str(k):v for k,v in failed.items()},'rule':'Use original screening order and shape to recompute all eight operations for every discrepant case. Other batch members are numerical padding only, excluded from enrolled statistics. Preserve all original results. Identical word and gap required for all enrolled cases.'}
    C.save(root/'layout-correction-prereg.json',contract)
    if not failed:
        C.save(dest,{**initial,'layout_correction':contract});return
    original=root/initial['native']['path'];assert C.file_sha(original)==initial['native']['sha256']
    with np.load(original) as raw:arrays={k:raw[k].copy() for k in initial['native']['fields']}
    corrections=[];scopes=list(initial['scopes']);torch.set_num_threads(4)
    prereg={'power_plan':{'all_cases':C.read(root/'prereg.json')['power_plan']}}
    with GpuLease(),torch.inference_mode():
        model,tok,eos=load_model()
        for start,failures in failed.items():
            batch=inputs['rows'][start:start+4];assert len(batch)==4
            scratch=root/f'layout-batch-{start}';scratch.mkdir(exist_ok=True);C.OUT=scratch
            replay=mediate(model,tok,eos,batch,{'passed':True},prereg)
            scopes.extend(replay['scopes'])
            with np.load(scratch/replay['native']['path']) as trace:
                for j,i in failures:
                    offset=i-start;new=replay['rows'][offset]
                    assert new['id']==rows[j]['id']
                    corrections.append({'enrolled_index':j,'screen_index':i,'before':rows[j]['conditions']['clean'],
                                        'after':new['conditions']['clean'],'screen':screen['rows'][i]['clean'],
                                        'batch_case_ids':[r['id'] for r in batch],'retained_only':new['id']})
                    rows[j]={**new,'index':j}
                    for k in arrays:arrays[k][j]=trace[k][offset]
            C.save(scratch/'batch-result.json',replay)
    matches=sum(match(i,row) for i,row in zip(enrolled,rows));power=C.read(root/'prereg.json')['power_plan']['n_required']
    summary=summarize(rows,matches==len(rows),power)
    checks={k:sum(r[k] for r in rows) for k in ['replay_logits_bitwise_equal','rescue_logits_bitwise_equal','replay_generation_equal','rescue_generation_equal']}
    checks.update(only_B35_and_source=all(s['only_allowed_layers'] and s['all_earlier_positions_bitwise_equal'] for s in scopes),
                  nonzero_KV_cases=sum(any(x>0 for a in r['delta_KV_norm_by_GQA'].values() for x in a) for r in rows),
                  nonzero_future_distribution_cases=sum(r['conditions']['head_full']['KL']>0 for r in rows))
    native=root/'native-slots-layout-v2.npz';np.savez(native,**arrays)
    C.save(dest,{**initial,'rows':rows,'summary':summary,'checks':checks,'scopes':scopes,'screen_to_replay_exact':matches,
                 'native':{**initial['native'],'path':native.name,'sha256':C.file_sha(native)},'layout_correction':contract,'corrections':corrections})
    print({'screen_exact':matches,'n':len(rows),'corrections':corrections,'checks':checks},flush=True)


if __name__=='__main__':main()
