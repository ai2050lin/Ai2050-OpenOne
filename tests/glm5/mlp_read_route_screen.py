"""Pre-intervention screening on documents disjoint from route discovery."""
import numpy as np
import torch

import mlp_read_route_common as C
from mlp_read_route_probe import batch_ids, inputs
from mlp_kv_behavior import score, word_indices
from mlp_fidelity_v2_collect_v4 import greedy
from mlp_functional_trace import load_model
from shared.dist_protocol.mechanism_trace import GpuLease
from tools.power_analysis import check_phase


def main():
    dest=C.OUT/'screen.json.gz'
    if dest.exists():return
    rows,prereg=inputs();route=C.read(C.OUT/'route-freeze.json')
    items=[r for r in rows if r['split']!='discovery'];assert len(items)==1472 and len(items)%4==0
    freeze={**C.identity(),'schema':'mlp_early_token_screen_execution.v1','source_sha256':C.file_sha(__file__),
            'route_sha256':C.file_sha(C.OUT/'route-freeze.json'),'inputs_sha256':prereg['inputs_sha256'],
            'n':len(items),'max_each_split':64,'batch':4,'min_margin_exclusive':0,'max_margin_inclusive':8}
    C.save(C.OUT/'screen-execution.json',freeze)
    assert route['discovery_n']==C.N_DISCOVERY
    torch.set_num_threads(4);records=[]
    with GpuLease(),torch.inference_mode():
        model,tok,eos=load_model();words=word_indices(tok)
        for lo in range(0,len(items),4):
            batch=items[lo:lo+4];x=batch_ids(batch)
            prefix=model(input_ids=x[:,:C.CUT],use_cache=True,logits_to_keep=1).past_key_values
            out=model(input_ids=x[:,C.CUT:],past_key_values=prefix,use_cache=True,logits_to_keep=1)
            logits=out.logits[:,-1].float();generated=greedy(model,out.past_key_values,logits,eos,limit=4)
            for j,row in enumerate(batch):
                v=score(row,logits[j],generated[j],tok,words)
                records.append({'index':lo+j,'id':row['id'],'source_id':row['source_id'],'expected':row['expected'],
                                'split':row['split'],'clean':v,'eligible':v['correct'] and 0<v['gap']<=8})
            if (lo+4)%160==0:print('screened',lo+4,'/',len(items),flush=True)
    enrolled=[];summary={}
    for name in ['confirm','lockbox']:
        group=[r for r in records if r['split']==name];eligible=[r for r in group if r['eligible']]
        chosen=sorted(eligible,key=lambda r:C.digest([C.SEED,'enroll',r['id']]))[:64]
        enrolled.extend(r['index'] for r in chosen)
        summary[name]={'screened':len(group),'clean_accuracy':float(np.mean([r['clean']['correct'] for r in group])),
                       'eligible':len(eligible),'enrolled':len(chosen),'median_gap':float(np.median([r['clean']['gap'] for r in group]))}
    if enrolled:power=check_phase(.10,len(enrolled))
    else:power={'pass':False,'n':0,'n_required':prereg['power_plan']['n_required']}
    C.write_gz(dest,{**freeze,'status':'complete','rows':records,'enrolled':enrolled,'summary':summary,'power_plan':power})
    print(summary,'power',power,flush=True)


if __name__=='__main__':main()
