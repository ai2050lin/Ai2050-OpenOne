"""Check each frozen semantic source separately by native V-only intervention."""
from collections import defaultdict

import numpy as np
import torch

import mlp_read_route_common as C
from mlp_read_route_causal import direct_read, prefix
from mlp_read_route_probe import LAYERS, answer_read, batch_ids, inputs, positions
from mlp_kv_behavior import word_indices
from mlp_functional_trace import load_model
from shared.dist_protocol.mechanism_trace import GpuLease


def main():
    out=C.OUT;dest=out/'by-role-result.json'
    if dest.exists():return
    source,prereg=inputs();pool=[r for r in source if r['split']!='discovery']
    original=C.read_gz(out/'causal-result-strict-v2.json.gz');route=C.read(out/'route-freeze.json')
    rows=original['rows'];idx=[r['screen_index'] for r in rows];lookup={r['screen_index']:r for r in rows}
    fixes={v['screen_index']:v['screen_batch_ids'] for v in original['layout_diagnostics']};by_id={r['id']:i for i,r in enumerate(pool)}
    plan={**C.identity(),'schema':'early_token_read_by_role_prereg.v1','scope':'post-result read-localization audit on the fixed exposed holdouts. No changes to the efficacy test.',
          'source_sha256':C.file_sha(__file__),'previous_sha256':C.file_sha(out/'causal-result-strict-v2.json.gz'),
          'roles':route['roles'],'reader_layers':LAYERS}
    C.save(out/'by-role-prereg.json',plan);torch.set_num_threads(4);found={}
    with GpuLease(),torch.inference_mode():
        model,tok,eos=load_model();words=word_indices(tok)
        def run(batch_indices,active):
            batch=[pool[i] for i in batch_indices];x=batch_ids(batch);pos=positions(batch,route['roles'])
            cache,_=prefix(model,x,pos);logits,base=answer_read(model,x,cache)
            values={role:direct_read(model,x,cache,positions(batch,[role]),base) for role in route['roles']}
            for j,i in enumerate(batch_indices):
                if i not in active:continue
                r=batch[j];gap=float(logits[j,words[r['expected']]].max()-logits[j,words[r['contrast']]].max())
                found[i]={'id':r['id'],'index':lookup[i]['index'],'baseline_gap_equal':gap==lookup[i]['conditions']['clean']['gap'],
                          'roles':{role:{str(b):{k:v[j] for k,v in metrics.items()} for b,metrics in data.items()} for role,data in values.items()}}
        for lo in range(0,len(idx),4):
            real=idx[lo:lo+4];pad=[i for i in range(4) if i not in real][:4-len(real)];active=set(real)-set(fixes)
            if active:run(real+pad,active)
        for i,original_batch in fixes.items():run([by_id[x] for x in original_batch],{i})
    cases=[found[i] for i in idx];summary={}
    for role in route['roles']:
        summary[role]={}
        for b in LAYERS:
            v=[r['roles'][role][str(b)] for r in cases]
            summary[role][str(b)]={'n':len(v),'Q_unchanged':sum(x['Q_unchanged'] for x in v),
                'nonzero_read':sum(x['actual_delta_L2']>0 for x in v),'median_cosine':float(np.median([x['cosine'] for x in v])),
                'median_relative_error':float(np.median([x['relative_error'] for x in v]))}
    C.save(dest,{**plan,'status':'complete','baseline_gap_matches':sum(r['baseline_gap_equal'] for r in cases),'summary':summary,'rows':cases})
    print(summary,flush=True)


if __name__=='__main__':main()
