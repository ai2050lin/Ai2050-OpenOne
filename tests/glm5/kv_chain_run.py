"""Run 14B fact-to-State-to-Code verification on fixed right-padded tensors."""

from itertools import batched
from pathlib import Path
import sys
import time

import numpy as np

import kv_length_run as Prior
import kv_chain_material as M


A,B,P,N,O,ROOT = Prior.A,Prior.B,Prior.P,Prior.N,Prior.O,Prior.ROOT
RUN = 'KV-CHAIN-20261117-v1'
OUT = ROOT/'tests/glm5/result/kv_chain_20261117_v1'
TMP = ROOT/'tests/glm5_temp/kv_chain_20261117_v1'
WIDTH = 352
BATCH = 8
SLAB = 128
BUDGET = 112*2**20
TARGET = (25,18)
EARLY = (20,21)


def setup():
    A.OUT,A.TEMP,A.RUN=OUT,TMP,RUN
    A.R.OUT,A.R.TEMP,A.R.RUN=OUT,TMP,RUN
    P.OUT,P.TMP,P.RUN,P.BUDGET=OUT,TMP,RUN,BUDGET
    P.check_write(OUT,BUDGET,operation=RUN)
    P.check_write(TMP,BUDGET,operation=RUN)
    P.setup()
    B.C.OUT=OUT


def stamp():
    return O.stamp()


def packet(model,rows):
    from kv_length_shape import packet as make
    assert max(len(r['token']['ids']) for r in rows)<=WIDTH
    return make(model,rows,WIDTH)


def forward(model,rows,label,site_step=None,output=None):
    batches = list(batched(rows,SLAB))
    for part, members in enumerate(batches,1):
        stamp0 = time.monotonic()
        panels = [packet(model,b) for b in batched(members,BATCH)]
        N.progress(label,0,40,stamp0,part=part,parts=len(batches))
        for i in range(40):
            with model.block(i) as layer:
                for p in panels:
                    x,mask=p['x'].cuda(),p['mask'].cuda()
                    q,k,v=model.qkv(layer,x,[y.cuda() for y in p['angles']])
                    o=model.attention(layer,q,k,v,mask)
                    if site_step is not None:
                        site_step(model,layer,i,p,x,q,k,v,o,mask)
                    p['x']=model.residual(layer,x,o).cpu()
            if (i+1)%5==0: N.progress(label,i+1,40,stamp0,part=part,parts=len(batches))
        for p in panels:
            yield p if output is None else output(model,p)


def verify_generation(model, rows, label, selector=None):
    """Query original all-vocabulary argmax across the proposed continuation."""
    records, pending, spaces = {},rows,{}
    for trial in range(4):
        follow=[]
        for p in forward(model,pending,label+f'-{trial+1}'):
            start=min(r['token']['start'] for r in p['rows'])-1
            stop=int(p['c'].max())+1
            x=p['x'][:,start:stop].cuda()
            y=model.logits(x)
            hit=y.argmax(-1).cpu().numpy()
            for j,r in enumerate(p['rows']):
                k=r['token']; ids=k['ids']; begin=k['start']; pivot=k['slots'][0]-1; end=len(ids)-1
                prediction=hit[j,begin-1-start:end+1-start].copy()
                expected=ids[begin:]; actual_state=int(hit[j,pivot-start]); original_state=actual_state
                if selector and r['id'] in selector:
                    actual_state=selector[r['id']]['top']
                    prediction[pivot-begin+1]=actual_state
                difference=np.flatnonzero(prediction[:-1]!=expected)
                seen=int(difference[0]) if len(difference) else len(expected)
                token=int(prediction[seen]); matched=not len(difference)
                if not matched and trial!=3 and (begin+seen)>pivot+1:
                    shown=model.tok.decode([token]); new=model.whitespace_proposal(r,begin+seen,token,shown) if shown.endswith('\n') else None
                    if new:
                        line,plan=new
                        spaces.setdefault(r['id'],[]).append((line,token)); follow.append(dict(r,token=plan)); continue
                codes=y[j,end-start,model.letters].float().cpu().tolist() if matched else None
                records[r['id']]=dict(id=r['id'], matched=matched, actual_state=actual_state, original_state=original_state,
                    state_id=ids[pivot+1], state_verified=(begin+seen>=pivot+1 and actual_state==ids[pivot+1]),
                    top=token if matched else None, pq=codes, prefix=ids if matched else None, breaks=k['breaks'],
                    emitted=expected[:seen]+[token], text=model.tok.decode(expected[:seen]+[token]),
                    divergence=None if matched else begin+seen, whitespace=spaces.get(r['id'],[]))
        pending=follow
        if not pending: break
    return [records[r['id']] for r in rows]


def attach(model,rows,records):
    return [dict(r,token=k,sites=M.groups(model.tok,dict(r,token=k))) for r,s in zip(rows,records,strict=True)
            if s['matched'] for k in [M.token_plan(model.tok,r,s['breaks'])] if k['ids']==s['prefix']]


def cells(rows, records, conflict=False):
    grid={}; eligible=[]
    for r,s in zip(rows,records,strict=True):
        k=(r['frame'],r['text'],r['distance'],r['compete'],r['map'])
        item=grid.setdefault(k,dict(n=0,state=0,code=0,format=0))
        item['n']+=1; item['state']+=int(s['state_verified']); item['format']+=int(s['matched'])
        letters=r['token']['letters']; valid=s['matched'] and s['state_verified'] and s['top'] in letters
        good=valid if conflict else valid and s['top']==letters[r['gold']=='Q']
        item['code']+=int(good)
        if good:eligible.append(r['id'])
    return dict(cells=[dict(key=k,**v) for k,v in sorted(grid.items())],eligible=eligible,
                n=len(rows),passed=all(v['code']/v['n']>=.85 for v in grid.values()))


def behavior(model, rows, stage):
    name=stage+'-greedy.json'
    if (OUT/name).exists():return A.load(name)
    seen=verify_generation(model,rows,stage+'-greedy')
    result=dict(records=seen, gate=cells(rows,seen))
    A.save(name,result)
    print(stage,'behavior',len(result['gate']['eligible']),'/',len(rows),'all cells',result['gate']['passed'],flush=True)
    return result


def pilot(model):
    rows=M.payload(model.tok,'pilot')
    A.save('pilot-material-index.json',dict(ids=[r['id'] for r in rows],lengths=sorted(set(len(r['token']['ids']) for r in rows))))
    return behavior(model,rows,'pilot')


def sources():
    names=set(P.load(Prior.OUT/'plan.json')['sources']) | {'tests/glm5/kv_length_run.py','tests/glm5/kv_length_shape.py'}
    names.update(x.relative_to(ROOT).as_posix() for x in Path(__file__).parent.glob('kv_chain_*.py') if x.stem not in ('kv_chain_report',))
    return {k:A.digest(ROOT/k) for k in sorted(names)}


def main(stage):
    setup()
    import torch
    torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
    with P.GpuLease(), torch.no_grad():
        t=time.monotonic(); N.progress('chain-load',0,2,t)
        snapshot=Prior.identity(); N.progress('chain-load',1,2,t)
        model=N.Engine(); N.progress('chain-load',2,2,t)
        try:
            if stage=='pilot':pilot(model)
            else:
                from kv_chain_study import run
                run(model,stage,snapshot)
        finally:
            model.close()


if __name__=='__main__':
    sys.modules['kv_chain_run']=sys.modules[__name__]
    main(sys.argv[1])
