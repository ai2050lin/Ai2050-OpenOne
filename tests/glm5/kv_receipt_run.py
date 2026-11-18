"""Phase 2955 controlled generated-history insertion and native held-out assays."""

from itertools import batched
from pathlib import Path
import json
import sys
import time

import numpy as np

import kv_chain_run as Chain
import kv_receipt_material as M


A,B,P,N,ROOT=Chain.A,Chain.B,Chain.P,Chain.N,Chain.ROOT
OUT=ROOT/'tests/glm5/result/kv_receipt_20261117_v1'
TMP=ROOT/'tests/glm5_temp/kv_receipt_20261117_v1'
BUDGET=68*2**20
WIDTH=256
CONSORTIUM=10


def setup(budget=BUDGET):
    Chain.OUT,Chain.TMP,Chain.RUN,Chain.WIDTH,Chain.BATCH,Chain.SLAB,Chain.BUDGET=OUT,TMP,M.RUN,WIDTH,CONSORTIUM,120,budget
    Chain.setup()


def digest(items):
    from hashlib import sha256
    return sha256(json.dumps(items,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def panels(model,stage):
    return [r for family in M.families(model.tok,stage) for r in family]


def donors(model,rows,stage):
    prompts=[]
    for group in batched(rows,CONSORTIUM):
        base=group[0]
        if not base['conflict']:continue
        donor=M.build(model.tok,base,donor=True)
        assert donor['prompt'].split('\nNote entry:',1)[-1]==base['prompt'].split('\nNote entry:',1)[-1]
        donor['token']['ids']=donor['token']['ids'][:donor['token']['slots'][0]]
        prompts.append(donor)
    out=[]; vectors=[]
    for p in Chain.forward(model,prompts,stage+'-role-donor'):
        t=model.t; ix=t.arange(len(p['rows'])); latent=p['x'][ix,p['c']]
        actual=model.logits(latent.cuda()).argmax(-1).cpu().tolist()
        out.extend((r['family'],token) for r,token in zip(p['rows'],actual,strict=True))
        vectors.append(latent.view(t.int16).numpy().copy())
    if vectors:B.write_np(stage+'-donors.npz',dict(family=np.array([x for x,_ in out]),top=np.array([x for _,x in out]),native=np.concatenate(vectors)))
    return dict(out)


def retained(model,p,selector):
    start=min(x['token']['start'] for x in p['rows'])-1
    t=model.t; end=int(p['c'].max())+1
    score=model.logits(p['x'][:,start:end].cuda())
    choices=score.argmax(-1).cpu().numpy(); out=[]
    for j,r in enumerate(p['rows']):
        k=r['token']; ids=k['ids']; first=k['start']; G=k['slots'][0]; I=r['insertion']; n=r['n']; C=len(ids)-1
        from_=first-1-start
        greedy=choices[j,from_:C+1-start].copy()
        raw_state=int(greedy[G-first]); greedy_state=int(selector.get(r['family'],raw_state))
        greedy[G-first]=greedy_state
        # Only the inserted history is set externally. The target's argmax at those positions is still archived.
        trial=np.array(ids[first:]); inner=greedy[I-first:I-first+n].tolist()
        greedy[I-first:I-first+n]=trial[I-first:I-first+n]
        mismatch=np.flatnonzero(greedy[:-1]!=trial)
        failure=None if not len(mismatch) else int(mismatch[0])+first
        next_=int(greedy[-1])
        out.append(dict(id=r['id'],family=r['family'],track=r['track'],n=n,frame=r['frame'],
            verified=failure is None,first_failure=failure,raw_state=raw_state,actual_state=greedy_state,state_ref=ids[G],
            planned_separator=int(ids[I+n]),model_separator=int(greedy[I+n-first]),
            failure_top=None if failure is None else int(greedy[failure-first]),
            role_donor=r['family'] in selector,inside_model_top=inner,inside_actual=ids[I:I+n],
            raw_output=greedy.tolist(),code_top=next_,pq=score[j,C-start,model.letters].float().cpu().tolist(),
            gold=k['letters'][r['gold']=='Q'],code_state=k['letters'][r['code']=='Q'], code_ids=k['letters']))
    return out


def gate(rows,records):
    result={}; good=[]
    for fam,observed in zip(batched(rows,CONSORTIUM),batched(records,CONSORTIUM),strict=True):
        conditions=[]
        for r,s in zip(fam,observed,strict=True):
            assert r['id']==s['id']
            key=(r['frame'],r['conflict'],r['track'],r['n']); n=result.setdefault(key,[0,0])
            # Qualify baseline behavior independently of the direction of any inserted-branch answer.
            passed=s['verified'] and s['code_top'] in s['code_ids'] and (r['n'] or r['conflict'] or s['code_top']==s['gold'])
            conditions.append(bool(passed))
            n[1]+=1;n[0]+=int(bool(passed))
        if all(conditions):
            assert len({s['actual_state'] for s in observed})==1
            good.append(fam[0]['family'])
    passing={k[:2] for k in result if all(y/n>=.85 for cell,(y,n) in result.items() if cell[:2]==k[:2])}
    elig=[r['family'] for r in rows[::CONSORTIUM] if r['family'] in good and (r['frame'],r['conflict']) in passing]
    return dict(eligible=elig,complete=good,frame_modes=sorted(passing),cells=[dict(frame=k[0],conflict=k[1],track=k[2],n=k[3],good=y,total=n) for k,(y,n) in sorted(result.items())])


def pilot(model):
    rows=panels(model,'pilot'); selector=donors(model,rows,'pilot')
    records=[]
    for p in Chain.forward(model,rows,'pilot-histories'):
        records.extend(retained(model,p,selector))
    g=gate(rows,records)
    A.save('pilot.json',dict(gate=g, records=records,digest=digest(rows)))
    print('Pilot',len(g['eligible']),len(g['complete']),len(rows)//CONSORTIUM,flush=True)
    for r,s in zip(rows,records,strict=True):
        if s['first_failure'] is not None:
            i=s['first_failure']; print(r['id'],i-r['insertion'],repr(model.tok.decode(s['failure_top'])),repr(model.tok.decode(r['token']['ids'][i])),flush=True)


def main():
    setup()
    import torch
    torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
    stage=sys.argv[1]
    with P.GpuLease(),torch.no_grad():
        t=time.monotonic(); N.progress('receipt-load',0,2,t)
        identity=Chain.Prior.identity();N.progress('receipt-load',1,2,t)
        model=N.Engine();N.progress('receipt-load',2,2,t)
        try:
            if stage=='pilot':pilot(model)
            else:
                from kv_receipt_study import run
                run(model,stage,identity)
        finally:model.close()


if __name__=='__main__':
    sys.modules['kv_receipt_run']=sys.modules[__name__]
    main()
