"""Matched native Q/K contribution edits for admitted three-cue branches."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import clt_b26_three_cues as B
from deploy.storage_policy import checked_dir,check_write
from source_binding_design import MODELS,read,sha

MODES=('target','random','sham','restore')


def freeze():
    pre=B.common.checked(); cand=B.seal(); result=read(B.OUT/'evaluation.json')
    branches=result['admitted']; assert branches and all(result['gates'][k]['passed'] for k in branches)
    assert result['candidate_sha']==sha(B.OUT/'candidate.json')
    B.common.commit('causal-freeze.json',dict(utc=datetime.now(timezone.utc).isoformat(),code=sha(Path(__file__)),branches=branches,prepared=cand['predictor_sha'],
                 evaluation_sha=sha(B.OUT/'evaluation.json'),prereg_sha=sha(B.OUT/'prereg.json'),scope=pre['materials']['intervention']['sha'],
                 dosage=.50,match_ratio=[.9,1.1],representative='cell=(family//4)%12; titles stratified; physical scope from parent preregistration'))


def run(stage):
    import torch as t
    from torch.nn import functional as F
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    from source_binding_runtime import Runtime
    from source_binding_resources import resources
    pre=B.common.checked(); lock=read(B.OUT/'causal-freeze.json');cand=B.seal();e=read(B.OUT/'evaluation.json')
    assert lock['code']==sha(Path(__file__)) and lock['prereg_sha']==sha(B.OUT/'prereg.json') and lock['prepared']==cand['predictor_sha']
    assert lock['evaluation_sha']==sha(B.OUT/'evaluation.json') and lock['branches']==e['admitted']
    branches=lock['branches']
    if stage=='intervention':
        cal=read(B.OUT/'calibration-causal.json'); assert cal['freeze_sha']==sha(B.OUT/'causal-freeze.json')
        branches=[k for k in branches if cal['branches'][k]['technical_pass']]; assert branches
    source='pilot' if stage=='calibration' else 'intervention'
    families=B.materials(source,pre)
    # Cycle through all query rules during pilot even when some 2x2 cells have more pilot representatives.
    target=lambda i: (i//4)%12 if stage=='intervention' else ((i//4)*5)%12
    reps=[r for r in families if r['cell']==target(r['family'])]
    assert len(reps)==len({r['family'] for r in reps})==B.SIZES[source]
    arms=['base']+[f'{k}_{m}' for k in branches for m in MODES]
    rows=[dict(r,arm=m) for r in reps for m in arms]
    check_write(B.OUT,18<<20,operation='Three-cue causal evidence');checked_dir(B.TMP)
    clock=time.monotonic(); rt=Runtime('qwen14');t.backends.cuda.matmul.allow_tf32=False
    model=np.load(B.OUT/'predictor.npz'); saved={}

    def progress(n):
        elapsed=time.monotonic()-clock; row=dict(stage=stage,completed=n,total=40,elapsed=elapsed,eta=elapsed*(40-n)/n if n else None,
                     cuda_peak=t.cuda.max_memory_allocated(),**resources())
        with (B.OUT/f'{stage}-causal-progress.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        assert elapsed<15*60 and (row.get('shared_gpu_bytes') or 0)<1<<30 and row['ram_available_bytes']>2<<30

    def put(k,x):saved.setdefault(k,[]).append(x.detach().float().cpu().numpy())

    def block(b):
        layer=Runtime.block(rt,b)
        if b!=B.BLOCK:return layer
        att=layer.self_attn
        weight={'q':att.q_proj.weight[B.QS].float(),'k':att.k_proj.weight[B.KS].float(),'o':att.o_proj.weight[:,B.QS].float()}
        compact={}
        for k in branches:
            for mode in ('selected','random'):
                ids=t.tensor(model[f'{k}_{mode}_512'],device='cuda')
                compact[k,mode]=(ids,t.tensor(model['mean_'+k],device='cuda')[ids],weight[k][:,ids].contiguous())
        state={'offset':0}

        def prepare(m,args,kw):state.update(z=kw['hidden_states'],mask=kw['attention_mask'][:,0,-1],pe=kw['position_embeddings'])
        def capture(k,loc):
            def hook(m,args,out):state[k]=out[...,loc].contiguous()
            return hook

        def normalized(branch,x,cos,sin):
            x=(att.q_norm if branch=='q' else att.k_norm)(x)
            return x*cos+rt.mod.rotate_half(x)*sin

        def head(q,k,v,mask,sites):
            score=(q.float()*k.float()).sum(-1)*att.scaling
            alpha=score.masked_fill(~mask,float('-inf')).softmax(-1)
            mass=(alpha*sites).sum(-1)
            h=F.scaled_dot_product_attention(q.unsqueeze(1),k.unsqueeze(1),v.unsqueeze(1),attn_mask=mask[:,None,None],scale=att.scaling)[0,0,0]
            return mass,h

        def edit(m,args):
            raw=args[0];rewritten=raw.clone();state_z=state['z'];n,T=state_z.shape[:2]
            batch=rows[state['offset']:state['offset']+n];state['offset']+=n
            cos,sin=state['pe'];orig_q=normalized('q',state['q'][:,-1:],cos[:,-1:],sin[:,-1:]);orig_k=normalized('k',state['k'],cos,sin)
            masses=[];doses=[];errors=[];writes=[]; touched=[]
            for i,row in enumerate(batch):
                part=slice(i,i+1); p=T-len(row['tokens']); q=orig_q[part];k=orig_k[part];v=state['v'][part];mask=state['mask'][part]
                sites=t.zeros((5,T),device='cuda')
                for j,span in enumerate(row['records']):sites[j,[p+x for x in span]]=1
                baseline,vector=head(q,k,v,mask,sites); natural=raw[i,-1,B.QS]
                errors.append((vector.float()-natural.float()).abs().max())
                mode=row['arm']; gold_mass=baseline; dose=t.zeros((),device='cuda'); realized=t.zeros((),device='cuda')
                if mode!='base' and not mode.endswith('_sham'):
                    branch,kind=mode.split('_'); positions=[T-1] if branch=='q' else sorted({p+x for r in row['cue_targets'] for x in row['records'][r]})
                    full=state[branch][part,positions];c=cos[part,positions];s=sin[part,positions]; base_norm=normalized(branch,full,c,s)
                    z=state_z[part,positions]
                    def component(kind):
                        ix,mu,W=compact[branch,kind]
                        return F.linear(z[...,ix].float()-mu,W)
                    shift=-.5*component('selected')
                    proposed=normalized(branch,(full.float()+shift).bfloat16(),c,s)
                    target=(proposed.float()-base_norm.float()).norm()
                    if kind=='random':
                        control=component('random');center=shift.norm()/control.norm().clamp_min(1e-10)
                        factors=center*t.linspace(-2.5,2.5,81,device='cuda').exp()
                        alternatives=normalized(branch,(full.float()-factors[:,None,None]*control).bfloat16(),c,s)
                        lengths=(alternatives.float()-base_norm.float()).flatten(1).norm(dim=-1)
                        best=(lengths-target).abs().argmin();proposed=alternatives[best].unsqueeze(0)
                    dose=(proposed.float()-base_norm.float()).norm()
                    if branch=='q':q=proposed
                    else:k=k.clone();k[:,positions]=proposed
                    gold_mass,after=head(q,k,v,mask,sites)
                    realized=(after.float()-natural.float()).norm()
                    rewritten[i,-1,B.QS]=after
                    if kind=='restore':rewritten[i,-1,B.QS]=natural;gold_mass=baseline
                masses.append(gold_mass);doses.append(dose);touched.append(realized)
                writes.append((weight['o']@(rewritten[i,-1,B.QS].float()-natural.float())).norm())
            put('mass',t.stack(masses));put('head',rewritten[:,-1,B.QS]);put('dose',t.stack(doses));put('write_delta',t.stack(writes));put('sdpa',t.stack(errors));put('computed_head_delta',t.stack(touched))
            return (rewritten,)

        att.register_forward_pre_hook(prepare,with_kwargs=True)
        for branch,region in [('q',B.QS),('k',B.KS),('v',B.KS)]:getattr(att,branch+'_proj').register_forward_hook(capture(branch,region))
        att.o_proj.register_forward_pre_hook(edit)
        return layer

    def update(b,layer,x,y,rr,T,offset=0,**kw):
        if b>=0 and offset+len(rr)==len(rows) and (b%5==4 or b==B.BLOCK):progress(b+1)

    rt.block=block; progress(0)
    assert snapshot(MODELS['qwen14'])==pre['provenance']['model_revision']
    with GpuLease(),t.inference_mode():
        _,logits=rt.forward(rows,update);scores=B.grade(rt,logits,rows); view=logits.reshape(len(reps),len(arms),-1)
        exact={arm:bool(t.equal(view[:,0].view(t.int32),view[:,i].view(t.int32))) for i,arm in enumerate(arms) if arm.endswith(('_restore','_sham'))}
        del view,logits
    n,m=len(reps),len(arms);data={k:np.concatenate(chunks).reshape(n,m,*chunks[0].shape[1:]) for k,chunks in saved.items()}
    data.update(ids=np.array([r['id'] for r in reps]),answer=np.array([s['gold_margin'] for s in scores]).reshape(n,m),top1=np.array([s['top1_correct'] for s in scores]).reshape(n,m))
    bootstrap=np.random.default_rng(B.SEED+96).integers(0,n,(10000,n))
    def ci(x):return [float(x.mean()),*np.quantile(x[bootstrap].mean(1),[.0125,.9875]).tolist()]
    r=np.arange(n); cues=np.array([x['cue_targets'] for x in reps]); rule=np.array([x['rule'] for x in reps]); z=np.take_along_axis(data['mass'],cues[:,None,:],axis=2)
    logs=np.log(np.maximum(z,1e-20)); preference=1.5*(logs-logs.mean(2,keepdims=True))[r,:,rule]
    report={}
    for k in branches:
        target,control,sham,restore=[arms.index(f'{k}_{x}') for x in MODES]
        doses=data['dose'];ratio=doses[:,control]/np.maximum(doses[:,target],1e-12)
        match=float(((ratio>=.9)&(ratio<=1.1)).mean())
        same_head=all(np.array_equal(data['head'][:,0].view(np.uint32),data['head'][:,i].view(np.uint32)) for i in (sham,restore))
        same_logits=exact[arms[sham]] and exact[arms[restore]]
        restored=np.array_equal(doses[:,target],doses[:,restore]) and np.array_equal(data['computed_head_delta'][:,target],data['computed_head_delta'][:,restore]) and doses[:,target].min()>0
        tech=bool(match>=.95 and same_head and same_logits and restored and data['sdpa'].max()<=B.CAP and
                  min(data['top1'][:,0].mean(),(data['answer'][:,0]>0).mean())>=.95)
        diff=preference[:,control]-preference[:,target]; effect=ci(diff)
        report[k]=dict(technical_pass=tech,dose_match=match,ratio=[float(ratio.min()),float(ratio.max())],head_exact=same_head,logits_exact=same_logits,
                       restore_executed=bool(restored),primary=effect,passes=bool(stage=='intervention' and tech and effect[1]>0),
                       answer_margin=ci(data['answer'][:,control]-data['answer'][:,target]),top1=data['top1'][:,[0,target,control,sham,restore]].mean(0).tolist(),
                       effects_by_rule={B.RULES[j]:float(diff[rule==j].mean()) for j in range(3)},
                       mean_O_write_change=[float(data['write_delta'][:,i].mean()) for i in (target,control)])
    stem=f'{stage}-causal';B.common.archive(stem+'.npz',data)
    B.common.commit(stem+'.json',dict(freeze_sha=sha(B.OUT/'causal-freeze.json'),traces_sha=sha(B.OUT/(stem+'.npz')),arms=arms,branches=report,n=n,
                 max_head_sdpa_error=float(data['sdpa'].max()),scores=scores,provenance=pre['provenance'],elapsed=time.monotonic()-clock))
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    B.configure(); stage=sys.argv[1]
    freeze() if stage=='freeze' else run(stage)
