"""Execute only admitted single-branch edits on a new family partition."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
from clt_b26_conflict import (BLOCK, CAP, HEAD, KS, OUT, QS, SEED, TMP, archive, checked, commit, materials, seal)
from deploy.storage_policy import check_write, checked_dir
from source_binding_design import MODELS, read, sha

CONFLICT_CELLS = (2, 3, 6, 7, 10, 11, 14, 15)
EDITS = ('target', 'random', 'sham', 'restore')


def freeze():
    pre=checked(); candidate=seal(); e=read(OUT/'evaluation.json')
    assert e['candidate_sha']==sha(OUT/'candidate.json')
    branches=e['causal_admitted']
    assert branches and all(e['gates'][b]['passed'] for b in branches)
    commit('intervention-freeze.json',dict(utc=datetime.now(timezone.utc).isoformat(),evaluation_sha=sha(OUT/'evaluation.json'),
           provenance=pre['provenance'],code=sha(Path(__file__)),predictor_sha=candidate['predictor_sha'],branches=branches,
           calibration_sha=pre['materials']['pilot']['sha'],test_sha=pre['materials']['intervention']['sha'],dose=.5,
           matching=[.9,1.1],eligibility='Each branch has independently passed its frozen natural prediction gate'))


def run(stage):
    import torch as t
    from torch.nn import functional as F
    from shared.dist_protocol.natural_register_runner import snapshot
    from shared.dist_protocol.mechanism_trace import GpuLease
    from source_binding_runtime import Runtime
    from source_binding_resources import resources
    pre=checked(); candidate=seal(); lock=read(OUT/'intervention-freeze.json'); e=read(OUT/'evaluation.json')
    assert lock['code']==sha(Path(__file__)) and lock['predictor_sha']==candidate['predictor_sha']
    assert lock['evaluation_sha']==sha(OUT/'evaluation.json') and lock['branches']==e['causal_admitted']
    branches=lock['branches']
    if stage=='intervention':
        calibration=read(OUT/'calibration-intervention.json')
        assert calibration['freeze_sha']==sha(OUT/'intervention-freeze.json')
        branches=[b for b in branches if calibration['results'][b]['technical_pass']]
        assert branches
    src='pilot' if stage=='calibration' else 'intervention'
    selected=[r for r in materials(src,pre) if r['cell']==CONFLICT_CELLS[(r['family']//2)%8]]
    arms=['base']+[f'{b}_{m}' for b in branches for m in EDITS]
    rows=[dict(r,arm=arm) for r in selected for arm in arms]
    assert len({r['family'] for r in selected})==len(selected)
    checked_dir(TMP); check_write(OUT,48<<20,operation='Conflict intervention')
    clock=time.monotonic(); rt=Runtime('qwen14')
    t.backends.cuda.matmul.allow_tf32=False
    fitted=np.load(OUT/'predictor.npz')

    def progress(n):
        elapsed=time.monotonic()-clock
        info=dict(stage=stage,completed=n,total=40,elapsed=elapsed,eta=(40-n)*elapsed/n if n else None,
                  cuda_peak=t.cuda.max_memory_allocated(),**resources())
        with (OUT/f'{stage}-intervention-progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(info)+'\n')
        print(json.dumps(info),flush=True)
        assert elapsed<15*60 and (info.get('shared_gpu_bytes') or 0)<1<<30
        assert info['ram_available_bytes']>2<<30 and info['commit_available_bytes']>4<<30

    progress(0)
    assert snapshot(MODELS['qwen14'])==pre['provenance']['model_revision']
    vocab=sorted({v for r in selected for v in r['choice_ids']})
    full=rt.weight('lm_head.weight','cpu'); decode=full[vocab].contiguous().to('cuda').float(); del full
    decode_id={v:i for i,v in enumerate(vocab)}
    traces={}

    def put(name,x):traces.setdefault(name,[]).append(x.detach().float().cpu().numpy())

    def block(b):
        layer=Runtime.block(rt,b)
        if b!=BLOCK:return layer
        att=layer.self_attn
        weights=dict(q=att.q_proj.weight[QS].float(),k=att.k_proj.weight[KS].float(),v=att.v_proj.weight[KS].float(),o=att.o_proj.weight[:,QS].float())
        units=t.tensor(candidate['units'],device='cuda')
        groups={}
        for key in branches:
            mu=t.tensor(fitted[f'mean_{key}'],device='cuda')
            groups[key]=[(idx,mu[idx],weights[key][:,idx].contiguous()) for name in ('selected','random')
                         for idx in [t.tensor(fitted[f'{key}_{name}_128'],device='cuda')]]
        s=dict(offset=0)

        def prepare(m,args,kw):s.update(z=kw['hidden_states'],mask=kw['attention_mask'][:,0,-1],pe=kw['position_embeddings'])
        def native(key,region):
            def hook(m,args,out):s[key]=out[...,region].contiguous()
            return hook

        def transformed(key,x,cos,sin):
            if key=='v':return x
            x=(att.q_norm if key=='q' else att.k_norm)(x)
            return x*cos+rt.mod.rotate_half(x)*sin

        def at_head(q,k,v,mask,records):
            score=(q[:,0,None].float()*k.float()).sum(-1)*att.scaling
            probs=score.masked_fill(~mask,float('-inf')).softmax(-1)
            mass=(probs*records).sum(-1)
            vec=F.scaled_dot_product_attention(q.unsqueeze(1),k.unsqueeze(1),v.unsqueeze(1),
                                               attn_mask=mask[:,None,None],scale=att.scaling)[:,0,0]
            return mass,vec

        def before_o(m,args):
            input=args[0]
            cos,sin=s['pe']; z=s['z']; n,w=z.shape[:2]
            batch=rows[s['offset']:s['offset']+n]; s['offset']+=n
            q,k,v=transformed('q',s['q'][:,-1:],cos[:,-1:],sin[:,-1:]),transformed('k',s['k'],cos,sin),s['v']
            original=input[:,-1,QS].clone()
            computed=[]; base_errors=[]; amount=[]; expected=[]; base_mass=[]
            result=input.clone()
            for i,row in enumerate(batch):
                p=w-len(row['tokens']); part=slice(i,i+1)
                gold=row['target_record']; other=row['row_slot'] if row['rule']==0 else row['name_slot']
                mask=s['mask'][part]; sites=t.zeros((2,w),device='cuda')
                for j,slot in enumerate((gold,other)):sites[j,[x+p for x in row['records'][slot]]]=1
                a,z0=at_head(q[part],k[part],v[part],mask,sites)
                base_errors.append((z0[0].float()-original[i].float()).abs().max())
                base_mass.append(a)
                method=row['arm']
                dose=t.tensor(0.,device='cuda'); gold_mass=a
                if method!='base' and not method.endswith('_sham'):
                    key,mode=method.split('_')
                    if key=='q':
                        pos=[w-1]; c,h=cos[part,-1:],sin[part,-1:]; full=s[key][part,-1:]
                    else:
                        pos=sorted({p+x for j in (gold,other) for x in row['records'][j]}); c,h=cos[part,pos],sin[part,pos]; full=s[key][part,pos]
                    f0=transformed(key,full,c,h)
                    yy=z[part,pos]
                    contributions=[F.linear(yy[...,ids].float()-mu,W) for ids,mu,W in groups[key]]
                    shift=-.5*contributions[0]
                    edited=transformed(key,(full.float()+shift).bfloat16(),c,h)
                    target_dose=(edited.float()-f0.float()).norm()
                    if mode=='random':
                        alt=contributions[1]
                        center=shift.norm()/alt.norm().clamp_min(1e-10)
                        scale=center*t.linspace(-2.5,2.5,81,device='cuda').exp()
                        variants=transformed(key,(full.float()-scale[:,None,None]*alt).bfloat16(),c,h)
                        distances=(variants.float()-f0.float()).flatten(1).norm(dim=-1)
                        nearest=(distances-target_dose).abs().argmin()
                        edited=variants[nearest:nearest+1]
                    dose=(edited.float()-f0.float()).norm()
                    local=dict(q=q[part].clone(),k=k[part].clone(),v=v[part].clone())
                    if key=='q':local['q']=edited
                    else:local[key][:,pos]=edited
                    gold_mass,vec=at_head(local['q'],local['k'],local['v'],mask,sites)
                    result[i,-1,QS]=vec[0]
                    if mode=='restore':
                        result[i,-1,QS]=original[i]
                        gold_mass=a
                computed.append(gold_mass)
                amount.append(dose)
                w_gold=decode[decode_id[row['choice_ids'][gold]]]-decode[decode_id[row['choice_ids'][other]]]
                expected.append(w_gold @ (weights['o'] @ result[i,-1,QS].float()))
            put('head',result[:,-1,QS]);put('mass',t.stack(computed));put('baseline_mass',t.stack(base_mass));put('dose',t.stack(amount));put('error',t.stack(base_errors));put('local_output_margin',t.stack(expected))
            return (result,)

        def mlp(m,args):put('A',args[0][:,-1,units])
        att.register_forward_pre_hook(prepare,with_kwargs=True)
        for k,region in [('q',QS),('k',KS),('v',KS)]:getattr(att,k+'_proj').register_forward_hook(native(k,region))
        att.o_proj.register_forward_pre_hook(before_o)
        layer.mlp.down_proj.register_forward_pre_hook(mlp)
        return layer

    def cb(b,layer,x,y,rr,width,offset=0,**kw):
        if b>=0 and offset+len(rr)==len(rows) and (b%5==4 or b==BLOCK):progress(b+1)

    rt.block=block
    with GpuLease(),t.inference_mode():
        _,logits=rt.forward(rows,cb); scores=rt.score(logits,rows)
        result=logits.reshape(len(selected),len(arms),-1)
        exact={label:bool(t.equal(result[:,0].view(t.int32),result[:,i].view(t.int32))) for i,label in enumerate(arms) if label.endswith(('_sham','_restore'))}
        del result,logits
    n,m=len(selected),len(arms)
    item={k:np.concatenate(v).reshape(n,m,*v[0].shape[1:]) for k,v in traces.items()}
    item['ids']=np.array([r['id'] for r in selected]);item['gold_margin']=np.array([s['gold_margin'] for s in scores]).reshape(n,m)
    item['top1']=np.array([s['top1_correct'] for s in scores]).reshape(n,m)
    boot=np.random.default_rng(SEED+70).integers(0,n,(10000,n))
    def interval(x):return [float(x.mean()),*np.quantile(x[boot].mean(1),[.0125,.9875]).tolist()]
    response=np.log(np.maximum(item['mass'][...,0],1e-20))-np.log(np.maximum(item['mass'][...,1],1e-20))
    summary={}
    for k in branches:
        target,random,sham,restore=[arms.index(f'{k}_{label}') for label in EDITS]
        ratio=item['dose'][:,random]/np.maximum(item['dose'][:,target],1e-12)
        match=float(((ratio>=.90)&(ratio<=1.10)).mean())
        exact_internal=all(np.array_equal(item[key][:,0].view(np.uint32),item[key][:,j].view(np.uint32)) for key in ('head','A') for j in (sham,restore))
        exact_output=exact[arms[sham]] and exact[arms[restore]]
        touched=np.array_equal(item['dose'][:,target],item['dose'][:,restore]) and np.min(item['dose'][:,target])>0
        measure=response if k in ('q','k') else item['local_output_margin']
        primary=interval(measure[:,random]-measure[:,target]); alternate=interval(item['gold_margin'][:,random]-item['gold_margin'][:,target])
        descriptive_by_rule={}
        for rule in (0,1):
            mask=np.array([r['rule']==rule for r in selected])
            descriptive_by_rule[str(rule)]=dict(n=int(mask.sum()),matched_effect=float((measure[:,random]-measure[:,target])[mask].mean()),
                                               target_vs_native=float((measure[:,target]-measure[:,0])[mask].mean()))
        mlp_rms=np.sqrt(np.mean((item['A'][:,target]-item['A'][:,0])**2,axis=0))
        mlp_random_rms=np.sqrt(np.mean((item['A'][:,random]-item['A'][:,0])**2,axis=0))
        verified=(match>=.95 and exact_internal and exact_output and touched and item['error'].max()<=CAP and item['top1'][:,0].mean()>=.95 and (item['gold_margin'][:,0]>0).mean()>=.95)
        summary[k]=dict(technical_pass=bool(verified),dose_match=match,range=[float(ratio.min()),float(ratio.max())],internal_exact=exact_internal,logits_exact=exact_output,
                        target_restore_dose_exact=bool(touched),primary=primary,passes=bool(stage=='intervention' and verified and primary[1]>0),
                        descriptive_by_rule=descriptive_by_rule,
                        activation_change_by_unit=[dict(unit=u,target_rms=float(a),random_rms=float(b)) for u,a,b in zip(candidate['units'],mlp_rms,mlp_random_rms,strict=True)],
                        natural_target_change=interval(measure[:,target]-measure[:,0]),answer_margin=alternate,
                        top1=[float(item['top1'][:,j].mean()) for j in (0,target,random,sham,restore)])
    name=f'{stage}-intervention'
    archive(name+'.npz',item)
    commit(name+'.json',dict(provenance=pre['provenance'],freeze_sha=sha(OUT/'intervention-freeze.json'),traces_sha=sha(OUT/(name+'.npz')),arms=arms,
                             families=n,max_head_replay_error=float(item['error'].max()),results=summary,all_technical_pass=all(v['technical_pass'] for v in summary.values()),
                             elapsed=time.monotonic()-clock,scores=scores))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    stage=sys.argv[1]
    freeze() if stage=='freeze' else run(stage)
