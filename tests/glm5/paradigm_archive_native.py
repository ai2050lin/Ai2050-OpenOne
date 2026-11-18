"""Frozen archive K/V, attention, G/U and residual-bypass interventions."""

import argparse
from itertools import batched
from pathlib import Path
import math
import time

import numpy as np

import paradigm_archive_field as A
import paradigm_path_core as P
from deploy.storage_policy import check_write


R,C,M,X,F,OUT=A.R,A.C,A.M,A.X,A.F,A.OUT
RUNNER='holdout-plan.json'
SLAB=64
ARMS=('sham','source_kv','source_k','source_v','head_donor','native_restore','freeze_g',
      'freeze_u','freeze_gu','freeze_control','receiver_matched','receiver_control_matched',
      'source_matched','source_control_matched','skip_only','mlp_only','block_later_head','block_later_gu')


def sha(name):
    return A.sha(OUT/name)


def source_sha():
    return A.sha(Path(__file__).resolve())


def freeze():
    assert not (OUT/RUNNER).exists()
    candidate=R.read('candidate.json')
    assert candidate['design']==sha('design.json')
    assert candidate['evidence']=={k:sha(k+'-files.json') for k in candidate['evidence']}
    plan=R.read('design.json')
    gate=R.read('behavior.json')
    assert gate['format']['passed']
    routes=[dict(route,control_head=(route['head']+10)%40) for route in candidate['routes']]
    gated=min(A.RULES[2:],key=lambda name:max(candidate['rule_bank_losses'][name]))
    R.write(RUNNER,dict(at=C.stamp(),parent=sha('design.json'),candidate=sha('candidate.json'),
        runner=source_sha(),routes=routes,rule=candidate['rule'],conditional_rule=gated,arms=ARMS,material=plan['material']['confirm'],
        geometry=dict(batch=8,width=M.F.WIDTH,slab=SLAB,base='code archive row',donor='paired prose archive row'),
        setup=dict(
            source='Native donor K+V, K only and V only for the six archive positions and selected GQA group, installed into the selected head at the stated query; other heads and queries stay identical at that block. Whole-head donor and exact native restoration at the same query.',
            receiver='Keep only selected G, U or both equal to original code-side values while maintaining source K/V edit. Recompute full MLP, compare to another frozen set of 32 neurons. Equate receiver doses after MLP using delivered BF16 residual norms.',
            match='Match selected and same-site GQA-separated control head before MLP at actual realized BF16 norm. Relative norm gap <=3%, cosine >=.90 against each actual displacement, each >=.003. Fixed grid .5–8x of larger original dose. Keep failed rows. Report extrapolation.',
            bypass='Skip only = source-edited after-attention residual plus original code-side MLP write; MLP only = original after-attention residual plus source-edited MLP write. Factorial interaction is full minus both isolated changes. Each uses same native add.',
            relay='Propagate the early route source edit. At the other candidate block restore either the later recipient head vector or the later 32 G/U values to their unedited code-side values. Score loss of early influence; retain invisible or zero cases.',
            propagation='Full native sequence recomputation with identical BF16 shape. Subsequent blocks propagate the full affected suffix through free onset. Unchanged causal prefix reused. Conditioned continuations ignored for native intervention.',
            calibration='First 8 heldout families are prespecified first packets of the same frozen confirmation. Stop if any no-op or restoration logit differs. Compute full sample only if this invariant holds, with no exclusions.',
            interpretation='Freeze, source and bypass effects are model interventions. Claim targeted closure only when family-mean signed source effect exceeds 10% of mean paired absolute total with positive 99% CI, source dose coverage and selected > control. Track G/U and bypass on both regimes and preserve reverse signs. Gates apply per candidate and stratum, no extrapolation from template terms.'
        )))
    print('Frozen heldout methods',sha(RUNNER),flush=True)


def endpoint(model,block,p,x,q,k,v,o,route,kind,head,sites):
    t=model.t
    j=t.arange(1,len(x),2,device='cuda')
    pos=sites[j]
    spans=t.tensor([r['spans']['archive'] for r in p['rows']],device='cuda')
    result=o.clone()
    if kind=='head':
        result[j,pos,head]=o[j-1,sites[j-1],head]
    else:
        group=head//model.groups
        dst=(j[:,None],group,spans[j])
        src=((j-1)[:,None],group,spans[j-1])
        kk=k.clone() if 'k' in kind else k
        vv=v.clone() if 'v' in kind else v
        if 'k' in kind:
            kk[dst]=k[src]
        if 'v' in kind:
            vv[dst]=v[src]
        changed=model.attention(block,q,kk,vv,p['mask'].cuda())
        result[j,pos,head]=changed[j,pos,head]
    return result


def match(model,r,left,right,j,pos):
    t=model.t
    b=r[j,pos];a=left[j,pos];c=right[j,pos]
    original=(a.float()-b.float(),c.float()-b.float())
    return P.realized_match(b,a,c,t,original)


def endpoint_arms(model,block,p,x,q,k,v,o,base,route):
    t=model.t
    sites=p['sites'][:,route['site_index']]
    j=t.arange(1,len(x),2,device='cuda')
    pos=sites[j]
    head=route['head']
    def writes(kind,other=head):
        return endpoint(model,block,p,x,q,k,v,o,route,kind,other,sites)
    def state(out):
        return P.extract(model,block,x,out,sites)
    fixed=state(o)
    assert t.equal(fixed['output'],base)
    modified=writes('kv')
    altered=state(modified)
    k_only=model.residual(block,x,writes('k'))
    v_only=model.residual(block,x,writes('v'))
    readout=model.residual(block,x,writes('head'))
    modified[j,pos,head]=o[j,pos,head]
    restored=model.residual(block,x,modified)
    assert t.equal(restored,base)
    selected=route['selected']
    g,u,gu=[P.neuron_restore(model,block,fixed,altered,selected,mode)[0] for mode in ('gate','up','both')]
    random=P.neuron_restore(model,block,fixed,altered,route['control'],'both')[0]
    g_matched,r_matched,receiver=match(model,altered['output'],gu,random,j,pos)
    c1,c2,source=match(model,fixed['after'],altered['after'],
                       state(writes('kv',route['control_head']))['after'],j,pos)
    def install(vec,site):
        branch=site.clone()
        branch[j,pos]=vec
        return branch
    def continue_after(vec):
        r=install(vec,fixed['after'])
        return r+block.mlp(block.post_attention_layernorm(r))
    # Use true down projection. Subtracting BF16 residuals would lose the original write.
    base_down=block.mlp.down_proj(fixed['a'])
    edit_down=block.mlp.down_proj(altered['a'])
    skip=altered['after']+base_down
    mlp=fixed['after']+edit_down
    alternatives=(base,altered['output'],k_only,v_only,readout,restored,g,u,gu,random,
        install(g_matched,altered['output']),install(r_matched,altered['output']),continue_after(c1),continue_after(c2),skip,mlp,
        altered['output'],altered['output'])
    low=int(pos.min());high=int(p['q'][:,0].max())+1
    assert low<high
    p['dose'][route['site']]=np.stack((source,receiver),1)
    observed=np.stack([P.selected_gua(model,s,selected+route['control'])[1::2] for s in (fixed,altered)],1)
    p['receiver'][route['site']]=observed
    return (low,high,t.stack([a[:,low:high] for a in alternatives]).cpu())


def advance(model,block,p,x,angles,start,stop,arm,relay=None):
    state=x.clone()
    state[:,start:stop]=arm
    q,k,v=model.qkv(block,state,angles)
    o=model.attention(block,q,k,v,p['mask'].cuda())
    if relay:
        target,base_heads,kind=relay
        t=model.t
        j=t.arange(1,len(x),2,device='cuda')
        sites=p['sites'][:,target['site_index']]
        pos=sites[j]
        if kind=='block_later_head':
            o[j,pos,target['head']]=base_heads[j,pos,target['head']]
        else:
            baseline=P.extract(model,block,x,base_heads,sites)
            changed=P.extract(model,block,state,o,sites)
            return P.neuron_restore(model,block,baseline,changed,target['selected'],'both')[0][:,start:stop]
    return model.residual(block,state,o)[:,start:stop]


def native_batch(model,group,holdout,weights,part,total,started,label):
    t=model.t
    w=model.model.lm_head.weight
    contrast=w[weights[0]].float()-w[weights[1]].float().mean(0)
    contrast/=contrast.norm()
    panels=X.panels(model,group,0)
    for p in panels:
        assert len(p['rows'])==8
        assert [r['archive'] for r in p['rows']]==[0,1]*4
        p.update(sites=A.positions(model,p),live={},dose={},receiver={},features={})
    routes={route['layer']:route for route in holdout['routes']}
    early,late=sorted(routes.values(),key=lambda route:route['layer'])
    C.N.progress(label,part*40,total,started)
    for layer in range(40):
        with model.block(layer) as block:
            for p in panels:
                x=p['x'].cuda()
                angles=[a.cuda() for a in p['angles']]
                q,k,v=model.qkv(block,x,angles)
                o=model.attention(block,q,k,v,p['mask'].cuda())
                clean=model.residual(block,x,o)
                for name,(start,stop,arms) in tuple(p['live'].items()):
                    crossing=name==early['site'] and layer==late['layer']
                    changed=t.stack([advance(model,block,p,x,angles,start,stop,a,
                        (late,o,kind) if crossing and kind in ('block_later_head','block_later_gu') else None)
                        for kind,a in zip(ARMS,arms.cuda(),strict=True)])
                    assert t.equal(changed[0],clean[:,start:stop]) and t.equal(changed[5],clean[:,start:stop])
                    p['live'][name]=(start,stop,changed.cpu())
                if layer in routes:
                    r=routes[layer]
                    name=r['site']
                    w=(contrast@block.self_attn.o_proj.weight.float()).view(model.heads,128)
                    read,_=A.source_observation(model,p,q,k,v,w)
                    j=t.arange(len(x),device='cuda')
                    p['features'][name]=read[j,p['sites'][:,r['site_index']],r['head']].cpu().numpy()
                    p['live'][name]=endpoint_arms(model,block,p,x,q,k,v,o,clean,r)
                p['x']=clean.cpu()
        if (layer+1)%5==0:
            C.N.progress(label,part*40+layer+1,total,started)
    observed=[]
    for p in panels:
        x=p['x'].cuda()
        ref=P.full_logits(model,p,x)[:,0]
        forms=[];tops=[];all_kl=[]
        fmt=F.opening_scores(model,ref,weights[:2])
        base=ref[1::2].log_softmax(-1)
        for name in A.SITES:
            start,stop,arms=p['live'][name]
            scores=[];top=[];kl=[]
            for i,branch in enumerate(arms.cuda()):
                current=x.clone()
                current[:,start:stop]=branch
                altered=P.full_logits(model,p,current)[:,0]
                if i in (0,5):
                    assert t.equal(altered,ref)
                assert t.equal(altered[::2],ref[::2]),(name,i)
                y=altered[1::2]
                scores.append(F.opening_scores(model,y,weights[:2]).cpu().numpy())
                top.append(y.argmax(-1).cpu().numpy())
                kl.append((base.exp()*(base-y.log_softmax(-1))).sum(-1).cpu().numpy())
            forms.append(np.stack(scores,1));tops.append(np.stack(top,1));all_kl.append(np.stack(kl,1))
        j=p['rows'][1::2]
        observed.append(dict(ids=np.array([r['id'] for r in j]),family=np.array([r['family'] for r in j]),
            cell=np.array([r['cell'] for r in j]),frame=np.array([r['frame'] for r in j]),bank=np.array([r['bank'] for r in j]),
            base=fmt[1::2].cpu().numpy(),donor=fmt[::2].cpu().numpy(),margin=np.stack(forms,1),top=np.stack(tops,1),
            original=ref.argmax(-1).cpu().numpy().reshape(-1,2),kl=np.stack(all_kl,1),
            features=np.stack([p['features'][site].reshape(-1,2) for site in A.SITES],1),
            dose=np.stack([p['dose'][site] for site in A.SITES],1),
            receiver=np.stack([p['receiver'][site] for site in A.SITES],1)))
    return {key:np.concatenate([r[key] for r in observed]) for key in observed[0]}


def intervene(model,rows,holdout,weights):
    started=time.monotonic()
    # First 8 families in two target cells. Then align remaining packets to full batches.
    groups=[rows[:32],*batched(rows[32:],SLAB)]
    names=[]
    for part,group in enumerate(groups):
        name=f'causal-{part:02}.npz';receipt=f'causal-{part:02}.json'
        if (OUT/receipt).exists():
            row=R.read(receipt)
            assert row['holdout']==sha(RUNNER) and row['sha']==sha(name)
        else:
            check_write(OUT/name,12*2**20,operation='原生逐边反事实写入')
            raw=native_batch(model,group,holdout,weights,part,40*len(groups),started,'archive-native')
            part_sha=A.artifact(name,raw)
            R.write(receipt,dict(sha=part_sha,holdout=sha(RUNNER),views=len(group),at=C.stamp()))
        names.append(name)
    R.write('causal-files.json',dict(names=names,holdout=sha(RUNNER)))


def run():
    import torch

    plan=R.read('design.json');heldout=R.read(RUNNER)
    assert heldout['parent']==sha('design.json') and heldout['candidate']==sha('candidate.json')
    assert heldout['runner']==source_sha() and plan['code']==A.code_identity() and heldout['material']==plan['material']['confirm']
    assert plan['software']==R.versions() and plan['old']==A.bind()
    gate=R.read('behavior.json')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    with C.P.GpuLease(),torch.no_grad():
        start=time.monotonic()
        C.N.progress('archive-holdout-load',0,2,start)
        assert C.Prior.identity()==plan['snapshot']
        model=C.N.Engine()
        C.N.progress('archive-holdout-load',2,2,start)
        try:
            weights=X.anchors(model,plan)
            cohorts=M.material(model.tok,'confirm')
            assert M.digest(cohorts)==heldout['material']['identity']
            if gate['format']['passed']:
                intervene(model,cohorts[2],heldout,weights)
                A.collect(model,cohorts[1],'confirm-cross',plan,weights)
            if all(v['passed'] for v in gate.values()):
                A.collect(model,cohorts[0],'confirm',plan,weights)
        finally:
            model.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','run'))
    args=parser.parse_args()
    A.setup()
    (freeze if args.action=='freeze' else run)()
