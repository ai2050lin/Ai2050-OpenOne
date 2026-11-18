"""Matched native residual edits and B32 bypass/G-U tests for frozen placement families."""

import argparse
from itertools import batched
from pathlib import Path
import time

import numpy as np

import paradigm_position_field as S
from deploy.storage_policy import check_write


R,C,M,X,F,P,OUT=S.R,S.C,S.M,S.X,S.F,S.P,S.OUT
PLAN='holdout-plan.json'
SLAB=64
EARLY=('sham','single_0','single_1','single_2','joint','alt','joint_match','alt_match','restore')
LATE=('skip_at_E','mlp_at_E','joint_at_E','freeze_g','freeze_u','freeze_gu','freeze_random','gu_match','random_match')
BRANCHES=EARLY+LATE


def source_id():
    return S.sha(Path(__file__).resolve())


def binding():
    return S.sha(OUT/PLAN)


def freeze():
    assert not (OUT/PLAN).exists()
    stage=R.read('design.json');c=R.read('candidate.json');gate=R.read('behavior.json')
    assert gate['passed'] and c['design']==S.sha(OUT/'design.json') and c['behavior']==S.sha(OUT/'behavior.json')
    assert c['discovery']==S.sha(OUT/'discover-files.json') and c['block'] in S.BOUNDARIES
    R.write(PLAN,dict(at=C.stamp(),code=source_id(),design=c['design'],candidate=S.sha(OUT/'candidate.json'),
        materials=stage['materials']['confirm'],arms=BRANCHES,fusion='target_1011',family_count=M.CAUSAL,
        protocol=dict(
          placement='Both crossed moves scored for all 128 confirm families. Native uses the two discovery-frozen layouts differing by one move and first 64 families; code residual is recipient, same-layout prose is donor. Original model tokenized inputs and full BF16 computation preserved.',
          block='At the frozen post-block boundary replace the complete residual at 3 frozen non-archive locations, each separately and jointly, against 3 alternate downstream positions. Original values are actively reinstalled in restore. E and direct archive slots are excluded.',
          matched='Equal norm jointly across six positions, measured after BF16 casting, relative error <=3%, dose >=.003 and cosine >=.9. Report native doses and scale excursions. Receiver equal dose is measured separately at post-B32 E. Use fixed grids, never output.',
          bypass='From frozen joint intervention at B32 E, retain only altered post-attention residual or only altered original B32 MLP write. Counterpart is unedited. All other positions are set to original at B32 in these arms. Joint-E reinitialization is the matched referent for the local split; group joint retains affected other positions.',
          receivers='On the continued joint branch restore selected G, U, G and U, or frozen random G/U to clean values at B32 E. Recompute native SiLU(G)*U and down write. Compare equal norm at E, absolute distance to original and signed direction. Full native norm-matched outputs are retained as separate arms.',
          validation='First 8 holdout families by frozen order run the same arms and exact line-by-line no-op and restoration check before scaling; no exclusions or tuning. Pre-archive identity causal negative control in discovery. Require 99% family bounds and equal-dose coverage. All reversals and 1111 exceptions are retained.'
        )))
    print('Frozen native holdout',binding(),flush=True)


def injected(model,p,base,c):
    t=model.t
    pair=t.arange(1,len(base),2,device='cuda')
    sites=p['sites'][pair]
    good=sites[:,c['selected']]
    other=sites[:,c['control']]
    actual_sources=t.as_tensor([r['source'][-1] for r in p['rows'][1::2]],device='cuda')
    assert (good>actual_sources[:,None]).all() and (other>actual_sources[:,None]).all()
    donor=base[pair-1]
    lo=int(t.cat((good,other),1).min());hi=int(p['q'][:,0].max())+1
    def transplant(which):
        out=base.clone()
        out[pair[:,None],which]=donor[t.arange(len(pair),device='cuda')[:,None],which]
        return out
    singles=[transplant(good[:,i:i+1]) for i in range(good.size(1))]
    joint=transplant(good)
    alt=transplant(other)
    both=t.cat((good,other),1)
    orig=base[pair[:,None],both].flatten(1)
    first=joint[pair[:,None],both].flatten(1)
    second=alt[pair[:,None],both].flatten(1)
    v1,v2,dose=P.realized_match(orig,first,second,t,(first.float()-orig.float(),second.float()-orig.float()))
    def replace(vec):
        updated=base.clone()
        updated[pair[:,None],both]=vec.view(len(pair),both.shape[1],base.shape[-1])
        return updated
    restored=joint.clone()
    restored[pair[:,None],good]=base[pair[:,None],good]
    assert t.equal(restored,base)
    p['early_dose']=dose
    return lo,hi,t.stack([b[:,lo:hi] for b in (base,*singles,joint,alt,replace(v1),replace(v2),restored)]).cpu()


def variants(model,block,p,x,o,original,joint_input,joint_heads,c):
    t=model.t
    pos=p['q'][:,0]
    fixed=P.extract(model,block,x,o,pos)
    altered=P.extract(model,block,joint_input,joint_heads,pos)
    assert t.equal(fixed['output'],original)
    j=t.arange(1,len(x),2,device='cuda')
    q=pos[j]
    # Both terms are native writes. Do not recover them by BF16 subtraction.
    clean_mlp=block.mlp.down_proj(fixed['a'])[j,q]
    joint_mlp=block.mlp.down_proj(altered['a'])[j,q]
    def replace(base,value):
        edited=base.clone()
        edited[j,q]=value
        return edited
    skip=replace(original,altered['after'][j,q]+clean_mlp)
    mlp=replace(original,fixed['after'][j,q]+joint_mlp)
    local=replace(original,altered['output'][j,q])
    selected=c['receivers']
    controls=c['random_units']
    g,u,gu=[P.neuron_restore(model,block,fixed,altered,selected,s)[0] for s in ('gate','up','both')]
    random=P.neuron_restore(model,block,fixed,altered,controls,'both')[0]
    b=altered['output'][j,q];first=gu[j,q];second=random[j,q]
    left,right,dose=P.realized_match(b,first,second,t,(first.float()-b.float(),second.float()-b.float()))
    p['late_dose']=dose
    p['gua']=np.stack([P.selected_gua(model,a,selected+controls)[1::2] for a in (fixed,altered)],1)
    return skip,mlp,local,g,u,gu,random,replace(altered['output'],left),replace(altered['output'],right)


def joined_slab(model,rows,weights,candidate,part,total,start):
    t=model.t
    panels=X.panels(model,rows,0)
    for p in panels:
        assert len(p['rows'])==8 and [r['archive'] for r in p['rows']]==[0,1]*4
        p['sites']=t.as_tensor([r['sites'] for r in p['rows']],device='cuda')
    C.N.progress('position-native',40*part,total,start)
    for b in range(40):
        with model.block(b) as block:
            for p in panels:
                x=p['x'].cuda()
                angles=[a.cuda() for a in p['angles']]
                mask=p['mask'].cuda()
                q,k,v=model.qkv(block,x,angles)
                o=model.attention(block,q,k,v,mask)
                base=model.residual(block,x,o)
                if b==candidate['block']:
                    p['live']=injected(model,p,base,candidate)
                elif b>candidate['block']:
                    lo,hi,arms=p['live']
                    next_state=[];fork=()
                    for i,a in enumerate(arms.cuda()):
                        changed=x.clone()
                        changed[:,lo:hi]=a
                        q1,k1,v1=model.qkv(block,changed,angles)
                        o1=model.attention(block,q1,k1,v1,mask)
                        current=model.residual(block,changed,o1)
                        if i in (0,8):
                            assert t.equal(current,base),(b,i)
                        if i==4 and b==S.GULAYER:
                            fork=variants(model,block,p,x,o,base,changed,o1,candidate)
                            assert t.equal(current[::2],base[::2])
                        next_state.append(current[:,lo:hi])
                    next_state.extend(a[:,lo:hi] for a in fork)
                    p['live']=(lo,hi,t.stack(next_state).cpu())
                p['x']=base.cpu()
        if (b+1)%5==0:
            C.N.progress('position-native',40*part+b+1,total,start)
    results=[]
    for p in panels:
        lo,hi,arms=p['live']
        assert len(arms)==len(BRANCHES)
        x=p['x'].cuda()
        natural=P.full_logits(model,p,x)[:,0]
        natural_m=F.opening_scores(model,natural,weights[:2]).cpu().numpy().reshape(-1,2)
        ref=natural[1::2].log_softmax(-1)
        margins=[];top=[];kl=[]
        for j,a in enumerate(arms.cuda()):
            changed=x.clone()
            changed[:,lo:hi]=a
            got=P.full_logits(model,p,changed)[:,0]
            if j in (0,8):
                assert t.equal(got,natural)
            assert t.equal(got[::2],natural[::2])
            now=got[1::2]
            margins.append(F.opening_scores(model,now,weights[:2]).cpu().numpy())
            top.append(now.argmax(-1).cpu().numpy())
            kl.append((ref.exp()*(ref-now.log_softmax(-1))).sum(-1).cpu().numpy())
        info={k:v[1::2] for k,v in S.meta(p).items()}
        info.update(edit=np.stack(margins,1),tops=np.stack(top,1),kl=np.stack(kl,1),natural=natural_m,
            original=natural.argmax(-1).cpu().numpy().reshape(-1,2),dose=np.stack((p['early_dose'],p['late_dose']),1),
            gua=p['gua'])
        results.append(info)
    return {k:np.concatenate([rec[k] for rec in results]) for k in results[0]}


def native(model,rows,c,weights,stamp):
    ids=set(dict.fromkeys(r['family'] for r in rows[:16*M.CAUSAL]))
    selected=[r for r in rows if r['family'] in ids and r['layout'] in c['layouts']]
    assert len(selected)==8*M.CAUSAL
    groups=[selected[:64],*batched(selected[64:],SLAB)]
    start=time.monotonic();files=[]
    for part,group in enumerate(groups):
        name=f'native-{part:02}.npz';rec=f'native-{part:02}.json'
        if (OUT/rec).exists():
            prior=R.read(rec)
            assert prior['holdout']==stamp and prior['sha']==S.sha(OUT/name)
        else:
            check_write(OUT/name,15*2**20,operation='原生残差旁路分片')
            result=joined_slab(model,group,weights,c,part,40*len(groups),start)
            sha=S.artifact(name,result)
            R.write(rec,dict(sha=sha,holdout=stamp,design=S.sha(OUT/'design.json'),at=C.stamp(),views=len(group)))
        files.append(name)
    S.bind_label('native',files,stamp)


def run():
    import torch
    stage=R.read('design.json');c=R.read('candidate.json');plan=R.read(PLAN)
    assert plan['code']==source_id() and plan['candidate']==S.sha(OUT/'candidate.json')
    assert plan['design']==S.sha(OUT/'design.json') and stage['code']==S.registry()
    assert stage['software']==R.versions() and stage['previous']==S.bind_previous() and R.read('behavior.json')['passed']
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    with C.P.GpuLease(),torch.no_grad():
        start=time.monotonic();C.N.progress('position-holdout-load',0,2,start)
        assert C.Prior.identity()==stage['model']
        model=C.N.Engine();C.N.progress('position-holdout-load',2,2,start)
        try:
            materials=M.material(model.tok,'confirm')
            assert M.digest(materials)==plan['materials']['identity']
            weights=X.anchors(model,stage)
            stamp=binding()
            native(model,materials[0],c,weights,stamp)
            S.collect(model,materials[0],'confirm',weights,candidate=c,binding=stamp)
            S.collect(model,materials[1],'fusion',weights,binding=stamp)
        finally:
            model.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','run'))
    args=parser.parse_args()
    S.setup()
    (freeze if args.action=='freeze' else run)()
