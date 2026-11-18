"""Cross two frozen multi-layer routes with reversal and recovery placements."""

import argparse
from itertools import batched
from pathlib import Path
import time

import numpy as np

import paradigm_order_field as A
from deploy.storage_policy import check_write


R,C,M,F,X,P,S,OUT=A.R,A.C,A.M,A.F,A.X,A.P,A.S,A.OUT
PLAN='holdout-plan.json'
SLAB=64
KEYS=('archive','request')
EARLY=('sham','proximal','prox_match','nearby_match',
       'archive_joint','archive_match','archive_scramble',
       'request_joint','request_match','request_scramble','restore')
FORKS=('skip','mlp','at_E','freeze_g','freeze_u','freeze_gu','freeze_random','gu_match','random_match')
ARMS=EARLY+tuple(f'{route}_{name}' for route in KEYS for name in FORKS)
PRECISION=.98


def sha(name):
    return A.sha(OUT/name)


def source_id():
    return A.sha(Path(__file__).resolve())


def freeze():
    assert not (OUT/PLAN).exists()
    d=R.read('design.json');c=R.read('candidate.json');gate=R.read('behavior.json')
    assert gate['passed'] and c['design']==sha('design.json') and c['behavior']==sha('behavior.json')
    assert c['items']==sha('discover-files.json')
    schedules=c['routes']
    assert all(len(r)==3 and r[0]==A.EARLY and r[1][0] in A.MIDDLE and r[2][0] in A.LATE for r in schedules.values())
    R.write(PLAN,dict(at=C.stamp(),code=source_id(),design=sha('design.json'),candidate=sha('candidate.json'),
        material=d['materials']['confirm'],arms=ARMS,first_families=M.CAUSAL,
        protocol=dict(
            operation='At 3 frozen points in each route, replace code-archive residual with same-family same-layout native prose-archive residual. Recompute complete causal suffix, not just one query. Run both routes in all 3 layouts. Original values are reinstalled at all selected points in restore arm. Exact no-op and restoration checked across first 8 family batches and all subsequent rows.',
            geometry='Distance pair (u-s, E-u) determines location u at fixed archive end s and onset E. Primary control is exactly same positions and layers for each row, with a frozen signed coordinate permutation orthogonalized to its native ray before rounding. Secondary control moves the common B0 edit by exactly +1 token to the next downstream position. Keep its native donor, match dose.',
            dose='Each branch measures native size relative to its own current BF16 pre-edit residual at each stage. Min actual displacement .003, relative norm tolerance 3%, cosine >= .90 to each ray. Fixed sample-wise grid; scales chosen without outcome. Location and distance receipts per row; directional alignment of scramble logged. Require >=98% before dose-specific claims.',
            downstream='At B32 E for each raw joint path, propagate either its post-attention residual plus clean MLP, or clean residual plus its own MLP, or its entire E output. All non-E states are restored in these three arms. On joint full suffix restore selected G, U, both, or random G/U; measure equal dose and distance to clean logit.',
            causal_gate='Each edge uses paired archive effect D in 64 families, and signed reductions toward the frozen expected edge (archive move negative, request move positive). A path accounts for an edge only if natural signs hold, raw joint explains >=10% with positive 99% CI and matched beats scramble with 99% CI at >=98% geometry/dose coverage. For shared path same frozen route must pass both edges. Compare own/cross route effects without substituting predictions for interventions.',
            controls='Record exact logit shams, unperturbed prose donor equality, adjacent native match, per-stage dose/gap/cosine, local joint partition and interaction, continued-suffix remainder, G/U equal-dose rescue and matched specificity. Keep reverse and failed-dose rows.'
        )))
    print('Frozen holdout interventions',sha(PLAN),flush=True)


def native_panel(model,p):
    t=model.t
    p['pair']=t.arange(1,len(p['rows']),2,device='cuda')
    p['origin']=t.as_tensor([r['source'][-1] for r in p['rows'][1::2]],device='cuda')
    p['chosen']=t.as_tensor([r['sites'] for r in p['rows'][1::2]],device='cuda')
    p['shifts']={}
    p['doses']={}
    p['gu_doses']={}
    p['gua']={}
    p['lo']=int(p['origin'].min())+1
    p['hi']=int(p['q'][:,0].max())+1
    assert [r['archive'] for r in p['rows']]==[0,1]*4


def position(p,site):
    where=p['origin']+1 if site==-1 else p['chosen'][:,site]
    assert (where>p['origin']).all() and (where<p['q'][p['pair'],0]).all()
    return where


def put(state,j,q,x):
    duplicate=state.clone()
    duplicate[j,q]=x
    return duplicate


def observed(model,base,j,q):
    return base[j-1,q]


def separate_dose(model,base_a,base_b,target_a,target_b):
    t=model.t
    mixed=t.cat((base_a,base_b),1)
    direction_a=target_a.float()-base_a.float()
    direction_b=target_b.float()-base_b.float()
    a,c,stats=P.realized_match(mixed,t.cat((target_a,base_b),1),t.cat((base_a,target_b),1),t,
        (t.cat((direction_a,t.zeros_like(direction_b)),1),t.cat((t.zeros_like(direction_a),direction_b),1)))
    return a[:,:base_a.size(1)],c[:,base_a.size(1):],stats


def at_early(model,p,states,clean):
    t=model.t
    j=p['pair']
    q=position(p,-1)
    neighbor=q+1
    assert (neighbor<p['q'][j,0]).all()
    donor=observed(model,clean,j,q)
    l,r,receipt=separate_dose(model,clean[j,q],clean[j,neighbor],donor,observed(model,clean,j,neighbor))
    states['proximal']=put(clean,j,q,donor)
    states['prox_match']=put(clean,j,q,l)
    states['nearby_match']=put(clean,j,neighbor,r)
    p['prox_dose']=receipt
    p['prox_geometry']=t.stack((q-p['origin'],p['q'][j,0]-q,neighbor-p['origin'],p['q'][j,0]-neighbor),1).cpu().numpy()


def at_route(model,p,states,clean,route,stage,site,signs,order):
    t=model.t
    j=p['pair']
    q=position(p,site)
    a,z,v=(route+part for part in ('_joint','_match','_scramble'))
    donor=observed(model,clean,j,q)
    raw=states[a]
    left=states[z]
    right=states[v]
    base_a=left[j,q];base_b=right[j,q]
    change=donor.float()-base_b.float()
    noise=change[:,order[stage]]*signs[stage]
    norm=change.square().sum(-1,keepdim=True).clamp_min(1e-12)
    noise=noise-change*(noise*change).sum(-1,keepdim=True)/norm
    rotated=(base_b.float()+noise).to(base_b.dtype)
    p['shifts'][route,stage]=t.stack((q,q-p['origin'],p['q'][j,0]-q),1).cpu().numpy()
    x,y,m=separate_dose(model,base_a,base_b,donor,rotated)
    actual=y.float()-base_b.float()
    cosine=(actual*change).sum(-1)/(actual.norm(dim=-1)*change.norm(dim=-1)).clamp_min(1e-12)
    p['doses'][route,stage]=np.column_stack((m,cosine.cpu().numpy()))
    states[a]=put(raw,j,q,donor)
    states[z]=put(left,j,q,x)
    states[v]=put(right,j,q,y)
    # Undo an actual native donor write in-place on the original state before continuation.
    restored=put(states['restore'],j,q,donor)
    restored[j,q]=clean[j,q]
    assert t.equal(restored,clean)
    states['restore']=restored


def variants(model,block,p,base,original,*,changed,heads,route,c):
    t=model.t
    q=p['q'][:,0]
    j=p['pair']
    at=q[j]
    native=P.extract(model,block,changed,heads,q)
    clean_mlp=block.mlp.down_proj(base['a'])[j,at]
    edited_mlp=block.mlp.down_proj(native['a'])[j,at]
    skip=put(original,j,at,native['after'][j,at]+clean_mlp)
    mlp=put(original,j,at,base['after'][j,at]+edited_mlp)
    merged=put(original,j,at,native['output'][j,at])
    selected=c['receivers']; controls=c['random']
    g,u,gu=[P.neuron_restore(model,block,base,native,selected,mode)[0] for mode in ('gate','up','both')]
    random=P.neuron_restore(model,block,base,native,controls,'both')[0]
    b=native['output'][j,at];first=gu[j,at];second=random[j,at]
    left,right,receipt=P.realized_match(b,first,second,t,(first.float()-b.float(),second.float()-b.float()))
    p['gu_doses'][route]=receipt
    p['gua'][route]=np.stack([P.selected_gua(model,a,selected+controls)[1::2] for a in (base,native)],1)
    v=(skip,mlp,merged,g,u,gu,random,put(native['output'],j,at,left),put(native['output'],j,at,right))
    return {f'{route}_{name}':value for name,value in zip(FORKS,v,strict=True)}


def slab(model,rows,c,weights,part,total,started):
    t=model.t
    panels=X.panels(model,rows,0)
    for p in panels:
        native_panel(model,p)
    rng=np.random.default_rng(M.SEED+1)
    masks=t.as_tensor(rng.choice([-1,1],size=(3,model.width)),device='cuda',dtype=t.float32)
    permutations=t.as_tensor(np.stack([rng.permutation(model.width) for _ in range(3)]),device='cuda')
    schedule={}
    for key in KEYS:
        for step,(block,site) in enumerate(c['routes'][key]):
            schedule.setdefault(block,[]).append((key,step,site))
    C.N.progress('order-native',part*40,total,started)
    for i in range(40):
        with model.block(i) as block:
            for p in panels:
                x=p['x'].cuda()
                angles=[k.cuda() for k in p['angles']]
                mask=p['mask'].cuda()
                q,k,v=model.qkv(block,x,angles)
                o=model.attention(block,q,k,v,mask)
                clean=model.residual(block,x,o)
                lo,hi=p['lo'],p['hi']
                if i==0:
                    states=dict.fromkeys(EARLY,clean)
                    at_early(model,p,states,clean)
                else:
                    states={};fork={}
                    if i==S.GULAYER:
                        original=P.extract(model,block,x,o,p['q'][:,0])
                        assert t.equal(original['output'],clean)
                    names,past=p['live']
                    cache=[]
                    for name,a in zip(names,past.cuda(),strict=True):
                        receiver=i==S.GULAYER and name in ('archive_joint','request_joint')
                        shared=next((value for prior,value in cache if t.equal(prior,a)),None) if not receiver else None
                        if shared is None:
                            changed=x.clone()
                            changed[:,lo:hi]=a
                            q1,k1,v1=model.qkv(block,changed,angles)
                            heads=model.attention(block,q1,k1,v1,mask)
                            shared=model.residual(block,changed,heads)
                            cache.append((a,shared))
                        states[name]=shared
                        if receiver:
                            route=name.removesuffix('_joint')
                            fork.update(variants(model,block,p,original,clean,changed=changed,heads=heads,route=route,c=c))
                    states.update(fork)
                for route,j,site in schedule.get(i,()):
                    at_route(model,p,states,clean,route,j,site,masks,permutations)
                assert t.equal(states['sham'],clean) and t.equal(states['restore'],clean)
                names=ARMS if i>=S.GULAYER else EARLY
                p['live']=(names,t.stack([states[key][:,lo:hi] for key in names]).cpu())
                p['x']=clean.cpu()
        if (i+1)%5==0:
            C.N.progress('order-native',part*40+i+1,total,started)
    results=[]
    for p in panels:
        lo,hi=p['lo'],p['hi']
        x=p['x'].cuda()
        logits=P.full_logits(model,p,x)[:,0]
        truth=F.opening_scores(model,logits,weights[:2]).cpu().numpy().reshape(-1,2)
        originals=logits.argmax(-1).cpu().numpy().reshape(-1,2)
        base=logits[1::2].log_softmax(-1)
        margins=[];tops=[];kl=[]
        for name,a in zip(*p['live'],strict=True):
            active=x.clone()
            active[:,lo:hi]=a.cuda()
            edited=P.full_logits(model,p,active)[:,0]
            if name in ('sham','restore'):
                assert t.equal(edited,logits)
            assert t.equal(edited[::2],logits[::2]),name
            after=edited[1::2]
            margins.append(F.opening_scores(model,after,weights[:2]).cpu().numpy())
            tops.append(after.argmax(-1).cpu().numpy())
            kl.append((base.exp()*(base-after.log_softmax(-1))).sum(-1).cpu().numpy())
        item={k:v[1::2] for k,v in S.meta(p).items()}
        def stages(key):
            return np.stack([np.stack([key[r,j] for j in range(3)],1) for r in KEYS],1)
        item.update(natural=truth,original=originals,edit=np.stack(margins,1),top=np.stack(tops,1),kl=np.stack(kl,1),
            stage_dose=stages(p['doses']),geometry=stages(p['shifts']),prox_dose=p['prox_dose'],
            prox_geometry=p['prox_geometry'],gu_dose=np.stack([p['gu_doses'][r] for r in KEYS],1),
            gua=np.stack([p['gua'][r] for r in KEYS],1))
        results.append(item)
    return {key:np.concatenate([a[key] for a in results]) for key in results[0]}


def intervene(model,rows,c,weights,stamp):
    sample=rows[:M.CAUSAL*6]
    assert len(sample)==M.CAUSAL*6 and len({r['family'] for r in sample})==M.CAUSAL
    groups=[sample[:48],*batched(sample[48:],SLAB)]
    names=[];started=time.monotonic()
    for i,group in enumerate(groups):
        data=f'native-{i:02}.npz';receipt=f'native-{i:02}.json'
        if (OUT/receipt).exists():
            saved=R.read(receipt)
            assert saved['holdout']==stamp and saved['sha']==sha(data)
        else:
            check_write(OUT/data,20*2**20,operation='联合阻断逐批预检')
            item=slab(model,group,c,weights,i,40*len(groups),started)
            S.R.write(receipt,dict(sha=S.artifact(data,item),design=sha('design.json'),holdout=stamp,at=C.stamp(),views=len(group)))
        names.append(data)
    S.bind_label('native',names,stamp)


def run():
    import torch

    plan=R.read(PLAN);design=R.read('design.json');c=R.read('candidate.json')
    assert plan['code']==source_id() and plan['candidate']==sha('candidate.json') and plan['design']==sha('design.json')
    assert design['code']==A.identity() and design['software']==R.versions() and design['previous']==A.prior_identity()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    with C.P.GpuLease(),torch.no_grad():
        start=time.monotonic();C.N.progress('order-confirm-load',0,2,start)
        assert C.Prior.identity()==design['model']
        model=C.N.Engine();C.N.progress('order-confirm-load',2,2,start)
        try:
            inputs=M.build(model.tok,'confirm')
            assert M.digest(inputs)==plan['material']['identity']
            weights=X.anchors(model,design)
            # Confirm prediction targets and early field with no intervention before native branches.
            S.collect(model,inputs[0],'confirm',weights,trace=True,binding=sha(PLAN))
            intervene(model,inputs[0],c,weights,sha(PLAN))
        finally:
            model.close()


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','run'))
    args=cli.parse_args()
    A.setup()
    (freeze if args.action=='freeze' else run)()
