"""Freeze and map archive-position effects through Qwen3-14B's pre-B32 residual field."""

import argparse
from itertools import batched
from pathlib import Path
import math
import time

import numpy as np

import paradigm_archive_field as A
import paradigm_position_material as M
from paradigm_four_report import behavior
from deploy.storage_policy import check_write, staging_file
from tools.power_analysis import check_phase


R,C,F,X,P=A.R,A.C,A.F,A.X,A.P
OUT=R.ROOT/'tests/glm5/result/paradigm_position_20261118_v1'
TMP=R.ROOT/'tests/glm5_temp'/OUT.name
SLAB=128
DISK=410*2**20
BOUNDARIES=(8,16,24,30)
GULAYER=32
CHANNELS=('input_rms','after_attention_rms','output_rms','output_projection','attention_write_rms','mlp_write_rms')
RAW_THRESHOLD=.0005
LAYOUT_EDGES=((0,1),(0,2),(1,3),(2,3))


def setup():
    R.OUT,R.TEMP=OUT,TMP
    R.M.RUN,R.M.SEED,R.M.WIDTH=M.RUN,M.SEED,M.F.WIDTH
    R.BATCH,R.SLAB=8,SLAB
    R.configure(DISK)


def sha(path):
    return A.sha(path)


def registry():
    own=[Path(__file__).resolve(),Path(M.__file__).resolve()]
    return A.code_identity()|{p.relative_to(R.ROOT).as_posix():sha(p) for p in own}


def bind_previous():
    return {k:sha(A.OUT/k) for k in ('manifest.json','analysis.json','full-field.png','discovery-cross-00.npz')}


def design():
    from transformers import AutoTokenizer
    assert not (OUT/'design.json').exists()
    tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    power=check_phase(.14,128,.20,.01)
    assert power['pass']
    R.write('design.json',dict(at=C.stamp(),run=M.RUN,model=C.Prior.identity(),software=R.versions(),
        provenance=R.metadata(),code=registry(),previous=bind_previous(),materials=M.inventory(tok),
        anchors=F.prior_material.alphabet(tok),power=power,threshold=RAW_THRESHOLD,
        allowed_blocks=BOUNDARIES,sites=M.ANCHORS,site_regions=M.REGION_IDS,gulayer=GULAYER,
        protocol=dict(
          material='Fixed wording, punc, archive and request lexical strings and widths 6/5. Move either active slot by swapping it with its own equal-width inactive marker. Every family crosses the two moves, the two archive formats and targets 1011/1111. Equal IDs outside the specific slot pair. New fact noun/count family held out.',
          behavior='Pilot paired four factors as Phase 2973, all nuisances >=.75; additionally request-style pairing must change print score in all 4 placements x 2 archive values at >=.75 over 16 families. Observed reversal not a gate.',
          fields='At every position through free onset for each discovery style pair, record input/attention/output RMS, output projection and attention/MLP write RMS of code minus prose through B32. Report earliest exact and .0005 RMS robust differences, threshold chosen before seeing model. Retain all 41 BF16 boundaries of one full prefix pair.',
          select='Select a one-move adjacent layout pair preferring opposite discovery means in 1011, then most separated 1011 means (earlier layout as tie-break). Choose one B8/B16/B24/B30 boundary and three distinct regions with best bank-consistent covariance score, downstream of archive and upstream of free onset in both layouts. Match another equal-count same-boundary downstream position set. Select 32 B32 onset G/U indices by signed native A write projection across both targets and all placements with random non-overlapping group.',
          prediction='Fit frozen discovery intercept per cell/layout. Compare this layout-only predictor with ridge over three native residual projections; treat outputs as held-out target. Prior 1011-only pair rule compared to additive and universal pair on first 64 confirm families at a predetermined balanced layout and archive version. Counterexamples retained regardless of significance.',
          method='Counterfactual code recipient uses aligned prose residual of non-source intermediate position(s), followed by full native rerun. Single strongest, each group member, joint, matched alternate positions, original state reinstall, B32 bypass separation and selected G/U counterfactuals, with full masks and exact sham. Native confirmation first 64 families.',
          claims='Use 99% family bootstrap. Sign reversal requires opposite-sign intervals under a single movement. Same-direction effect requires positive intervals at both. Mediation requires ≥10% matched movement closure and positive 99% CI; require both equal-dose coverages >=.98 and compare joint vs control and single. Negative mediation retained; no user format claim beyond free onset.',
          resources='BF16 batch8 width160 streaming CPU/CUDA. D-only checked per slab. Preserve lockbox hash; no model access.'
        )))
    print('Frozen design',sha(OUT/'design.json'),flush=True)


def contrast(model,weights):
    w=model.model.lm_head.weight
    d=w[weights[0]].float()-w[weights[1]].float().mean(0)
    return d/d.norm()


def meta(p):
    return dict(ids=np.array([r['id'] for r in p['rows']]),family=np.array([r['family'] for r in p['rows']]),
        cell=np.array([r['cell'] for r in p['rows']]),layout=np.array([r['layout'] for r in p['rows']]),
        bank=np.array([r['bank'] for r in p['rows']]),archive=np.array([r['archive'] for r in p['rows']]),
        sites=np.array([r['sites'] for r in p['rows']]),query=p['q'].cpu().numpy(),
        source=np.array([r['source'] for r in p['rows']]))


def artifact(name,data):
    budget=sum(np.asarray(x).nbytes for x in data.values())+2**18
    path=check_write(OUT/name,budget,operation='档案位置全场逐批写入')
    with staging_file(path) as temp,temp.open('wb') as stream:
        np.savez_compressed(stream,**data)
    return sha(path)


def bind_label(label,files,binding=None):
    R.write(label+'-files.json',dict(files=files,design=sha(OUT/'design.json'),holdout=binding))


def load(label,binding=None):
    index=R.read(label+'-files.json')
    assert index['design']==sha(OUT/'design.json') and index['holdout']==binding
    values={}
    for name in index['files']:
        record=R.read(Path(name).with_suffix('.json').name)
        assert record['design']==index['design'] and record['holdout']==binding and record['sha']==sha(OUT/name)
        with np.load(OUT/name,allow_pickle=False) as data:
            for k in data.files:
                values.setdefault(k,[]).append(data[k])
    return {k:np.concatenate(v) for k,v in values.items()}


def capture(model,block,p,x,o,d,layer,units,project):
    t=model.t
    query=p['q'][:,0]
    state=P.extract(model,block,x,o,query)
    after=state['after'];end=state['output']
    aw=block.self_attn.o_proj(o.reshape(*x.shape[:-1],model.width).contiguous())
    mw=block.mlp.down_proj(state['a'])
    delta=lambda a:a[1::2].float()-a[::2].float()
    rms=lambda a:delta(a).square().mean(-1).sqrt()
    p['trace'].append(t.stack((rms(x),rms(after),rms(end),delta(end)@d,rms(aw),rms(mw)),-1).cpu().numpy())
    if layer==GULAYER:
        j=t.arange(len(x),device='cuda')
        aa=state['a'][j,query]
        cell=int(p['rows'][0]['cell']==15)
        units[cell]+=(delta(aa).mean(0)*project).cpu().numpy()
    return end


def run_slab(model,group,label,part,total,start,weights,candidate,trace):
    t=model.t
    anchors=weights[:2]
    d=contrast(model,weights)
    panels=X.panels(model,group,0)
    if trace:
        for p in panels:
            p['trace']=[]
        if part==0:
            panels[0]['native']=[]
    units=np.zeros((2,17408),np.float32)
    C.N.progress(label,40*part,total,start)
    for b in range(40):
        with model.block(b) as block:
            project=(d@block.mlp.down_proj.weight.float()) if trace and b==GULAYER else None
            for p in panels:
                x=p['x'].cuda()
                angles=[a.cuda() for a in p['angles']]
                if 'native' in p:
                    p['native'].append(F.base.bit_pattern(x[:2,:int(p['q'][0,0])+1]))
                q,k,v=model.qkv(block,x,angles)
                o=model.attention(block,q,k,v,p['mask'].cuda())
                out=capture(model,block,p,x,o,d,b,units,project) if trace and b<=GULAYER else model.residual(block,x,o)
                if candidate and b==candidate['block']:
                    j=t.arange(len(x),device='cuda')[:,None]
                    position=t.as_tensor([r['sites'] for r in p['rows']],device='cuda')
                    p['project']=(out[j,position].float()@d).cpu().numpy()
                p['x']=out.cpu()
        if (b+1)%5==0:
            C.N.progress(label,40*part+b+1,total,start)
    measured=[];pool=[];pred=[];pred_top=[]
    for p in panels:
        x=p['x'].cuda()
        logits=P.full_logits(model,p,x)
        row=meta(p)
        row.update(margin=F.probe(model,p,logits,weights),top=logits.argmax(-1).cpu().numpy())
        if trace:
            row['trace']=np.stack(p['trace'],1)
            if 'native' in p:
                row['field']=np.stack((*p['native'],F.base.bit_pattern(x[:2,:int(p['q'][0,0])+1])),1)
                row['field_ids']=row['ids'][:2]
        if candidate:
            row['project']=p['project']
        if label=='fusion':
            pool.extend(F.base.bit_pattern(logits[:,0].to(t.bfloat16)))
            if len(pool)==16:
                score,top,_=A.predict(model,pool,p['rows'][0]['archive'],anchors)
                pred.append(score);pred_top.append(top)
                pool=[]
        measured.append(row)
    assert not pool
    data={key:np.concatenate([a[key] for a in measured if key in a]) for key in set().union(*measured)}
    if trace:
        data['units']=units[None]
    if pred:
        assert (data['cell'].reshape(-1,16)==np.arange(16)).all()
        data.update(pred=np.stack(pred),pred_top=np.stack(pred_top))
    return data


def collect(model,rows,label,weights,candidate=None,trace=False,binding=None):
    start=time.monotonic();total=40*math.ceil(len(rows)/SLAB);files=[]
    for part,group in enumerate(batched(rows,SLAB)):
        name=f'{label}-{part:02}.npz'; receipt=f'{label}-{part:02}.json'
        d=sha(OUT/'design.json')
        if (OUT/receipt).exists():
            old=R.read(receipt)
            assert old['design']==d and old['holdout']==binding and old['sha']==sha(OUT/name)
        else:
            check_write(OUT/name,180*2**20 if trace and part==0 else 18*2**20,operation='位置阶段逐批预检')
            data=run_slab(model,group,label,part,total,start,weights,candidate,trace)
            s=artifact(name,data)
            R.write(receipt,dict(sha=s,design=d,holdout=binding,at=C.stamp(),views=len(group)))
        files.append(name)
    bind_label(label,files,binding)


def choose_candidate():
    data=load('discover')
    n=len(data['family'])//2
    t=data['cell'][::2];l=data['layout'][::2];b=data['bank'][::2]
    assert n==32*8 and (data['archive']==np.tile([0,1],n)).all()
    y=data['margin'][1::2,1]-data['margin'][::2,1]
    key=(t==15).astype(int)*4+l
    average=np.array([y[key==k].mean() for k in range(8)])
    def value(edge):
        left,right=edge
        r,s=average[left],average[right]
        return r*s<0,abs(r-s)
    layouts=max(LAYOUT_EDGES,key=value)
    positions=data['sites'][::2]
    archive_end=data['source'][::2,-1]
    relevant=np.isin(l,layouts)
    legal_sites=(positions[relevant]>archive_end[relevant,None]).all(0)
    trace=data['trace']
    j=np.arange(n)[:,None]
    archive_projection=np.stack([trace[:,z,:,3][j,positions] for z in BOUNDARIES],1)
    variants=[]
    y_res=y-average[key]
    for bi,block in enumerate(BOUNDARIES):
        z=archive_projection[:,bi]
        by=np.stack([np.mean((z[b==i]-z[b==i].mean(0))*y_res[b==i,None],axis=0)/
                     np.maximum(z[b==i].std(0),1e-5) for i in (0,1)])
        score=np.minimum(abs(by[0]),abs(by[1]))*((by[0]*by[1])>0)
        order=np.argsort(-score,kind='stable')
        used=set();selected=[]
        for index in order:
            area=M.REGION_IDS[index]
            if legal_sites[index] and area not in used:
                selected.append(int(index));used.add(area)
                if len(selected)==3:break
        assert len(selected)==3
        remainder=[int(i) for i in reversed(order) if legal_sites[i] and i not in selected]
        assert len(remainder)>=3
        variants.append((float(score[selected].sum()),int(block),selected,remainder[:3],score.tolist()))
    quality,block,selected,control,rank=max(variants,key=lambda i:i[0])
    all_features=archive_projection[:,BOUNDARIES.index(block),selected]
    means=np.stack([all_features[key==k].mean(0) for k in range(8)])
    centered=all_features-means[key]
    scales=centered.std(0).clip(1e-5)
    z=centered/scales
    beta=np.linalg.solve(z.T@z+2*np.eye(len(selected)),z.T@y_res)
    score=np.linalg.norm(data['units'].sum(0),axis=0)
    unit_order=np.argsort(-score,kind='stable')
    rand=np.random.default_rng(M.SEED)
    R.write('candidate.json',dict(at=C.stamp(),design=sha(OUT/'design.json'),discovery=sha(OUT/'discover-files.json'),
        behavior=sha(OUT/'behavior.json'),block=block,layouts=layouts,
        layout_move='archive' if layouts[0]^layouts[1]==2 else 'request',
        selected=selected,control=control,rank=rank,rank_total=quality,
        receivers=unit_order[:32].tolist(),random_units=rand.choice(unit_order[64:],32,replace=False).tolist(),
        baseline=average.tolist(),feature_means=means.tolist(),scales=scales.tolist(),beta=beta.tolist()))
    print('Frozen candidate',block,layouts,[M.ANCHORS[i] for i in selected],flush=True)


def run_discovery():
    import torch
    plan=R.read('design.json')
    assert plan['code']==registry() and plan['previous']==bind_previous() and plan['software']==R.versions()
    assert not (OUT/'candidate.json').exists()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    with C.P.GpuLease(),torch.no_grad():
        tic=time.monotonic()
        C.N.progress('position-load',0,2,tic)
        assert C.Prior.identity()==plan['model']
        model=C.N.Engine()
        C.N.progress('position-load',2,2,tic)
        try:
            weights=X.anchors(model,plan)
            pilot=M.material(model.tok,'pilot')
            assert M.digest(pilot)==plan['materials']['pilot']['identity']
            collect(model,pilot[1],'pilot-factor',weights)
            collect(model,pilot[0],'pilot-position',weights)
            factors=behavior(load('pilot-factor'),pilot[1],model.tok)
            scores=load('pilot-position')['margin'][:,1].reshape(16,2,4,2)
            rates=((scores[:,1]-scores[:,0])>0).mean(0)
            gate=dict(at=C.stamp(),factors=factors,position_style=rates.tolist(),
                      passed=bool((rates>=.75).all() and all(x['passed'] for x in factors.values())))
            R.write('behavior.json',gate)
            print('Pilot',gate['passed'],rates.tolist(),flush=True)
            if gate['passed']:
                data=M.material(model.tok,'discovery')
                assert M.digest(data)==plan['materials']['discovery']['identity']
                collect(model,data[0],'discover',weights,trace=True)
                choose_candidate()
        finally:
            model.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','discover'))
    args=parser.parse_args()
    setup()
    (design if args.action=='freeze' else run_discovery)()
