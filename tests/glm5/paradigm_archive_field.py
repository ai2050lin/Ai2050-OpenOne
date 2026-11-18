"""Discover archive paths with native BF16 fields and prospective low-order forecasts."""

import argparse
from itertools import batched, combinations
from pathlib import Path
import math
import time

import numpy as np

import paradigm_archive_material as M
import paradigm_path_run as X
import paradigm_path_core as P
from deploy.storage_policy import check_write, staging_file
from tools.power_analysis import check_phase


R,C,F=X.R,X.C,X.F
OUT=R.ROOT/'tests/glm5/result/paradigm_archive_20261118_v1'
TMP=R.ROOT/'tests/glm5_temp'/OUT.name
DISK=590*2**20
SITES=('onset','grammar')
RULES=('add','pair','target_1011','aligned_1011','margin_4','margin_8')
SLAB=128


def setup():
    R.OUT,R.TEMP=OUT,TMP
    R.M.RUN,R.M.SEED,R.M.WIDTH=M.RUN,M.SEED,M.F.WIDTH
    R.BATCH,R.SLAB=8,SLAB
    R.configure(DISK)


def sha(path):
    return X.digest(path)


def code_identity():
    paths={Path(__file__).resolve(),Path(M.__file__).resolve(),Path(X.__file__).resolve(),
           Path(P.__file__).resolve()}
    return F.hash_sources()|{p.relative_to(R.ROOT).as_posix():sha(p) for p in paths}


def bind():
    old=X.OUT
    return {key:sha(old/key) for key in ('manifest.json','analysis.json','plan.json','confirm-00.npz','legacy-replay.json')}


def freeze():
    from transformers import AutoTokenizer

    assert not (OUT/'design.json').exists()
    tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    power=check_phase(.14,128,.20,.01)
    assert power['pass']
    R.write('design.json',dict(at=C.stamp(),run=M.RUN,snapshot=C.Prior.identity(),provenance=R.metadata(),
        software=R.versions(),code=code_identity(),old=bind(),material=M.inventory(tok),
        power=power,anchors=F.prior_material.alphabet(tok),sites=SITES,rules=RULES,
        design=dict(
            material='Six-token archive. Style, paired payload wording and leading delimiter fully crossed on discovery and first 32 heldout families. Same payload in each code/prose pair. Bank 4 changes print/Report with quotes, parentheses and payload held fixed. Two new confirm frames, three new archive banks.',
            behavior='Same four paired behavior gates as Phase 2972: each of the eight nuisance strata >=.75 in pilot. Absolute readouts and free-onset style tracked separately.',
            field='At every valid position through onset and 40 blocks compute residual read, norm and MLP G/U/A contrasts. Explicit QK and source V/O scalar projection for all heads. Native BF16 full valid prefix at all 41 boundaries of one archive-matched pair; no padding or post-onset forced tokens in that artifact.',
            selection='One head per site, different layers (onset first), ranked on discovery mean absolute across the two target cells of the signed mean archive-swap source write, multiplied by split-bank agreement. 32 signed native MLP writes per site/layer by aggregated absolute downstream contribution, frozen random same-block units. No confirmation adjustment.',
            inference='99% family-cluster bootstrap; validate policy on all 128 holdout families using KL of logits reconstructed from zero, singleton and double conditions only. Compare additive and universal pair; effect intervals and gain exceptions kept. Fit archive-margin prediction from candidate source writes plus target intercept and score against target means. Heldout native effects measured separately on first 64 families.',
            gates='Prediction improvement must have 99% lower bound >0 on both additive and universal pair. Native source requires mean closure >.10 of absolute natural archive contrast and 99% lower bound >0. Equal-norm controls require >=.98 eligible, 3% dose tolerance, 0.90 cosine; local MLP intervention requires a paired selected-vs-control CI and exact shared restoration. Keep effects and failures per route.',
            resources='BF16 CUDA+CPU streamed layers, batch 8, width 160. Caps and writes checked on D each slab. Retain lockbox hash with no model access.'
        )))
    print('Frozen discovery/holdout material',sha(OUT/'design.json'),flush=True)


def artifact(name,arrays):
    size=sum(np.asarray(x).nbytes for x in arrays.values())+2**18
    path=check_write(OUT/name,size,operation='逐层档案场写入')
    with staging_file(path) as temp,temp.open('wb') as stream:
        np.savez_compressed(stream,**arrays)
    return sha(path)


def contents(name):
    with np.load(OUT/name,allow_pickle=False) as data:
        return {k:data[k] for k in data.files}


def positions(model,p):
    return model.t.tensor([[row['query'][0],row['spans']['grammar'][-1]] for row in p['rows']],device='cuda')


def source_observation(model,p,q,k,v,out):
    t=model.t
    mask=p['mask'].cuda()
    sites=t.tensor([r['spans']['archive'] for r in p['rows']],device='cuda')
    n,h,_,d=q.shape
    ix=sites[:,None,:,None].expand(n,h,-1,d)
    k=k.repeat_interleave(model.groups,dim=1)
    logits=(q.float()@k.float().transpose(2,3))*d**-.5
    alpha=logits.masked_fill(~mask,-t.inf).softmax(-1)
    a=alpha.gather(3,sites[:,None,None,:].expand(n,h,q.size(2),-1))
    v=v.repeat_interleave(model.groups,dim=1).gather(2,ix)
    scalar=(v.float()*out[None,:,None]).sum(-1)
    read=(a*scalar[:,:,None]).sum(-1).transpose(1,2)
    return read,a.sum(-1).transpose(1,2)


def traced(model,block,p,x,o,q,k,v,contrast,unit,layer,source_read,project):
    t=model.t
    j=t.arange(len(x),device='cuda')
    sites=p['sites']
    state=P.extract(model,block,x,o,sites[:,0])
    read,mass=source_observation(model,p,q,k,v,source_read)
    after=state['after'];end=state['output']
    residual=t.stack(((x.float()@contrast),x.float().norm(dim=-1),(after.float()@contrast),
                       (end.float()@contrast),end.float().norm(dim=-1)),dim=-1)
    deltas=[(state[key][4:].float()-state[key][:4].float()).abs().mean(-1) for key in ('g','u','a')]
    field=t.cat((residual[4:]-residual[:4],t.stack(deltas,-1)),dim=-1)
    p['trace'].append(field.half().cpu().numpy())
    p['map'].append((read[4:]-read[:4]).mean(0).half().cpu().numpy())
    p['head'].append(t.stack((read[j[:,None],sites],mass[j[:,None],sites]),-1).cpu().numpy())
    a=state['a'][j[:,None],sites].float()
    cell=int(p['rows'][0]['cell']==15)
    unit[cell,layer]+=(a[4:]-a[:4]).mean(0).cpu().numpy()*project
    if 'field' in p:
        p['field'].append(F.base.bit_pattern(x[[0,4],:int(sites[0,0])+1]))
    return end


def policy(target,archive,margin):
    return np.asarray((False,True,target==11,target==11 and archive==0,
                       abs(margin)<=4,abs(margin)<=8),dtype=bool)


def predict(model,parts,archive,anchors):
    t=model.t
    z=t.as_tensor(np.stack(parts),device='cuda').view(t.bfloat16).float()
    base=z[0]
    result=[];tokens=[];margins=[]
    for target in M.TARGETS:
        bits=[v for v in (8,4,2,1) if target&v]
        add=base+(z[bits]-base).sum(0)
        interaction=t.stack([z[i+j]-z[i]-z[j]+base for i,j in combinations(bits,2)]).sum(0)
        pair=add+interaction
        margin=float(F.opening_scores(model,add,anchors))
        use=policy(target,archive,margin)
        candidates=t.stack([pair if u else add for u in use])
        y=z[target].log_softmax(-1)
        kl=(y.exp()*(y-candidates.log_softmax(-1))).sum(-1)
        tops=candidates.argmax(-1)
        result.append(kl.cpu().numpy())
        tokens.append(t.cat((y.argmax().reshape(1),tops)).cpu().numpy())
        margins.append(margin)
    return np.stack(result),np.stack(tokens),np.asarray(margins)


def collect(model,rows,label,plan,weights,trace=False):
    t=model.t
    started=time.monotonic();total=40*math.ceil(len(rows)/SLAB)
    names=[]
    anchors=weights[:2]
    w=model.model.lm_head.weight
    contrast=(w[anchors[0]].float()-w[anchors[1]].float().mean(0))
    contrast/=contrast.norm()
    for part,group in enumerate(batched(rows,SLAB)):
        name=f'{label}-{part:02}.npz'
        receipt=f'{label}-{part:02}.json'
        bound=sha(OUT/'design.json')
        if (OUT/receipt).exists():
            saved=R.read(receipt)
            assert saved['sha']==sha(OUT/name) and saved['design']==bound
            names.append(name)
            continue
        check_write(OUT/name,190*2**20 if trace and part==0 else 16*2**20,operation='模型逐批磁盘复查')
        panels=X.panels(model,group,0)
        if trace:
            for p in panels:
                assert len(p['rows'])==8
                p.update(sites=positions(model,p),trace=[],map=[],head=[])
            units=np.zeros((2,40,2,17408),np.float32)
            if part==0 and label=='discovery-cross':
                panels[0]['field']=[]
        C.N.progress(label,part*40,total,started)
        for layer in range(40):
            with model.block(layer) as block:
                if trace:
                    source_read=(contrast@block.self_attn.o_proj.weight.float()).view(model.heads,128)
                    project=(contrast@block.mlp.down_proj.weight.float()).cpu().numpy()
                for p in panels:
                    inp=p['x'].cuda()
                    q,k,v=model.qkv(block,inp,[a.cuda() for a in p['angles']])
                    o=model.attention(block,q,k,v,p['mask'].cuda())
                    out=traced(model,block,p,inp,o,q,k,v,contrast,units,layer,source_read,project) if trace else model.residual(block,inp,o)
                    p['x']=out.cpu()
            if (layer+1)%5==0:
                C.N.progress(label,part*40+layer+1,total,started)
        scores=[];pred=[];pred_top=[];pred_margin=[];pool=[]
        for p in panels:
            x=p['x'].cuda()
            logits=P.full_logits(model,p,x)
            row=dict(ids=np.asarray([r['id'] for r in p['rows']]),family=np.asarray([r['family'] for r in p['rows']]),
                cell=np.asarray([r['cell'] for r in p['rows']]),frame=np.asarray([r['frame'] for r in p['rows']]),
                bank=np.asarray([r['bank'] for r in p['rows']]),variant=np.asarray([r['variant'] for r in p['rows']]),
                query=p['q'].cpu().numpy(),margin=F.probe(model,p,logits,weights),top=logits.argmax(-1).cpu().numpy())
            if trace:
                row.update(head=np.stack(p['head'],1),map=np.stack(p['map'],0)[None],trace=np.stack(p['trace'],1))
                if 'field' in p:
                    row['native_field']=np.stack((*p['field'],F.base.bit_pattern(x[[0,4],:int(p['sites'][0,0])+1])),1)
                    row['native_field_ids']=row['ids'][[0,4]]
            if label in ('discovery','confirm'):
                pool.extend(F.base.bit_pattern(logits[:,0].to(t.bfloat16)))
                if len(pool)==16:
                    a,b,c=predict(model,pool,p['rows'][0]['archive'],anchors)
                    pred.append(a);pred_top.append(b);pred_margin.append(c)
                    pool=[]
            scores.append(row)
        assert not pool
        data={key:np.concatenate([a[key] for a in scores if key in a]) for key in set().union(*scores)}
        if pred:
            assert (data['cell'].reshape(-1,16)==np.arange(16)).all()
            data.update(pred=np.stack(pred),pred_top=np.stack(pred_top),pred_margin=np.stack(pred_margin))
        if trace:
            data['units']=units[None]
        digest=artifact(name,data)
        R.write(receipt,dict(sha=digest,design=bound,views=len(group),at=C.stamp()))
        names.append(name)
    R.write(label+'-files.json',dict(names=names,design=sha(OUT/'design.json')))


def load(label):
    record=R.read(label+'-files.json')
    assert record['design']==sha(OUT/'design.json')
    collected={}
    for name in record['names']:
        assert sha(OUT/name)==R.read(Path(name).with_suffix('.json').name)['sha']
        for key,value in contents(name).items():
            collected.setdefault(key,[]).append(value)
    return {key:np.concatenate(l) for key,l in collected.items()}


def paired(data,key):
    a=data[key].reshape(-1,8,*data[key].shape[1:])
    return a[:,4:]-a[:,:4]


def choose(plan):
    natural=load('discovery')
    cross=load('discovery-cross')
    assert len(natural['pred'])==M.SIZES['discovery']
    designs=natural['bank'][::16]
    kl=natural['pred'].mean(1)
    losses={name:[float(kl[designs==b,j].mean()) for b in sorted(set(designs))]
            for j,name in enumerate(RULES)}
    rule=min(RULES,key=lambda name:max(losses[name]))
    hs=paired(cross,'head')[...,0]
    target=cross['cell'][::8]
    banks=cross['bank'][::8]
    chosen=[];used=set()
    random=np.random.default_rng(M.SEED)
    unit_scores=cross['units'].sum(0)
    for site in range(2):
        # Match signed responses within both archived wording banks.
        by_bank=np.stack([np.stack([hs[(target==cell)&(banks==b),:,:,site].mean((0,1))
                                    for cell in M.TARGETS]) for b in sorted(set(banks))])
        agree=(by_bank[0]*by_bank[1])>0
        ranking=((((by_bank[0]+by_bank[1])/2)**2*agree).sum(0))**.5
        ranking[:3]=0
        for old in used:
            ranking[max(0,old-2):old+3]=0
        layer,head=np.unravel_index(ranking.argmax(),ranking.shape)
        used.add(layer)
        local=np.linalg.norm(unit_scores[:,layer,site],axis=0)
        order=np.argsort(-local,kind='stable')
        selected=order[:32]
        control=random.choice(order[64:],size=32,replace=False)
        chosen.append(dict(layer=int(layer),head=int(head),site=SITES[site],site_index=site,
                           score=float(ranking[layer,head]),selected=selected.tolist(),control=control.tolist(),
                           observed_bank_targets=by_bank[:,:,layer,head].tolist()))
    features=np.stack([hs[:,:,r['layer'],r['site_index'],r['head']] for r in chosen],-1).reshape(-1,2)
    delta=paired(cross,'margin')[:,:,1].reshape(-1)
    cells=target.repeat(4)
    intercept=np.asarray([delta[cells==i].mean() for i in M.TARGETS])
    means=np.stack([features[cells==i].mean(0) for i in M.TARGETS])
    dummies=(cells==15).astype(int)
    residual=features-means[dummies]
    scale=residual.std(0).clip(1e-5)
    design=residual/scale
    coeff=np.linalg.solve(design.T@design+2*np.eye(2),design.T@(delta-intercept[dummies]))
    data=dict(at=C.stamp(),design=sha(OUT/'design.json'),evidence={label:sha(OUT/(label+'-files.json'))
        for label in ('pilot','discovery','discovery-cross')},rule=rule,rule_bank_losses=losses,
        routes=chosen,forecast=dict(mean=intercept.tolist(),feature_mean=means.tolist(),scale=scale.tolist(),
            coefficients=coeff.tolist()))
    R.write('candidate.json',data)
    print('FROZEN',rule,[(r['layer'],r['head'],r['site']) for r in chosen],flush=True)


def discovery():
    import torch
    from paradigm_four_report import behavior

    plan=R.read('design.json')
    assert plan['code']==code_identity() and plan['old']==bind() and plan['software']==R.versions()
    assert not (OUT/'candidate.json').exists()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    with C.P.GpuLease(),torch.no_grad():
        start=time.monotonic()
        C.N.progress('archive-load',0,2,start)
        assert plan['snapshot']==C.Prior.identity()
        model=C.N.Engine()
        C.N.progress('archive-load',2,2,start)
        try:
            weights=X.anchors(model,plan)
            allrows=M.material(model.tok,'pilot')
            assert M.digest(allrows)==plan['material']['pilot']['identity']
            collect(model,allrows[0],'pilot',plan,weights)
            gate=behavior(load('pilot'),allrows[0],model.tok)
            R.write('behavior.json',gate)
            print('Pilot', {k:v['minimum_stratum'] for k,v in gate.items()},flush=True)
            if gate['format']['passed']:
                allrows=M.material(model.tok,'discovery')
                assert M.digest(allrows)==plan['material']['discovery']['identity']
                collect(model,allrows[0],'discovery',plan,weights)
                collect(model,allrows[1],'discovery-cross',plan,weights,trace=True)
                choose(plan)
        finally:
            model.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','discovery'))
    args=parser.parse_args()
    setup()
    (freeze if args.action=='freeze' else discovery)()
