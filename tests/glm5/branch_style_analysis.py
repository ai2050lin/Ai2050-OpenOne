"""Discovery-only selection, frozen paired predictions, and native-unit patches."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import branch_style_joint as C
from shared.dist_protocol.mechanism_trace import unbits,file_sha

GRID=[.1,1.,10.,100.,1000.]
METHODS=['zero','mean','lexical','embedding','additive','shared','interaction','shuffle']

def pairs(rows):
    groups=defaultdict(dict)
    for i,r in enumerate(rows):groups[(r['family'],r['branch'],r['task'])][r['style']]=i
    assert all(set(v)=={0,1} for v in groups.values())
    return [(k,v[0],v[1]) for k,v in sorted(groups.items())]

def ridge_fit(x,y,lam):
    mean=x.mean(0);scale=x.std(0);scale=np.maximum(scale,.01)
    a=(x-mean)/scale;ym=y.mean(0)
    coef=np.linalg.solve(a.T@a+np.eye(a.shape[1])*lam,a.T@(y-ym))
    return {'mean':mean,'scale':scale,'ym':ym,'coef':coef,'lam':lam}

def ridge_predict(m,x):return (x-m['mean'])/m['scale']@m['coef']+m['ym']

def bootstrap_difference(a,b,family,seed=C.SEED):
    names=sorted(set(family));d=np.array([np.mean(a[np.array(family)==f]-b[np.array(family)==f]) for f in names])
    rng=np.random.default_rng(seed);means=d[rng.integers(0,len(d),(4000,len(d)))].mean(1)
    return {'families':len(d),'mean_baseline_minus_method':float(d.mean()),'ci_97p5':np.quantile(means,[.0125,.9875]).tolist()}

def data(part,group):
    rows=C.selected_rows(part);pp=pairs(rows);folder=C.OUT/part
    hs=np.load(folder/'hs.npy',mmap_mode='r');a=np.load(folder/'A.npy',mmap_mode='r')
    yy=[];xx=[];lex=[];meta=[]
    blocks=sorted(set(l for l,u in group));rng=np.random.default_rng(C.SEED+61)
    proj={l:rng.choice([-1.,1.],size=(2560,16)).astype(np.float32)/40 for l in blocks}
    embedding=embedding_means(part,rows)
    ep=np.random.default_rng(C.SEED+62).choice([-1.,1.],size=(2560,16)).astype(np.float32)/40
    emb=[]
    for key,i0,i1 in pp:
        r=rows[i0];f,b,t=key
        y=np.array([float(unbits(a[i1,l,1,u]))-float(unbits(a[i0,l,1,u])) for l,u in group])
        content=np.concatenate([unbits(hs[i0,l,2])@proj[l] for l in blocks])
        cond=np.array([b==0,b==1,t==0,t==1,b==0 and t==0,b==1 and t==0],float)
        # Known token/length information, not the target native response.
        features=np.zeros(2*len(C.NAMES)+2*len(C.COLORS)+6)
        features[[x[0] for x in C.NAMES].index(r['slots']['X'])]=1
        features[len(C.NAMES)+[x[0] for x in C.NAMES].index(r['slots']['Y'])]=1
        j=2*len(C.NAMES)
        features[j+[x[0] for x in C.COLORS].index(r['slots']['C'])]=1
        features[j+len(C.COLORS)+[x[0] for x in C.COLORS].index(r['slots']['D'])]=1
        features[-6:]=[len(rows[i0]['tokens']),len(rows[i1]['tokens']),*rows[i0]['readouts'][:2],t,b]
        xx.append(np.r_[content,cond]);lex.append(np.r_[features,cond]);yy.append(y)
        emb.append(np.r_[embedding[i0]@ep,(embedding[i1]-embedding[i0])@ep,features,cond])
        meta.append({'family':f,'branch':b,'task':t,'i0':i0,'i1':i1})
    return {'y':np.array(yy),'hidden':np.array(xx),'lexical':np.array(lex),'embedding':np.array(emb),'meta':meta,'content_dims':16*len(blocks)}

def embedding_means(part,rows):
    """hs[0] baseline from exact input IDs and frozen embedding rows, no forward."""
    path=C.OUT/part/'input_embedding_means.npy'
    if path.exists():return np.load(path,mmap_mode='r')
    import torch
    from safetensors import safe_open
    key='model.embed_tokens.weight'
    index=C.read(C.MODEL/'model.safetensors.index.json')['weight_map']
    ids=sorted({t for r in rows for t in r['tokens']});lookup={v:i for i,v in enumerate(ids)}
    with safe_open(C.MODEL/index[key],framework='pt',device='cpu') as f:
        vectors=f.get_tensor(key)[torch.tensor(ids)].float().numpy()
    value=np.empty((len(rows),2560),np.float32)
    for i,r in enumerate(rows):value[i]=vectors[[lookup[t] for t in r['tokens']]].mean(0)
    np.save(path,value)
    C.save(path.with_suffix('.json'),{'derived':'Mean of native token embedding rows for actual full prefix, FP32 accumulation; not contextual state.','model_snapshot':C.read(C.OUT/'prereg.json')['model_snapshot'],'inputs_sha256':C.file_sha(C.OUT/'inputs.json'),'array_sha256':file_sha(path)})
    return value

def features(d,name):
    if name in ['zero','mean']:return np.zeros((len(d['y']),1))
    if name=='lexical':return d['lexical']
    if name=='embedding':return d['embedding']
    x=d['hidden'];k=d['content_dims'];q=x[:,:k];c=x[:,k:]
    if name=='additive':return c
    if name=='shared':return x
    if name in ['interaction','shuffle']:
        t=np.array([r['task'] for r in d['meta']])[:,None]
        # Shared content plus task interaction. Branch-2 is never an input label
        # with its own fitted coefficient; its state must transfer.
        return np.c_[x,q*t]
    raise ValueError(name)

def select_group():
    rows=C.selected_rows('discovery');pp=pairs(rows)
    family=sorted({r['family'] for r in rows},key=lambda f:C.digest([f,91]))
    fit=set(family[:len(family)*2//3]);dev=set(family)-fit
    a=np.load(C.OUT/'discovery'/'A.npy',mmap_mode='r')
    byblock=[]
    dense_score=[];dense_effect=[]
    start=time.time()
    for l in range(36):
        scores=[];effects=[]
        for b in [0,1]:
            for t in [0,1]:
                subset=[(f,i0,i1) for (f,bb,tt),i0,i1 in pp if bb==b and tt==t]
                d=np.stack([unbits(a[i1,l,1])-unbits(a[i0,l,1]) for f,i0,i1 in subset])
                raw=np.stack([unbits(a[i0,l,1]) for f,i0,i1 in subset])
                fi=np.array([f in fit for f,i0,i1 in subset]);va=~fi
                m=d[fi].mean(0);v=d[va].mean(0)
                denom=np.maximum(raw[fi].std(0),.02)
                scores.append(np.minimum(abs(m),abs(v))/denom*(m*v>0))
                effects.append(m)
        score=np.min(scores,axis=0)
        dense_score.append(score);dense_effect.append(effects)
        indices=np.argsort(-score)[:8]
        byblock.append({'block':l,'units':indices.tolist(),'scores':score[indices].tolist(),'score':float(score[indices].mean())})
        C.log('discovery_selection',l+1,36,start)
    np.savez_compressed(C.OUT/'discovery_full_coordinate_effects.npz',joint_score=np.array(dense_score),mean_style_difference=np.array(dense_effect),cell_order=np.array(['b0-t0','b0-t1','b1-t0','b1-t1']))
    chosen=sorted(byblock,key=lambda r:-r['score'])[:4]
    group=[(r['block'],u) for r in chosen for u in r['units']]
    return group,fit,dev,byblock

def freeze():
    from threadpoolctl import threadpool_limits
    with threadpool_limits(4):_freeze()

def _freeze():
    assert C.read(C.OUT/'discovery'/'summary.json')['gate_passed'],'expanded discovery behavior failed'
    if (C.OUT/'frozen.json').exists():raise RuntimeError('method already frozen; do not refit')
    group,fit,dev,block_scores=select_group()
    d=data('discovery',group);y=d['y'];meta=d['meta']
    tr=np.array([r['family'] in fit and r['branch']<2 for r in meta]);va=np.array([r['family'] in dev and r['branch']<2 for r in meta]);train=np.array([r['branch']<2 for r in meta])
    scale=np.maximum(y[tr].std(0),.02);yn=y/scale
    models={};cv={};arrays={'target_scale':scale}
    rng=np.random.default_rng(C.SEED+903)
    for name in METHODS:
        x=features(d,name);target=yn.copy()
        if name=='shuffle':
            # Shuffle whole family response bundles, preserving paired branches/tasks.
            fam=sorted({r['family'] for r in meta});perm=dict(zip(fam,rng.permutation(fam)))
            lookup={(r['family'],r['branch'],r['task']):i for i,r in enumerate(meta)}
            for i,r in enumerate(meta):target[i]=yn[lookup[(perm[r['family']],r['branch'],r['task'])]]
        losses=[]
        for lam in GRID:
            m=ridge_fit(x[tr],target[tr],lam);loss=float(np.mean((ridge_predict(m,x[va])-yn[va])**2));losses.append((loss,lam))
        loss,lam=min(losses);m=ridge_fit(x[train],target[train],lam)
        if name=='zero':
            m['coef'][:]=0;m['ym'][:]=0
            loss=float(np.mean(yn[va]**2))
        cv[name]={'validation_nmse':loss,'lambda':lam,'parameters':int(m['coef'].size),'features':x.shape[1]}
        for k,v in m.items():arrays[name+'_'+k]=v
    randoms=[]
    for j in range(2):
        g=[]
        for l in sorted(set(l for l,u in group)):
            banned={u for ll,u in group if ll==l};pool=np.array([u for u in range(9728) if u not in banned])
            g.extend((l,int(u)) for u in rng.choice(pool,8,replace=False))
        randoms.append(g)
    np.savez(C.OUT/'frozen_models.npz',**arrays)
    frozen={'schema':'branch_style_frozen.v1','group':group,'random_groups':randoms,'block_scores':block_scores,'cv':cv,'target_scale':'discovery style-difference SD floor .02','fit_branches':[0,1],'heldout_branch':2,'input_scope':'Paired conditional prediction: observes real chat-state H_l at selected native blocks, not debate state or logits. Lexical baseline uses identities/lengths visible in source. Not embedding-to-output independent prediction.','model_snapshot':C.read(C.OUT/'prereg.json')['model_snapshot'],'inputs_sha256':C.file_sha(C.OUT/'inputs.json'),'model_sha256':C.file_sha(C.OUT/'frozen_models.npz'),'collector_sha256':C.file_sha(C.__file__),'analysis_sha256':C.file_sha(__file__),'opened_confirm':False}
    C.save(C.OUT/'frozen.json',frozen)
    print(json.dumps({'frozen_group':group,'cv':cv},ensure_ascii=False),flush=True)

def evaluate():
    from threadpoolctl import threadpool_limits
    with threadpool_limits(4):_evaluate()

def _evaluate():
    f=C.read(C.OUT/'frozen.json');assert C.file_sha(__file__)==f['analysis_sha256'],'analysis changed after freeze'
    assert C.file_sha(C.__file__)==f['collector_sha256'],'collector changed after freeze'
    d=data('confirm',f['group']);a=np.load(C.OUT/'frozen_models.npz');y=d['y']/a['target_scale'];family=[r['family'] for r in d['meta']]
    report={};losses={};preds={}
    for name in METHODS:
        m={k:a[name+'_'+k] for k in ['mean','scale','ym','coef']};pred=ridge_predict(m,features(d,name));preds[name]=pred
        loss=((pred-y)**2).mean(1);losses[name]=loss
        cells={}
        for b,t in itertools_product():
            idx=np.array([r['branch']==b and r['task']==t for r in d['meta']]);truth=y[idx].reshape(-1);p=pred[idx].reshape(-1)
            centered_y=(y[idx]-y[idx].mean(0)).reshape(-1)
            centered_p=(pred[idx]-pred[idx].mean(0)).reshape(-1)
            corr=float(np.corrcoef(truth,p)[0,1]) if np.std(p)>1e-10 else None
            cc=float(np.corrcoef(centered_y,centered_p)[0,1]) if np.std(centered_p)>1e-10 else None
            cells[f'b{b}-t{t}']={'nmse':float(loss[idx].mean()),'correlation':corr,'content_centered_correlation':cc,'sign_accuracy':float(np.mean(np.sign(truth)==np.sign(p)))}
        report[name]={'nmse':float(loss.mean()),'cells':cells}
    tests={}
    for baseline in ['zero','mean','lexical','embedding','additive','shared','shuffle']:
        tests[baseline]=bootstrap_difference(losses[baseline],losses['interaction'],family)
    ratio=1-report['interaction']['nmse']/max(report['additive']['nmse'],1e-12)
    passed=ratio>=.05 and tests['additive']['ci_97p5'][0]>0 and all(v['correlation']>0 for v in report['interaction']['cells'].values())
    np.savez(C.OUT/'confirm_predictions.npz',truth=y,**preds)
    C.save(C.OUT/'prediction_report.json',{'kind':'real','scope':f['input_scope'],'methods':report,'family_bootstrap':tests,'interaction_reduction_vs_additive':ratio,'primary_prediction_passed':passed,'model_snapshot':f['model_snapshot'],'frozen_sha256':file_sha(C.OUT/'frozen.json'),'confirm_behavior':C.read(C.OUT/'confirm'/'summary.json')['behavior']})
    print(json.dumps({'methods':{k:v['nmse'] for k,v in report.items()},'pass':passed,'ratio':ratio,'bootstrap':tests},ensure_ascii=False),flush=True)

def itertools_product():
    return [(b,t) for b in range(3) for t in range(2)]

def kl(a,b):
    from scipy.special import logsumexp
    la=a-logsumexp(a,axis=-1,keepdims=True);lb=b-logsumexp(b,axis=-1,keepdims=True)
    return (np.exp(la)*(la-lb)).sum(-1)

def intervene():
    import torch
    f=C.read(C.OUT/'frozen.json');assert file_sha(C.__file__)==f['collector_sha256']
    rows=C.selected_rows('confirm');pp=pairs(rows);folder=C.OUT/'confirm'
    arrays=np.load(folder/'A.npy',mmap_mode='r');lp=np.load(folder/'logits.npy',mmap_mode='r')
    families=sorted({r['family'] for r in rows});work=[]
    lookup={(r['family'],r['branch'],r['task'],r['style']):i for i,r in enumerate(rows)}
    for family in families:
        branch=int(C.digest([family,'causal'])[:8],16)%3
        wrong=families[(families.index(family)+1)%len(families)]
        for task in [0,1]:
            for style in [0,1]:
                target=lookup[(family,branch,task,style)];donor=lookup[(family,branch,task,1-style)];bad=lookup[(wrong,branch,task,1-style)]
                work.append({'target':target,'donor':donor,'wrong':bad,'family':family,'branch':branch,'task':task,'style':style})
    path=C.OUT/'causal_rows.jsonl';done=set()
    if path.exists():done={(r['target'],r['condition']) for r in map(json.loads,path.read_text(encoding='utf-8').splitlines())}
    start=time.time();conditions=['base','target','random0','random1','wrong_donor','identity']
    C.log('causal',len(done),len(work)*len(conditions),start)
    with C.gpu_lease(),torch.inference_mode():
        engine=C.Engine(False)
        for condition in conditions:
            todo=[w for w in work if (w['target'],condition) not in done]
            for off in range(0,len(todo),4):
                batch=todo[off:off+4]
                group=f['group'] if condition in ['target','wrong_donor','identity'] else f['random_groups'][int(condition[-1])] if condition.startswith('random') else []
                patch={}
                for l in sorted({l for l,u in group}):
                    units=[u for ll,u in group if ll==l]
                    donor=[w['wrong'] if condition=='wrong_donor' else w['target'] if condition=='identity' else w['donor'] for w in batch]
                    value=np.stack([unbits(arrays[i,l,1,units]) for i in donor])
                    patch[l]=(units,torch.tensor(value,device='cuda'))
                engine.patch=patch if patch else None
                logits,scores,_=engine.run([rows[w['target']] for w in batch],score=True)
                for j,w in enumerate(batch):
                    donor_log=unbits(lp[w['donor']]);base_log=unbits(lp[w['target']])
                    rec={**w,'condition':condition,'kl_donor_to_treated':float(kl(donor_log,logits[j])),'kl_base_to_treated':float(kl(base_log,logits[j])),'correct_margin':float(max(scores[j,:2])-max(scores[j,2:])),'style_margin':float(scores[j,0]-scores[j,1])}
                    with path.open('a',encoding='utf-8') as out:out.write(json.dumps(rec,ensure_ascii=False)+'\n');out.flush()
                    done.add((w['target'],condition))
                if len(done)%32==0 or off+len(batch)==len(todo):C.log('causal_'+condition,len(done),len(work)*len(conditions),start)
    causal_report()

def causal_report():
    rows=[json.loads(s) for s in (C.OUT/'causal_rows.jsonl').read_text(encoding='utf-8').splitlines()]
    table={(r['target'],r['condition']):r for r in rows};conditions=sorted({r['condition'] for r in rows});targets=sorted({r['target'] for r in rows})
    summary={};per={}
    for condition in conditions:
        group=[table[t,condition] for t in targets];base=[table[t,'base'] for t in targets]
        gains=np.array([b['kl_donor_to_treated']-g['kl_donor_to_treated'] for b,g in zip(base,group)])
        per[condition]=gains
        summary[condition]={'mean_kl_improvement_toward_donor':float(gains.mean()),'mean_kl_from_base':float(np.mean([g['kl_base_to_treated'] for g in group])),'content_choice_accuracy':float(np.mean([g['correct_margin']>0 for g in group])),'cells':{f'b{b}-t{t}':{'n':sum(g['branch']==b and g['task']==t for g in group),'mean_gain':float(np.mean([v for g,v in zip(group,gains) if g['branch']==b and g['task']==t]))} for b,t in itertools_product()}}
    family=[table[t,'base']['family'] for t in targets]
    tests={r:bootstrap_difference(per['target'],per[r],family) for r in ['random0','random1','wrong_donor']}
    C.save(C.OUT/'causal_report.json',{'kind':'real','scope':'Opposite-style donor patch at final-prefix native A; full vocabulary distribution and teacher-forced candidate margins. No freely generated translation claim.','conditions':summary,'target_gain_minus_control':tests,'identity_max_kl':max(table[t,'identity']['kl_base_to_treated'] for t in targets),'model_snapshot':C.read(C.OUT/'frozen.json')['model_snapshot'],'frozen_sha256':file_sha(C.OUT/'frozen.json')})
    print(json.dumps({'causal':summary,'tests':tests},ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','evaluate','intervene','causal-report']);a=p.parse_args()
    {'freeze':freeze,'evaluate':evaluate,'intervene':intervene,'causal-report':causal_report}[a.command]()

if __name__=='__main__':main()
