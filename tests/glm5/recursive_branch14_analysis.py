"""Discovery-only selection, frozen predictors, heldout branch/chain analysis."""
import argparse
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
from scipy.special import expit
from scipy.stats import t as student

import recursive_branch14 as R


def load(split):
    allrows=R.materials();meta=R.read(R.OUT/split/'collection.json')
    return [allrows[i] for i in meta['row_indices']],meta


def field(split,b,f,readout=0):
    return R.unbits(np.load(R.OUT/split/f'B{b}-{f}.npy',mmap_mode='r')[:,readout])


def target(split,b,units,readout=0):
    return np.stack([R.unbits(np.load(R.OUT/split/f'B{b}-{f}.npy',mmap_mode='r')[:,readout,units]) for f in ['G','V','A']],-1).reshape(-1,len(units)*3)


def semantic(rows):
    # All variables are known before any measured activation. For unseen
    # domains, both training domain dummies are zero; no target-field input.
    return np.array([[r['role']=='operative',r['role']=='quoted',r['lexical'],
        r['domain']=='mathematics',r['domain']=='software',r['metadata']['operation']=='reverse',
        r['role']=='operative' and r['metadata']['operation']=='reverse',
        r['role']=='quoted' and r['metadata']['operation']=='reverse',
        r['role']=='operative' and r['lexical'],r['role']=='quoted' and r['lexical']]
        for r in rows],dtype=np.float64)


def lexical(rows):
    a=np.zeros((len(rows),516),np.float64)
    for i,r in enumerate(rows):
        t=r['tokens'][:r['content_end']+1]
        for v in t:
            k=(v*2654435761)&0xffffffff;a[i,k%512]+=1 if (k//512)%2 else -1
        for x,y in zip(t,t[1:]):
            k=((x*73856093)^(y*19349663))&0xffffffff;a[i,k%512]+=1 if (k//512)%2 else -1
        a[i,:512]/=max(len(t),1)**.5
        a[i,512:]=[len(t),len(t)**2,r['content_start'],r['lexical']]
    return a


def nuisance(rows):
    return np.array([[1,r['content_end'],r['content_end']**2,r['lexical'],r['frame']==1,
        r['domain']=='mathematics',r['metadata']['option_order']] for r in rows],float)


def regression(x,y):
    from sklearn.linear_model import Ridge
    xm=x.mean(0);xs=x.std(0);xs=np.where(xs>1e-8,xs,1.)
    ym=y.mean(0);ys=y.std(0);ys=np.where(ys>1e-8,ys,1.)
    model=Ridge(alpha=1.,fit_intercept=False,solver='cholesky').fit((x-xm)/xs,(y-ym)/ys)
    return dict(xm=xm,xs=xs,ym=ym,ys=ys,coef=model.coef_.T)


def predict(model,x):return ((x-model['xm'])/model['xs']@model['coef'])*model['ys']+model['ym']


def put_model(name,model):np.savez(R.OUT/(name+'.npz'),**model)


def get_model(name):
    with np.load(R.OUT/(name+'.npz')) as f:return {k:f[k] for k in f.files}


def relation_effect(rows,y,residual=True):
    if residual:
        x=nuisance(rows);x[:,1:]=(x[:,1:]-x[:,1:].mean(0))/np.maximum(x[:,1:].std(0),1.)
        y=y-x@np.linalg.lstsq(x,y,rcond=None)[0]
    pos=np.array([r['role']!='inventory' for r in rows]);neg=~pos
    effect=y[pos].mean(0)-y[neg].mean(0)
    return effect/np.maximum(y.std(0),1e-8)


def down_norm(block):
    w=R.C.weight(f'model.layers.{block}.mlp.down_proj.weight');norm=np.empty(R.WIDTH,np.float32)
    for lo in range(0,R.WIDTH,512):norm[lo:lo+512]=w[:,lo:lo+512].float().norm(dim=0).numpy()
    return norm


def matched(units,score,seed):
    rng=np.random.default_rng(seed);out=[];forbidden=set(map(int,units))
    z=np.log(np.maximum(score,1e-10))
    for u in units:
        order=np.argsort(np.abs(z-z[u]));eligible=[int(j) for j in order if int(j) not in forbidden and int(j) not in out][:max(64,len(units)*4)]
        out.append(int(rng.choice(eligible)))
    return out


def select():
    R.admission();assert not (R.OUT/'selection.json').exists(),'Frozen selection exists'
    rows,meta=load('discovery');start=time.monotonic();groups={};backward={}
    assert meta['behavior_passed'],'Discovery behavior gate failed; stop affected internal analysis'
    for h,(b,u) in R.CANDIDATES.items():
        ix=np.array([i for i,r in enumerate(rows) if r['hypothesis']==h]);rr=[rows[i] for i in ix]
        g=field('discovery',b,'G')[ix];a=field('discovery',b,'A')[ix]
        effect=relation_effect(rr,g);order=np.argsort(-effect,kind='stable')
        units=[u]+[int(j) for j in order if j!=u][:7]
        score=a.std(0)*down_norm(b);controls=matched(units,score,R.SEED+b)
        groups[h]=dict(block=b,original_unit=u,units=units,matched_units=controls,
            discovery_G_adjusted_effect=[float(effect[j]) for j in units],
            match_scale=[dict(unit=j,scale=float(score[j]),control=k,control_scale=float(score[k])) for j,k in zip(units,controls)])
        y=target('discovery',b,[u])[ix];lx=lexical(rr);sem=semantic(rr)
        emb=np.asarray(np.load(R.OUT/'discovery/prefix-embedding.npy',mmap_mode='r')[ix],np.float64)
        for method,x in [('lexical',lx),('semantic',sem),('combined',np.column_stack([lx,sem])),('embedding',emb),('position',lx[:,-4:])]:
            put_model(f'fit-{h}-{method}',regression(x,y))
        del g,a,y,emb
        g=field('discovery',39,'G')[ix];e=relation_effect(rr,g)
        roots=np.argsort(-e,kind='stable')[:8].tolist();children=roots;edges=[]
        for parent in (38,37,36):
            yy=target('discovery',parent+1,children)[ix]
            aa=field('discovery',parent,'A')[ix];mu=aa.mean(0);sd=np.maximum(aa.std(0),1e-8)
            yn=(yy-yy.mean(0))/np.maximum(yy.std(0),1e-8)
            corr=((aa-mu)/sd).T@yn/len(ix);rank=np.max(np.abs(corr),axis=1)
            parents=np.argsort(-rank,kind='stable')[:32].tolist();random=matched(parents,sd,R.SEED+parent+len(h))
            prefix=f'chain-{h}-B{parent}'
            for method,x in [('parent',aa[:,parents]),('random',aa[:,random]),('lexical',lx),('combined',np.column_stack([lx,aa[:,parents]]))]:
                put_model(prefix+'-'+method,regression(x,yy))
            edges.append(dict(parent_block=parent,child_block=parent+1,parents=parents,children=children,random_parents=random,
                discovery_max_joint_corr=[float(rank[j]) for j in parents],fit_prefix=prefix))
            children=parents
        backward[h]=dict(roots=roots,edges=edges)
        R.log('discovery_selection',len(groups),2,start,hypothesis=h)
    modelhash={p.name:R.file_sha(p) for p in sorted(R.OUT.glob('fit-*.npz'))+sorted(R.OUT.glob('chain-*.npz'))}
    R.save('selection.json',dict(kind='discovery_frozen',discovery_collection_sha256=R.file_sha(R.OUT/'discovery/collection.json'),
        plan_sha256=R.file_sha(R.OUT/'plan.json'),groups=groups,backward=backward,model_hashes=modelhash,
        discovery_behavior_cells=meta['behavior_cells'],discovery_behavior_passed=meta['behavior_passed'],
        heldout_responses_seen=False,source_sha256=R.file_sha(__file__)))


def inference(values,alpha=.05):
    v=np.asarray(values,float);n=len(v);m=float(v.mean());sd=float(v.std(ddof=1)) if n>1 else 0.
    if sd>0:
        statistic=m/(sd/np.sqrt(n));p=float(2*student.sf(abs(statistic),n-1));half=float(student.ppf(1-alpha/2,n-1)*sd/np.sqrt(n))
        dz=m/sd
    else:p=1. if m==0 else 0.;half=0.;dz=None
    return dict(n_families=n,mean=m,sd=sd,dz=dz,p_two_sided=p,alpha=alpha,ci=[m-half,m+half])


def paired(rows,y,contrast='relation'):
    by=defaultdict(dict)
    for r,val in zip(rows,y):by[r['family']][r['role'],r['lexical']]=val
    values=[]
    for family in sorted(by):
        a=by[family];assert len(a)==6
        if contrast=='relation':v=sum((a['operative',l]+a['quoted',l])/2-a['inventory',l] for l in (0,1))/2
        elif contrast=='operative':v=sum(a['operative',l]-a['quoted',l] for l in (0,1))/2
        elif contrast=='title':v=sum(a[s,1]-a[s,0] for s in ['operative','quoted','inventory'])/3
        elif contrast=='interaction':v=((a['operative',1]+a['quoted',1])/2-a['inventory',1])-((a['operative',0]+a['quoted',0])/2-a['inventory',0])
        elif contrast=='operative_vs_inventory':v=sum(a['operative',l]-a['inventory',l] for l in (0,1))/2
        elif contrast=='quoted_vs_inventory':v=sum(a['quoted',l]-a['inventory',l] for l in (0,1))/2
        values.append(v)
    return np.asarray(values)


def error(y,p,model):
    scaled=((y-p)/model['ys'])**2;meanbase=((y-model['ym'])/model['ys'])**2
    mse=scaled.mean(0);base=meanbase.mean(0)
    return dict(standardized_MSE=float(mse.mean()),mean_baseline_MSE=float(base.mean()),
        ratio_to_mean=float(mse.mean()/max(base.mean(),1e-20)),
        per_field_MSE=mse.reshape(-1,3).mean(0).tolist(),
        per_field_ratio=(mse.reshape(-1,3).mean(0)/np.maximum(base.reshape(-1,3).mean(0),1e-20)).tolist())


def compose(pred):
    z=pred.reshape(len(pred),-1,3).copy();z[:,:,2]=z[:,:,0]*expit(z[:,:,0])*z[:,:,1]
    return z.reshape(len(pred),-1)


def analyze():
    R.admission();selection=R.read(R.OUT/'selection.json');start=time.monotonic()
    for name,h in selection['model_hashes'].items():assert R.file_sha(R.OUT/name)==h,name
    results=[];chains=[];examples=[];behavior=[]
    for split in R.SPLITS[1:]:
        rows,meta=load(split);behavior.extend(meta['behavior_cells'])
        for h,(b,u) in R.CANDIDATES.items():
            gate=all(c['passed'] for c in meta['behavior_cells'] if c['hypothesis']==h)
            if not gate:
                results.append(dict(hypothesis=h,block=b,unit=u,split=split,behavior_passed=False,primary_passed=False,
                    reason='Behavior gate failed; internal mechanism analysis not admitted'))
                chains.append(dict(hypothesis=h,split=split,passed=False,reason='Behavior gate failed'))
                continue
            ix=np.array([i for i,r in enumerate(rows) if r['hypothesis']==h]);rr=[rows[i] for i in ix]
            y=target(split,b,[u])[ix];lx=lexical(rr);sem=semantic(rr)
            emb=np.asarray(np.load(R.OUT/split/'prefix-embedding.npy',mmap_mode='r')[ix],np.float64)
            preds={};errors={};contrasts={}
            for method,x in [('lexical',lx),('semantic',sem),('combined',np.column_stack([lx,sem])),('embedding',emb),('position',lx[:,-4:])]:
                model=get_model(f'fit-{h}-{method}');preds[method]=predict(model,x);errors[method]=error(y,preds[method],model)
                if method in ['semantic','combined']:errors[method+'_composed']=error(y,compose(preds[method]),model)
            for j,f in enumerate(['G','V','A']):
                contrasts[f]={c:inference(paired(rr,y[:,j],c),alpha=.05/6 if (f,c)==('G','relation') else .05) for c in ['relation','operative','title','interaction','operative_vs_inventory','quoted_vs_inventory']}
            primary=contrasts['G']['relation'];gain=errors['combined']['per_field_MSE'][0]/max(errors['lexical']['per_field_MSE'][0],1e-20)
            both=all(contrasts['G'][c]['mean']>0 for c in ['operative_vs_inventory','quoted_vs_inventory'])
            passed=bool(gate and both and primary['mean']>0 and primary['dz'] is not None and primary['dz']>=.35 and primary['p_two_sided']<.05/6 and gain<=.9)
            results.append(dict(hypothesis=h,block=b,unit=u,split=split,rows=len(rr),families=len({r['family'] for r in rr}),
                contrasts=contrasts,prediction=errors,combined_to_lexical_G_MSE=gain,behavior_passed=gate,primary_passed=passed,
                relation_effect_passed=bool(both and primary['mean']>0 and primary['dz'] is not None and primary['dz']>=.35 and primary['p_two_sided']<.05/6),
                quadrant_counts={role:{label:int(sum((np.array([r['role']==role for r in rr]))&
                    ((y[:,0]>=0) if label[0]=='+' else (y[:,0]<0))&((y[:,1]>=0) if label[1]=='+' else (y[:,1]<0))))
                    for label in ['++','+-','-+','--']} for role in ['operative','quoted','inventory']}))
            for role in ['operative','quoted','inventory']:
                use=np.array([i for i,r in enumerate(rr) if r['role']==role]);order=use[np.argsort(y[use,0])]
                for pole,chosen in [('low',order[:3]),('high',order[-3:])]:
                    for j in chosen:examples.append(dict(hypothesis=h,split=split,role=role,pole=pole,values=dict(zip(['G','V','A'],map(float,y[j]))),row=rr[j]))
            chain=selection['backward'][h];edge_results=[]
            for e in chain['edges']:
                yy=target(split,e['child_block'],e['children'])[ix];aa=field(split,e['parent_block'],'A')[ix]
                errs={}
                for method,x in [('parent',aa[:,e['parents']]),('random',aa[:,e['random_parents']]),('lexical',lx),('combined',np.column_stack([lx,aa[:,e['parents']]]))]:
                    mm=get_model(e['fit_prefix']+'-'+method);pred=predict(mm,x);errs[method]=error(yy,pred,mm)
                    if method=='parent':errs['composed']=error(yy,compose(pred),mm)
                edge_results.append(dict(parent_block=e['parent_block'],child_block=e['child_block'],errors=errs))
            # Only B36 A is observed. All successive A values are predictions.
            first=chain['edges'][-1];a=field(split,36,'A')[ix][:,first['parents']]
            for e in reversed(chain['edges']):
                mm=get_model(e['fit_prefix']+'-parent');pred=compose(predict(mm,a));a=pred.reshape(len(ix),-1,3)[:,:,2]
            yy=target(split,39,chain['roots'])[ix];free=error(yy,pred,mm)
            topedge=edge_results[0]['errors'];success=(free['ratio_to_mean']<=.9 and
                free['standardized_MSE']<=.9*topedge['lexical']['standardized_MSE'] and
                all(e['errors']['parent']['ratio_to_mean']<=.9 and e['errors']['parent']['standardized_MSE']<=.9*e['errors']['lexical']['standardized_MSE'] and
                    e['errors']['parent']['standardized_MSE']<=.9*e['errors']['random']['standardized_MSE'] for e in edge_results))
            chains.append(dict(hypothesis=h,split=split,edges=edge_results,free_rollout=free,passed=bool(success and gate)))
            R.log('heldout_analysis',len(results),6,start,hypothesis=h,split=split)
    R.save('response-report.json',dict(kind='real',results=results,backward_chains=chains,behavior=behavior,
        hypothesis_passed={h:all(r['primary_passed'] for r in results if r['hypothesis']==h) for h in R.CANDIDATES},
        backward_passed={h:all(r['passed'] for r in chains if r['hypothesis']==h) for h in R.CANDIDATES},
        formula='G=g*z; V=u*z; A=SiLU(G)V. Primary relation=(operative+quoted)/2-inventory. All family variants remain paired.',
        evidence='held-out response prediction/association; not training-time recursion or causal edges',
        source_sha256=R.file_sha(__file__),selection_sha256=R.file_sha(R.OUT/'selection.json')))
    R.save('counterexamples.json',dict(examples=examples,selection='Each heldout partition/hypothesis/role: 3 lowest and 3 highest G; all were retained regardless of explanation'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['select','analyze']);a=ap.parse_args();R.configure()
    select() if a.stage=='select' else analyze()


if __name__=='__main__':main()
