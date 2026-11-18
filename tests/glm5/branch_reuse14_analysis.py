"""Frozen native-coordinate models; grouped fitting and independent confirmation."""
import argparse
import gc
import time
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

import branch_reuse14 as B
import branch_reuse14_collect as C
import recursive_branch14 as R
from recursive_branch14_analysis import inference, matched

BRANCHES=['number','coordinate','label']
ALPHAS=[1.,10.,100.,1000.]
METHODS=['correlation','parameter']


def load(split):
    meta=B.read(B.OUT/split/'collection.json');allrows=B.read(B.OUT/'materials.json')['rows']
    assert meta['row_ids']==[allrows[i]['id'] for i in meta['row_indices']]
    return [allrows[i] for i in meta['row_indices']],meta


def field(split,block,kind,readout=0,units=None):
    meta=B.read(B.OUT/split/'collection.json');out=[]
    for chunk in meta['chunks']:
        with np.load(B.OUT/split/chunk['name']/f'B{block}.npz') as data:
            a=data[kind][:,readout]
            if units is not None:a=a[:,units]
            out.append(B.unbits(a))
    return np.concatenate(out).astype(np.float64)


def state(split,boundary,readout=0):
    meta=B.read(B.OUT/split/'collection.json');out=[]
    for chunk in meta['chunks']:
        with np.load(B.OUT/split/chunk['name']/'states.npz') as data:
            out.append(B.unbits(data['hidden'][:,boundary,readout]))
    return np.concatenate(out).astype(np.float64)


def pooled(split):
    meta=B.read(B.OUT/split/'collection.json');out=[]
    for chunk in meta['chunks']:
        with np.load(B.OUT/split/chunk['name']/'states.npz') as data:out.append(data['prefix_embedding'])
    return np.concatenate(out).astype(np.float64)


def target(split,block,units,readout=0):
    return np.stack([field(split,block,f,readout,units) for f in 'GVA'],-1).reshape(-1,len(units)*3)


def lexical(rows,readout=0):
    x=np.zeros((len(rows),516))
    for i,r in enumerate(rows):
        tokens=r['tokens'][:r['content_end']+1] if readout==0 else r['tokens']
        for token in tokens:
            h=(token*2654435761)&0xffffffff;x[i,h%512]+=1 if (h//512)%2 else -1
        for a,b in zip(tokens,tokens[1:]):
            h=((a*73856093)^(b*19349663))&0xffffffff;x[i,h%512]+=1 if (h//512)%2 else -1
        x[i,:512]/=len(tokens)**.5
        x[i,512:]=[len(tokens),len(tokens)**2,r['content_start'],r['title']]
    return x


def metadata(rows):
    # Known document role, not an inferred semantic feature or answer label.
    return np.array([[r['role']=='rule',r['title'],r['content_end'],r['content_end']**2] for r in rows],float)


def fold_ids(rows):
    families=sorted({r['family'] for r in rows})
    mapping={f:i%3 for i,f in enumerate(families)}
    return np.array([mapping[r['family']] for r in rows])


def fit(x,y,rows=None,alpha=None):
    x=np.asarray(x,np.float64);y=np.asarray(y,np.float64)
    if y.ndim==1:y=y[:,None]
    scores=[]
    if alpha is None:
        assert rows is not None
        folds=fold_ids(rows)
        for value in ALPHAS:
            losses=[]
            for f in range(3):
                tr=folds!=f;te=~tr;m=fit(x[tr],y[tr],alpha=value)
                losses.append(float(np.mean(((predict(m,x[te])-y[te])/m['ys'])**2)))
            scores.append(float(np.mean(losses)))
        alpha=ALPHAS[int(np.argmin(scores))]
    xm=x.mean(0);xs=x.std(0);xs=np.where(xs>1e-7,xs,1.)
    ym=y.mean(0);ys=y.std(0);ys=np.where(ys>1e-7,ys,1.)
    xx=(x-xm)/xs;yy=(y-ym)/ys
    model=dict(xm=xm,xs=xs,ym=ym,ys=ys,alpha=np.array(alpha),cv_scores=np.array(scores))
    if x.shape[1]>len(x):
        gram=xx@xx.T;gram.flat[::len(gram)+1]+=alpha
        dual=solve(gram,yy,assume_a='pos',check_finite=False)
        if y.shape[1]>len(x):model.update(train=xx,dual=dual)
        else:model['coef']=xx.T@dual
    else:
        gram=xx.T@xx;gram.flat[::len(gram)+1]+=alpha
        model['coef']=solve(gram,xx.T@yy,assume_a='pos',check_finite=False)
    return model


def predict(model,x):
    xx=(x-model['xm'])/model['xs']
    z=xx@model['coef'] if 'coef' in model else (xx@model['train'].T)@model['dual']
    return z*model['ys']+model['ym']


def put(name,model):
    directory=B.OUT/'fits';directory.mkdir(exist_ok=True)
    path=directory/(name+'.npz');assert not path.exists()
    np.savez_compressed(path,**model)


def get(name):
    with np.load(B.OUT/'fits'/(name+'.npz')) as data:return {k:data[k] for k in data.files}


def paired(rows,values,branch=None):
    ix=[i for i,r in enumerate(rows) if branch is None or r['branch']==branch]
    fam=sorted({rows[i]['family'] for i in ix});out=[]
    for family in fam:
        chosen=[i for i in ix if rows[i]['family']==family]
        rule=[i for i in chosen if rows[i]['role']=='rule'];record=[i for i in chosen if rows[i]['role']=='record']
        assert len(rule)==len(record)==2
        out.append(values[rule].mean(0)-values[record].mean(0))
    return np.array(out)


def common_score(rows,values):
    scores=[]
    for branch in ['number','coordinate']:
        delta=paired(rows,values,branch);scores.append(delta.mean(0)/np.maximum(delta.std(0,ddof=1),1e-7))
    return np.minimum(*scores)


def down_norm(block):
    w=R.C.weight(f'model.layers.{block}.mlp.down_proj.weight');out=[]
    for start in range(0,C.WIDTH,512):out.extend(w[:,start:start+512].float().norm(dim=0).numpy())
    return np.asarray(out)


def parameter_score(parent,children,sd,y):
    """Direct residual read/write prior; frozen-RMS coefficient, not full Jacobian."""
    import torch
    torch.set_num_threads(4)
    gamma=R.C.weight(f'model.layers.{parent+1}.post_attention_layernorm.weight').float()
    read=torch.cat([R.C.weight(f'model.layers.{parent+1}.mlp.{proj}.weight')[children].float()*gamma for proj in ['gate_proj','up_proj']])
    scale=np.r_[y.reshape(len(y),-1,3)[:,:,0].std(0),y.reshape(len(y),-1,3)[:,:,1].std(0)]
    read=read/torch.from_numpy(np.maximum(scale,1e-7)).float()[:,None]
    down=R.C.weight(f'model.layers.{parent}.mlp.down_proj.weight');out=[]
    for lo in range(0,C.WIDTH,512):out.extend((read@down[:,lo:lo+512].float()).square().sum(0).sqrt().numpy())
    return np.array(out)*sd


def plan():
    B.configure();C.admit();assert not (B.OUT/'analysis-plan.json').exists()
    B.save('analysis-plan.json',dict(created_local=time.strftime('%Y-%m-%d %H:%M'),
        primary=dict(unit='B19/U64',fields=['V','A'],branches=BRANCHES,n=80,dz_min=.5,alpha=.05/6,
            baseline='Prefix embedding plus position/title versus that same input plus known rule/record label; combined heldout MSE must be <= .9 baseline in each primary field and branch.',
            positive_gate='Every one of the six contrasts has positive mean, dz >= .5, p < .05/6 and prediction gain. Both title strata must have positive differences.'),
        group='B19/U64 plus seven largest minimum paired A dz across number/coordinate discovery; matched units by std(A)*down-column norm. Fixed before confirmation.',
        graphs=dict(roots='Eight B39 units with largest minimum paired A dz across two fitting branches; same native indices at both readouts.',
            candidates=METHODS,edges='B36 A -> B37 G/V/A -> B38 G/V/A -> B39 G/V/A, 32 parents per edge. Correlation uses max absolute standardized correlation to joint child G/V/A. Parameter uses direct residual weight read-write coefficient; omits attention and RMS rank-one derivative.',
            predictors='Standardized ridge; alpha in [1,10,100,1000] chosen by three grouped-family folds on discovery only. Direct A and SiLU(predG)*predV rollout are distinct hypotheses.',
            comparators='Mean, 512-hash uni/bigram+position/title, prefix embedding, three amplitude-matched random32 sets, full source residual state (unequal information/dimension benchmark).',
            observational_gate='Each local edge and branch has MSE <= .9*mean, .9*lexical, .9*mean(random3 losses). Every free root rollout has MSE <= .5*mean, .9*lexical and .9*embedding. Separate readouts and methods, no cherry-picked global pass.',
            output='Given actual B36 parent A at answer position, roll to B39 root A and fit raw h40 reader, then native final RMSNorm and lm_head. Compare full-vocabulary teacher KL to mean h40, prefix embedding and full h37; true B39 roots are a measured oracle. This is late-state conditional fidelity, not text-only atlas fidelity.',
            output_gate='Per-branch mean KL <= .9 times mean h40 and prefix embedding baseline; full h37 benchmark reported separately.',
            intervention_gate='Predicted minus natural root G/V/A effect must reduce squared error by >=10% over zero-effect in each branch; actual standardized RMS effect >=.05 or indeterminate. This is required separately from observation.'),
        intervention=dict(families_per_branch=40,selection='First40 confirm family IDs in each eligible branch, all four variants; 480 rows. No outcome-based sampling.',
            B19='All body positions after title through content end: single64 dose0/2; group8 dose0/2; matched8 dose0/2; realized-norm random direction for single0/group0; baseline and dose1 numerical no-op.',
            B36='Only final answer-prompt position: correlation32 and parameter32 dose0/2, each matched32 dose0/2, baseline and dose1 no-op. Record all B37..39 graph target changes.',
            main_inference='Six B19 primary effects: single0/group0 x three branches, family aggregation; alpha .05/6. Damage alone is not semantic selectivity. Other dose/control and B36 outcome contrasts are secondary; predictive causal criterion is above.',
            control='Native BF16 exact no-op; baseline relativeL2 <= .02 and KL <= .001; norm control all operated positions <=2% realized write norm mismatch, otherwise invalid subset explicitly labeled.',
            limits='40 families/branch has approximately .89 normal-approximation power for dz=.5 at nominal .05, about .70 with six-test correction. Null behavior effects cannot exclude smaller effects or broad capability risks.'),
        statistics='Paired family t intervals. All branches/strata shown, repeated variants not independent n. Family bootstrap 1000 draws for prediction ratios. Numeric thresholds are prospective; no untested mechanism promoted.',
        scope='Three behavioral-eligible branches only. Label is unseen by fitted predictors; number and coordinate confirm have new wording. Sequence failed calibration and cannot support any mechanism verdict. No training-time formation or universal absence claim.',
        trial_plan_sha256=B.file_sha(B.OUT/'trial-plan.json'),source_sha256=B.file_sha(__file__)))


def select():
    B.configure(600*2**20);C.admit();rows,meta=load('discovery');assert meta['behavior']['passed']
    assert (B.OUT/'analysis-plan.json').exists() and not (B.OUT/'frozen-selection.json').exists()
    start=time.monotonic();embedding=pooled('discovery');lex={r:lexical(rows,r) for r in [0,1]};sem=metadata(rows)
    with threadpool_limits(limits=4):
      for block,unit,name in [(19,64,'primary'),(26,13099,'secondary')]:
        y=target('discovery',block,[unit]);base=np.column_stack([embedding,sem[:,1:]])
        for key,x in [('lexical',lex[0]),('embedding',base),('combined',np.column_stack([base,sem[:,0]])),('position',sem[:,1:])]:put(name+'-'+key,fit(x,y,rows))
      a=field('discovery',19,'A');score=common_score(rows,a);rank=np.argsort(-score,kind='stable')
      units=[64]+[int(j) for j in rank if j!=64][:7];amp=a.std(0)*down_norm(19)
      group=dict(block=19,units=units,matched=matched(units,amp,B.SEED+19),scores=[float(score[j]) for j in units])
      del a
      a=field('discovery',39,'A');score=common_score(rows,a);roots=np.argsort(-score,kind='stable')[:8].tolist();del a
      graphs={}
      for method in METHODS:
        children=roots;edges=[]
        for parent in [38,37,36]:
          a=field('discovery',parent,'A');y=target('discovery',parent+1,children);sd=np.maximum(a.std(0),1e-7)
          if method=='correlation':
            corr=((a-a.mean(0))/sd).T@((y-y.mean(0))/np.maximum(y.std(0),1e-7))/len(a)
            score=np.max(np.abs(corr),axis=1)
          else:score=parameter_score(parent,children,sd,y)
          parents=np.argsort(-score,kind='stable')[:32].tolist();amp=sd*down_norm(parent)
          random=[matched(parents,amp,B.SEED+parent+j*1000) for j in range(3)]
          edge=dict(parent=parent,child=parent+1,parents=parents,children=children,random=random,
                    rank_scores=[float(score[j]) for j in parents])
          for readout in [0,1]:
            if readout==1:a=field('discovery',parent,'A',1)
            y=target('discovery',parent+1,children,readout);prefix=f'{method}-B{parent}-r{readout}'
            predictors={'parent':a[:,parents],'lexical':lex[readout],'embedding':embedding,
                        **{f'random{j}':a[:,p] for j,p in enumerate(random)}}
            for key,x in predictors.items():put(prefix+'-'+key,fit(x,y,rows))
            hs=state('discovery',parent+1,readout);put(prefix+'-residual',fit(hs,y,rows));del hs
          edges.append(edge);children=parents;del a,y;gc.collect()
          B.log('discovery_graph_models',len(graphs)*3+len(edges),6,start,method=method,parent=parent)
        graphs[method]=dict(roots=roots,edges=edges)
      # Common decoder, and inputs available at the same B36 ending boundary.
      y=state('discovery',40,1);rootA=field('discovery',39,'A',1,roots)
      put('output-root',fit(rootA,y,rows));put('output-embedding',fit(embedding,y,rows))
      put('output-lexical',fit(lex[1],y,rows));put('output-residual',fit(state('discovery',37,1),y,rows))
      np.savez_compressed(B.OUT/'output-mean.npz',value=y.mean(0))
    B.save('frozen-selection.json',dict(kind='discovery_only_frozen',group=group,graphs=graphs,
        source_sha256=B.file_sha(__file__),analysis_plan_sha256=B.file_sha(B.OUT/'analysis-plan.json'),
        trial_plan_sha256=B.file_sha(B.OUT/'trial-plan.json'),discovery_sha256=B.file_sha(B.OUT/'discovery/collection.json'),
        models={p.name:B.file_sha(p) for p in sorted((B.OUT/'fits').glob('*.npz'))},
        output_mean_sha256=B.file_sha(B.OUT/'output-mean.npz'),heldout_responses_seen=False,
        limitations='Group/roots selected for shared rule-record response; this does not control all wording or establish a semantic concept. Parameter-prior ranks omit attention and normalization-state dependence.'))


def admit():
    C.admit();selection=B.read(B.OUT/'frozen-selection.json')
    assert selection['source_sha256']==B.file_sha(__file__)
    assert selection['analysis_plan_sha256']==B.file_sha(B.OUT/'analysis-plan.json')
    for name,sha in selection['models'].items():assert sha==B.file_sha(B.OUT/'fits'/name)
    return selection


def row_loss(y,pred,scale):return np.mean(((y-pred)/scale)**2,axis=1)


def ratio_summary(rows,loss,base):
    families=sorted({r['family'] for r in rows})
    a=np.array([loss[[i for i,r in enumerate(rows) if r['family']==f]].mean() for f in families])
    b=np.array([base[[i for i,r in enumerate(rows) if r['family']==f]].mean() for f in families])
    rng=np.random.default_rng(B.SEED);ix=rng.integers(0,len(a),(1000,len(a)))
    ratios=a[ix].mean(1)/np.maximum(b[ix].mean(1),1e-14)
    return dict(ratio=float(a.mean()/max(b.mean(),1e-14)),ci95=np.quantile(ratios,[.025,.975]).tolist(),n_families=len(families))


def rollout(graph,method,readout,initial=None,split='confirm',compose=False):
    current=field(split,36,'A',readout,graph['edges'][-1]['parents']) if initial is None else initial
    for edge in reversed(graph['edges']):
        model=get(f'{method}-B{edge["parent"]}-r{readout}-parent');pred=predict(model,current)
        shaped=pred.reshape(len(pred),-1,3)
        current=shaped[:,:,0]*expit(shaped[:,:,0])*shaped[:,:,1] if compose else shaped[:,:,2]
        if compose:shaped[:,:,2]=current
    return pred


def analyze():
    B.configure();selection=admit();rows,meta=load('confirm');assert meta['behavior']['passed']
    start=time.monotonic();embedding=pooled('confirm');sem=metadata(rows);base=np.column_stack([embedding,sem[:,1:]])
    responses=[];edges_out=[];rolls=[];predictions={}
    with threadpool_limits(limits=4):
      for block,unit,name in [(19,64,'primary'),(26,13099,'secondary')]:
        y=target('confirm',block,[unit]);models={k:get(name+'-'+k) for k in ['lexical','embedding','combined','position']}
        pred={k:predict(m,{'lexical':lexical(rows),'embedding':base,'combined':np.column_stack([base,sem[:,0]]),'position':sem[:,1:]}[k]) for k,m in models.items()}
        for branch in BRANCHES:
          ix=np.array([i for i,r in enumerate(rows) if r['branch']==branch]);rr=[rows[i] for i in ix]
          for f,field_name in enumerate('GVA'):
            delta=paired(rr,y[ix,f]);primary=name=='primary' and field_name in 'VA';stat=inference(delta,.05/6 if primary else .05)
            err={key:float(np.mean((pp[ix,f]-y[ix,f])**2)) for key,pp in pred.items()}
            strata=[float(np.mean(y[[i for i in ix if rows[i]['title']==t and rows[i]['role']=='rule'],f])-np.mean(y[[i for i in ix if rows[i]['title']==t and rows[i]['role']=='record'],f])) for t in [0,1]]
            passed=primary and stat['mean']>0 and stat['dz'] is not None and stat['dz']>=.5 and stat['p_two_sided']<.05/6 and min(strata)>0 and err['combined']<=.9*err['embedding']
            responses.append(dict(candidate=name,block=block,unit=unit,branch=branch,field=field_name,primary=primary,
                contrast=stat,title_differences=strata,MSE=err,combined_over_embedding=err['combined']/max(err['embedding'],1e-14),passed=bool(passed)))
      for method,graph in selection['graphs'].items():
       for readout in [0,1]:
        for edge in graph['edges']:
          parent=edge['parent'];y=target('confirm',parent+1,edge['children'],readout);a=field('confirm',parent,'A',readout)
          prefix=f'{method}-B{parent}-r{readout}';model=get(prefix+'-parent');scale=model['ys']
          pred={'parent':predict(model,a[:,edge['parents']])}
          for key,x in [('lexical',lexical(rows,readout)),('embedding',embedding),('residual',state('confirm',parent+1,readout))]+[(f'random{j}',a[:,p]) for j,p in enumerate(edge['random'])]:pred[key]=predict(get(prefix+'-'+key),x)
          losses={k:row_loss(y,v,scale) for k,v in pred.items()};losses['mean']=row_loss(y,model['ym'],scale)
          losses['random_average']=np.mean(np.stack([losses[f'random{j}'] for j in range(3)]),axis=0)
          for branch in BRANCHES:
            ix=np.array([i for i,r in enumerate(rows) if r['branch']==branch]);rr=[rows[i] for i in ix]
            ratios={key:ratio_summary(rr,losses['parent'][ix],v[ix]) for key,v in losses.items() if key!='parent'}
            passed=all(ratios[key]['ratio']<=.9 for key in ['mean','lexical','random_average'])
            edges_out.append(dict(method=method,readout=readout,parent=parent,branch=branch,ratios=ratios,passed=passed))
          del a;gc.collect()
        rootmodel=get(f'{method}-B38-r{readout}-parent');y=target('confirm',39,graph['roots'],readout)
        baseline={key:predict(get(f'{method}-B38-r{readout}-'+key),x) for key,x in [('lexical',lexical(rows,readout)),('embedding',embedding)]}
        baseline['mean']=np.broadcast_to(rootmodel['ym'],y.shape)
        for mode in ['direct','composed']:
          pred=rollout(graph,method,readout,compose=mode=='composed');predictions[f'{method}-r{readout}-{mode}']=pred
          loss=row_loss(y,pred,rootmodel['ys']);bl={key:row_loss(y,p,rootmodel['ys']) for key,p in baseline.items()}
          for branch in BRANCHES:
            ix=np.array([i for i,r in enumerate(rows) if r['branch']==branch]);rr=[rows[i] for i in ix]
            ratios={key:ratio_summary(rr,loss[ix],v[ix]) for key,v in bl.items()}
            passed=ratios['mean']['ratio']<=.5 and all(ratios[k]['ratio']<=.9 for k in ['lexical','embedding'])
            rolls.append(dict(method=method,readout=readout,mode=mode,branch=branch,ratios=ratios,passed=passed))
        B.log('confirm_graph_analysis',len(rolls),24,start,method=method,readout=readout)
    np.savez_compressed(B.OUT/'confirm-predictions.npz',**predictions)
    B.save('response-and-graph-report.json',dict(kind='real',responses=responses,edges=edges_out,rollouts=rolls,
        primary_passed=all(r['passed'] for r in responses if r['primary']),
        predictions_sha256=B.file_sha(B.OUT/'confirm-predictions.npz'),selection_sha256=B.file_sha(B.OUT/'frozen-selection.json'),
        source_sha256=B.file_sha(__file__),inference='Family aggregation and percentile intervals; material families reuse words/frames, so this does not establish independent natural-language population sampling.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['plan','select','analyze']);args=p.parse_args()
    globals()[args.stage]()
