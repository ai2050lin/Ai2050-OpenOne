"""Family-aware frozen prediction, failures and native intervention scoring."""
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import binomtest, beta, spearmanr
import mlp_condition_design as D
from shared.dist_protocol.mechanism_trace import unbits

def load(split,plan):
    materials=D.families(plan,split);values=[];rows=[];records=[]
    for family in materials:
        path=D.OUT/f'{split}-{family["family"]:04d}.npz';meta=D.READ(path.with_suffix('.json'))
        assert meta['sha256']==D.HASH(path) and meta['plan_sha256']==D.HASH(D.OUT/'prereg.json')
        with np.load(path) as arr:values.append(unbits(arr['selected']))
        rows.extend(family['rows']);records.extend(meta['rows'])
    return rows,records,np.concatenate(values)

def nuisance(rows,vocabulary):
    # The mapping is not available at earlier readouts. No target response enters X.
    out=[]
    for r in rows:
        for pos in r['positions']:
            final=pos==len(r['tokens'])-1
            out.append([1.,r['identity'],r['role'],r['late'],r['mapping'] if final else 0.,float(final),pos/1024,
                *[float(r['tokens'][pos]==t) for t in vocabulary]])
    return np.asarray(out,dtype=np.float64)

def feature(base,source,mode):
    if mode=='baseline':return base
    if mode=='source':return np.column_stack([base,source])
    return np.column_stack([base,source,source[:,None]*base[:,1:6]])

def ridge(x,y):
    mean=x.mean(0);scale=x.std(0);mean[0]=0;scale[scale<1e-7]=1
    x=(x-mean)/scale;penalty=np.eye(x.shape[1]);penalty[0,0]=0
    w=np.linalg.solve(x.T@x+penalty,x.T@y)
    return {'mean':mean.tolist(),'scale':scale.tolist(),'weights':w.tolist()}

def predict(x,fit):return ((x-fit['mean'])/fit['scale'])@fit['weights']

def select(plan):
    assert not (D.OUT/'selection.json').exists() and not list(D.OUT.glob('holdout-[0-9]*.npz'))
    rows,records,data=load('discovery',plan)
    vocab=sorted({r['tokens'][pos] for r in rows for pos in r['positions']})
    counts={v:sum(r['tokens'][p]==v for r in rows for p in r['positions']) for v in vocab}
    vocab=sorted(vocab,key=lambda v:(-counts[v],v))[:64]
    base=nuisance(rows,vocab)
    result={'provenance':plan['provenance'],'plan_sha256':D.HASH(D.OUT/'prereg.json'),'vocabulary':vocab,'units_frozen_from_old_atlas':True}
    for p,(label,src,u,dst,v) in enumerate(D.PAIRS):
        x=data[:,2*p,2].reshape(-1).astype(np.float64);y=data[:,2*p+1,2].reshape(-1).astype(np.float64)
        old=D.ROOT/('tests/glm5/result/mlp_tail_goodfire_v1' if src>=32 else 'tests/glm5/result/mlp_tail_b28_b31_v1')
        oldfield=np.load(old/f'B{src}-A.npy',mmap_mode='r')
        oldx=unbits(oldfield[:16384,u])
        result[label]={'scale':max(float(data[:,2*p,2,-1].std()),.001),
            'rare_abs_threshold':float(np.quantile(abs(oldx),.99)),
            'predictors':{mode:ridge(feature(base,x,mode),y) for mode in ('baseline','source','conditional')},
            'old_source_manifest_sha256':D.HASH(old/'manifest.json')}
    result['discovery_behavior']={f:float(np.mean([np.argmax(s['scores'])==r['gold'] for r,s in zip(rows,records) if r['frame']==f])) for f in {r['frame'] for r in rows}}
    result['behavior_gate_passed']=min(result['discovery_behavior'].values())>=.85
    D.SAVE(D.OUT/'selection.json',result)
    print({'discovery_behavior':result['discovery_behavior'],'gate':result['behavior_gate_passed'],
           'scales':{p:result[p]['scale'] for p,*_ in D.PAIRS}},flush=True)

def paired(a,b,alpha=.005):
    a=np.asarray(a,bool);b=np.asarray(b,bool);plus=int((a&~b).sum());minus=int((b&~a).sum())
    return {'n':len(a),'edit_only':plus,'control_only':minus,'edited_rate':float(a.mean()),'control_rate':float(b.mean()),
       'gain':float(a.mean()-b.mean()),'p':float(binomtest(plus,plus+minus,.5).pvalue) if plus+minus else 1.,'alpha':alpha}

def correlation(x,y):
    if np.std(x)==0 or np.std(y)==0:return None
    return float(np.corrcoef(x,y)[0,1])

def heldout(plan):
    frozen=D.READ(D.OUT/'selection.json');rows,records,data=load('holdout',plan)
    base=nuisance(rows,frozen['vocabulary']);report={};perfamily={};predictions={}
    for p,(label,src,u,dst,v) in enumerate(D.PAIRS):
        x=data[:,2*p,2].reshape(-1).astype(np.float64);y=data[:,2*p+1,2].reshape(-1).astype(np.float64)
        pred={m:predict(feature(base,x,m),fit) for m,fit in frozen[label]['predictors'].items()}
        mse={m:np.square(value-y).reshape(-1,16*6).mean(1) for m,value in pred.items()}
        result={'old_threshold':frozen[label]['rare_abs_threshold'],'pearson':correlation(x,y),
                'spearman':float(spearmanr(x,y).statistic),'winsorized':correlation(np.clip(x,*np.quantile(x,[.01,.99])),np.clip(y,*np.quantile(y,[.01,.99])))}
        for mode in ('source','conditional'):
            wins=mse[mode]<mse['baseline'];ratio=float(mse[mode].mean()/mse['baseline'].mean());pval=float(binomtest(int(wins.sum()),len(wins),.5,alternative='greater').pvalue)
            result[mode]={'family_wins':int(wins.sum()),'families':len(wins),'relative_mse':ratio,'p':pval,
                'passed':bool(wins.mean()>=.65 and pval<=.005 and ratio<=.9)}
        rare=abs(x)>=frozen[label]['rare_abs_threshold'];rare_families=rare.reshape(-1,16*6).any(1)
        result['rare']={'positions':int(rare.sum()),'families':int(rare_families.sum()),'powered':bool(rare_families.sum()>=128),
            'source_abs_max':float(abs(x).max()),'pearson':correlation(x[rare],y[rare]) if rare.any() else None,
            'relative_mse':float(np.square(pred['conditional'][rare]-y[rare]).mean()/np.square(pred['baseline'][rare]-y[rare]).mean()) if rare.any() else None}
        # Report signed factorial contrasts and nuisance sensitivity; these are descriptive.
        final=data[:,2*p:2*p+2,2,-1].reshape(-1,2,2,2,2,2)
        contrasts={}
        for axis,name in enumerate(('identity','role','position','mapping'),1):
            difference=np.take(final,1,axis)-np.take(final,0,axis)
            contrasts[name]={'source_mean':float(difference[...,0].mean()),'target_mean':float(difference[...,1].mean()),
                'source_rms':float(np.sqrt(np.square(difference[...,0]).mean())),'target_rms':float(np.sqrt(np.square(difference[...,1]).mean()))}
        result['factor_contrasts']=contrasts
        report[label]=result;perfamily[label]={m:v.tolist() for m,v in mse.items()};predictions.update({label+'_'+m:v.astype(np.float32) for m,v in pred.items()})
    np.savez_compressed(D.OUT/'heldout-predictions.npz',**predictions)
    behavior={f:float(np.mean([np.argmax(s['scores'])==r['gold'] for r,s in zip(rows,records) if r['frame']==f])) for f in {r['frame'] for r in rows}}
    return report,perfamily,behavior

def report(plan):
    obs,losses,behavior=heldout(plan);family_records=[D.READ(D.OUT/f'causal-{i:04d}.json') for i in range(D.HOLDOUT)]
    for r in family_records:
        assert r['restore_bitwise']
        for label,sha in r['prediction_hashes'].items():assert D.HASH(D.OUT/f'prediction-{r["family"]:04d}-{label}.json')==sha
    curves=[];counter=[]
    for label,*_ in D.PAIRS:
        for dose in D.DOSES:
            pairs=[(r,next(c for c in r['curves'] if c['pair']==label and c['dose']==dose)) for r in family_records]
            valid=[(r,c) for r,c in pairs if c['norm_valid']]
            def switch(r,c,key):return np.argmax(c[key]['scores'])!=np.argmax(r['base_scores'])
            actual=[switch(r,c,'edit') for r,c in valid];control=[switch(r,c,'control') for r,c in valid]
            s=paired(actual,control);s['passed']=bool(s['gain']>=.05 and s['p']<=.005 and min(behavior.values())>=.85)
            observed=np.array([c['target_delta'] for _,c in valid]);predicted=np.array([c['prediction']['target_delta'] for _,c in valid])
            curve={'pair':label,'dose':dose,'n':len(valid),'invalid':len(pairs)-len(valid),'switch':s,
              'target_delta_mean':float(observed.mean()) if len(valid) else None,
              'control_delta_mean':float(np.mean([c['control_target_delta'] for _,c in valid])) if valid else None,
              'target_prediction_mse':float(np.mean((observed-predicted)**2)) if valid else None,
              'target_nochange_mse':float(np.mean(observed**2)) if valid else None,
              'distribution_prediction_KL':float(np.mean([c['edit']['KL_edit_to_prediction'] for _,c in valid])) if valid else None,
              'distribution_nochange_KL':float(np.mean([c['edit']['KL_edit_to_nochange'] for _,c in valid])) if valid else None,
              'full_vocab_changes':sum(c['edit']['top1']!=r['base_top1'] for r,c in valid),
              'strata':{k:{'n':sum(r['baseline_correct'] and (r['low_margin']==low) for r,c in valid),
                         'switches':sum(r['baseline_correct'] and (r['low_margin']==low) and switch(r,c,'edit') for r,c in valid)} for k,low in [('low_margin',True),('ordinary_margin',False)]}}
            if dose==D.PRIMARY:
                resc=[(r,c) for r,c in valid if c['rescue_norm_valid']]
                a=[switch(r,c,'edit') and not switch(r,c,'rescue') for r,c in resc]
                b=[switch(r,c,'edit') and not switch(r,c,'sham') for r,c in resc]
                curve['mediation']=paired(a,b);curve['mediation']['passed']=bool(s['passed'] and curve['mediation']['gain']>=.05 and curve['mediation']['p']<=.005)
                counter.extend({'family':r['family'],'sample_id':r['sample_id'],'pair':label,'reason':'norm mismatch' if not c['norm_valid'] else 'no predicted answer switch',
                                'base_scores':r['base_scores'],'record':c} for r,c in pairs if not c['norm_valid'] or not switch(r,c,'edit'))
            curves.append(curve)
    capraw=D.READ(D.OUT/'capability.json')['rows'];cap={}
    original=np.array([np.argmax(r['scores']['base0'])==r['gold'] for r in capraw])
    for label,*_ in D.PAIRS:
        for sign in (-1,1):
            key=label+str(sign);new=np.array([np.argmax(r['scores'][key])==r['gold'] for r in capraw]);loss=int((original&~new).sum())
            upper=float(beta.ppf(1-.0125,loss+1,len(new)-loss)) if loss<len(new) else 1.
            cap[key]={'n':len(new),'lost':loss,'gained':int((~original&new).sum()),'loss_upper':upper,'passed':upper<=.02}
    materials={r['id']:r for f in D.families(plan,'holdout') for r in f['rows']}
    examples=[]
    for label,*_ in D.PAIRS:
        worst=np.argsort(np.array(losses[label]['conditional'])-np.array(losses[label]['baseline']))[-8:][::-1]
        examples.extend({'family':int(f),'pair':label,'reason':'largest excess prediction MSE versus nuisance baseline',
                         'conditional_mse':losses[label]['conditional'][f],'baseline_mse':losses[label]['baseline'][f],
                         'prompt':D.families(plan,'holdout')[int(f)]['rows'][0]['prompt']} for f in worst)
    record={'schema':'mlp_conditional_report.v1','kind':'real','run_id':D.RUN,'model_snapshot':D.SNAPSHOT,'provenance':plan['provenance'],
      'preregistration':plan,'plan_sha256':D.HASH(D.OUT/'prereg.json'),'predictor_sha256':D.HASH(D.OUT/'frozen-predictor.json'),
      'observations':obs,'behavior_by_frame':behavior,'dose_curve':curves,'capability':cap,'restored_families':len(family_records),
      'counterexamples':counter,'prediction_counterexamples':examples,
      'counterexample_prompts':{r['sample_id']:materials[r['sample_id']]['prompt'] for r in counter},
      'scope':'one physical CUDA BF16 node; English factorial long texts; baseline-conditioned intervention forecast, not independent graph-only next-token prediction; rare stratum requires 128 families'}
    D.SAVE(D.OUT/'report.json',record)
    print({'observations':obs,'behavior':behavior,'capability':cap,'counterexamples':len(counter)},flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['select','report']);args=parser.parse_args()
    globals()[args.stage](D.checked())
