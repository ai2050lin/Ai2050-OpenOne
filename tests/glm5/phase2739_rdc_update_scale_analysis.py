"""Whole own-coordinate geometry and all-unit gate/up relations; no axis matching."""
from collections import defaultdict
from itertools import combinations
from rdc_update_common import *


def cosine(v):
    v=v.astype(float);v/=np.linalg.norm(v,axis=-1,keepdims=True).clip(1e-12)
    return (v@v.T).astype(np.float32)


def corr(a,b):
    a=a-a.mean();b=b-b.mean();return float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))


def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(2);start=time.monotonic();out=BASE/'scale_analysis';rows=gzread(BASE/'scale/material.json.gz');models=('qwen4','qwen14','glm4')
    if (out/'result.json').exists():return
    assert all((BASE/'scale'/m/'result.json').exists() for m in models)
    assert (BASE/'scale_batch/qwen4/result.json').exists(),'Matched batch-shape Q4 capture required after large-model batching amendment'
    guard(80*1024**2);allgrams={};summary=[];labels=[]
    for model in models:
        folder=BASE/('scale_batch' if model=='qwen4' else 'scale')/model;spec=read(folder/'runtime.json');features=defaultdict(list);gates=[];ups=[];products=[];source_metrics=[]
        for r in rows:
            with np.load(folder/'fields'/f'{r["sample_id"]}.npz') as z:
                h=unbits(z['H']);src=unbits(z['H_early_sources']);src=src[:int(z['positions'][-1])+1]
                features['embedding'].append(h[0,-1]);features['early_query'].append(h[spec['early'],-1]);features['final_query'].append(h[-1,-1])
                features['source_mean'].append(src.mean(0));features['postnorm'].append(unbits(z['postnorm'])[-1]);features['MLP_activation'].append(unbits(z['activation'])[-1])
                g=unbits(z['gate'])[-1].astype(float);u=unbits(z['up'])[-1].astype(float);a=unbits(z['activation'])[-1].astype(float)
                gates.append(g);ups.append(u);products.append(a)
                reconstructed=g/(1+np.exp(-np.clip(g,-700,700)))*u
                source_metrics.append({'sample_id':r['sample_id'],'native_unit_product_relative_RMS':float(np.linalg.norm(a-reconstructed)/max(np.linalg.norm(a),1e-20)),
                  'visible_sources':len(src),'width':src.shape[-1]})
        packets={k:cosine(np.array(v)) for k,v in features.items()};allgrams[model]=packets
        npz(out/f'{model}_all_sample_cosine.npz',**packets)
        g=np.array(gates);u=np.array(ups);a=np.array(products);stats={}
        for cohort in sorted({r['cohort'] for r in rows}):
            ix=[i for i,r in enumerate(rows) if r['cohort']==cohort];gc=g[ix]-g[ix].mean(0);uc=u[ix]-u[ix].mean(0)
            cov=(gc*uc).mean(0);den=np.sqrt((gc*gc).mean(0)*(uc*uc).mean(0));valid=den>0
            # Zero denominator is explicitly masked; zero placeholder is not a correlation claim.
            c=np.divide(cov,den,out=np.zeros_like(cov),where=valid)
            for name,value in {'gate_mean':g[ix].mean(0),'up_mean':u[ix].mean(0),'activation_mean':a[ix].mean(0),'gate_up_covariance':cov,'gate_up_correlation':c,'correlation_defined':valid}.items():stats[cohort+'__'+name]=value
            summary.append({'model':model,'cohort':cohort,'rows':len(ix),'units':len(c),'defined_correlations':int(valid.sum()),
              'all_unit_mean_abs_correlation':float(np.abs(c[valid]).mean()),'scope':'Same-unit gate/up covariance across material. All units retained; no claim of full pair-of-units covariance or causal independence.'})
        npz(out/f'{model}_all_unit_statistics.npz',**stats)
        save(out/f'{model}_native_product_audit.json',source_metrics)
        labels.append({'model':model,'width':spec['width'],'units':spec['units'],'early':spec['early'],'last':spec['last_block'],
          'field_root':str(folder.relative_to(BASE)),'execution_shape_protocol':spec.get('execution_shape_protocol'),
          'native_B1_shape_comparison':str((folder/'shape_audit/result.json').relative_to(BASE))})
    strata=defaultdict(list)
    for i,r in enumerate(rows):strata[(r['cohort'],r['split'])].append(i)
    pairs=np.triu_indices(len(rows),1)
    pairgroups=defaultdict(list)
    for i,(a,b) in enumerate(zip(*pairs)):pairgroups[tuple(sorted([(rows[a]['cohort'],rows[a]['split']),(rows[b]['cohort'],rows[b]['split'])]))].append(i)
    def residual_vector(gram):
        v=gram[pairs].astype(float).copy()
        for ii in pairgroups.values():v[ii]-=v[ii].mean()
        return v
    rng=np.random.default_rng(273900);perms=[]
    for _ in range(200):
        p=np.arange(len(rows))
        for ii in strata.values():p[ii]=rng.permutation(ii)
        perms.append(p)
    npz(out/'frozen_diagnostic_permutations.npz',permutations=np.array(perms))
    comparisons=[]
    for m,n in combinations(models,2):
      for key in allgrams[m]:
        ga=allgrams[m][key];gb=allgrams[n][key];va=residual_vector(ga);vb=residual_vector(gb)
        null=np.array([corr(va,residual_vector(gb[p][:,p])) for p in perms])
        comparisons.append({'models':[m,n],'feature':key,'all_pairs_correlation':corr(ga[pairs].astype(float),gb[pairs].astype(float)),
          'cohort_pair_centered_correlation':corr(va,vb),'within_cohort_shuffle_mean':float(null.mean()),
          'within_cohort_shuffle_interval95':np.quantile(null,[.025,.975]).tolist(),
          'scope':'Post-outcome descriptive alignment and200within-cohort/split row shuffles, NOT independent significance or fresh predictive confirmation. Repeated semantic expressions/documents remain dependent.'})
    result={'timestamp':stamp(),'source':snapshot(__file__),'samples':128,'sample_order':[r['sample_id'] for r in rows],'models':labels,
      'unit_reports':summary,'cross_model':comparisons,'seconds':time.monotonic()-start,
      'scope':'Complete own-coordinate dot products,128x128sample matrices and every native unit. Coordinates and different depths never identified across models. Same template, tokenizer, training and scale differences remain confounded.'}
    save(out/'result.json',result);ledger('new_three_model_all_coordinate_analysis',result['seconds']);print('SCALE_ANALYSIS_DONE',result['seconds'],flush=True)

if __name__=='__main__':main()
