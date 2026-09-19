"""Matched first3step all-unit lag covariance, separating temporal phase means."""
from collections import Counter
from rdc_construction_common import *
from rdc_construction_storage import FIELD_STORE,verify_storage

OUT=BASE/'phase2746/runtime_reuse'


def main():
    start=time.monotonic();verify_storage(2*1024**3)
    protocol={'timestamp':stamp(),'source':snapshot(__file__),
        'question':'Does raw pooled adjacent-step coactivity survive identical3step horizons, whole-unit-vector RMS control, and subtraction of shared output-phase means?',
        'design_status':'Exploratory follow-up after observing full-retention lag correlations; no new independent confirmation.',
        'rows':896,'steps':[0,1,2],'transitions':[[0,1],[1,2]],
        'weights':'Equal source_group within family, equal rows per group, equal two transitions; not treating transitions as independent language samples.',
        'views':['raw','whole_unit_vector_RMS'],
        'identity':'Cov_(i,t)(a_it,b_it)=mean_t Cov_i(a_it,b_it)+Cov_t(mean_i a_it,mean_i b_it). The same decomposition holds for both variances.',
        'defined':'Within-step correlation=mean_t Cov_i(a_t,a_(t+1)) / sqrt(mean_t Var_i(a_t) mean_t Var_i(a_(t+1))). It removes step-specific cohort means, not all identity/lexical/history confounding.',
        'axes':'All36layers, all3unit fields gate/up/product, every9728native unit. No cross-layer functional alignment by index; no top-k or zero-amplitude dismissal.',
        'scope':'These are original native time-series statistics, not semantic percentages or proof that one unit performs the same language operation.'}
    immutable(OUT/'protocol.json',protocol)
    records=gzread(BASE/'phase2746/runtime/records.json.gz');reports=[]
    for family in sorted({r['family'] for r in records}):
        rows=[r for r in records if r['family']==family];docs=Counter(r['source_group'] for r in rows)
        # view, moment, step, block, field, native unit
        mean=np.zeros((2,3,36,3,9728));second=np.zeros_like(mean)
        product=np.zeros((2,2,36,3,9728))
        for r in rows:
            path=BASE/r['field_path'];assert sha(path)==r['field_sha256']
            weight=1/(len(docs)*docs[r['source_group']])
            with np.load(path) as z:a=unbits(z['units'][:3]).astype(float)
            for v in range(2):
                x=a if v==0 else a/np.sqrt(np.mean(a*a,-1,keepdims=True)).clip(1e-12)
                mean[v]+=weight*x;second[v]+=weight*x*x
                product[v]+=weight*x[:-1]*x[1:]
        ma,mb=mean[:,:-1].mean(1),mean[:,1:].mean(1)
        total_cov=product.mean(1)-ma*mb
        within_cov=(product-mean[:,:-1]*mean[:,1:]).mean(1)
        phase_cov=(mean[:,:-1]*mean[:,1:]).mean(1)-ma*mb
        va=(second[:,:-1]-mean[:,:-1]**2).mean(1).clip(0)
        vb=(second[:,1:]-mean[:,1:]**2).mean(1).clip(0)
        total_va=(second[:,:-1].mean(1)-ma**2).clip(0)
        total_vb=(second[:,1:].mean(1)-mb**2).clip(0)
        error=float(abs(total_cov-within_cov-phase_cov).max());assert error<1e-8*max(1.,float(abs(total_cov).max()))
        denom=np.sqrt(va*vb);total_denom=np.sqrt(total_va*total_vb)
        valid=denom>1e-30;total_valid=total_denom>1e-30
        within=np.zeros_like(denom);total=np.zeros_like(denom)
        np.divide(within_cov,denom,out=within,where=valid)
        np.divide(total_cov,total_denom,out=total,where=total_valid)
        assert max(float(abs(within).max()),float(abs(total).max()))<=1+1e-8
        arrays={'step_means':mean,'step_second_moments':second,'adjacent_product_means':product,
            'pooled_covariance':total_cov,'within_step_covariance':within_cov,'between_step_mean_covariance':phase_cov,
            'within_step_correlation':within,'within_step_valid':valid,'pooled_correlation':total,'pooled_valid':total_valid}
        path=FIELD_STORE/'runtime_reuse'/(family+'.npz');verify_storage(sum(a.nbytes for a in arrays.values()))
        npz(path,**arrays)
        summary=[]
        for v,view in enumerate(protocol['views']):
            for block in range(36):
                mask=valid[v,block,2];values=within[v,block,2,mask]
                summary.append({'view':view,'block':block,'valid_units':int(mask.sum()),
                    'median_product_pooled_correlation':float(np.median(total[v,block,2,total_valid[v,block,2]])),
                    'median_product_within_step_correlation':float(np.median(values)),
                    'fraction_product_units_positive_within_step':float(np.mean(values>0)),
                    'unit_mean_pooled_covariance':float(total_cov[v,block,2].mean()),
                    'unit_mean_within_step_covariance':float(within_cov[v,block,2].mean()),
                    'unit_mean_between_step_mean_covariance':float(phase_cov[v,block,2].mean())})
        rec={'timestamp':stamp(),'family':family,'rows':len(rows),'source_groups':len(docs),
            'field_path':path.relative_to(BASE).as_posix(),'field_sha256':sha(path),'field_bytes':path.stat().st_size,
            'covariance_identity_max_error':error,'summaries':summary,'source':snapshot(__file__)}
        save(OUT/'commits'/(family+'.json'),rec);reports.append(rec)
        print('MATCHED_RUNTIME_REUSE',family,round(time.monotonic()-start,1),flush=True)
    result={'timestamp':stamp(),'all_passed':True,'source':snapshot(__file__),'protocol':protocol,'reports':reports,
        'maximum_covariance_identity_error':max(r['covariance_identity_max_error'] for r in reports),
        'seconds':time.monotonic()-start,'scope':protocol['scope']}
    save(OUT/'result.json',result);ledger('phase2746_runtime_reuse',result['seconds'])
    print('MATCHED_RUNTIME_REUSE_COMPLETE',result['seconds'],flush=True)


if __name__=='__main__':
    start=time.monotonic()
    try:main()
    except Exception as exc:
        failure(OUT,start,exc);raise
