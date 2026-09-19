"""Stronger effect baselines and exact full-gradient cross-expression distances."""
from rdc_update_common import *

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=2);out=BASE/'learning';start=time.monotonic()
    if (out/'forecast_audit.json').exists():return
    rows=gzread(BASE/'program_material.json.gz');frozen=read(out/'frozen.json')
    with np.load(out/'oracle_derivatives.npz') as z:y=z['derivatives'].astype(float)
    with np.load(out/'frozen_forecast.npz') as z:prediction=z['derivatives'].astype(float)
    train=[i for i,r in enumerate(rows) if r['split']=='train'];mean=y[train].mean(0);repmeans={}
    for rep in ('en','zh','python','en_reordered'):
        repmeans[rep]=y[[i for i in train if rows[i]['representation']==rep]].mean(0)
    controls=[]
    for split,rep in sorted({(r['split'],r['representation']) for r in rows}):
        ix=[i for i,r in enumerate(rows) if r['split']==split and r['representation']==rep]
        for j,name in enumerate(frozen['direction_order']):
          for part,label in enumerate(frozen['parts']):
            err=(prediction[ix,j,part]-y[ix,j,part])**2;rep_error=(repmeans[rep][j,part]-y[ix,j,part])**2
            controls.append({'split':split,'representation':rep,'direction':name,'part':label,'rows':len(ix),
              'forecast_mse':float(err.mean()),'representation_only_mse':float(rep_error.mean()),
              'relative_to_representation_mean':float(err.sum()/max(rep_error.sum(),1e-30)),
              'source_cluster_error_advantage':clustered(err-rep_error,[rows[i]['source_group'] for i in ix])})
    lookup={(r['source_group'],r['representation']):i for i,r in enumerate(rows)}
    en=np.array([i for i,r in enumerate(rows) if r['representation']=='en']);isometry=[];grams={}
    with np.load(out/'gram.npz') as z:
      for part in ('full','content','format'):
        gram=z[part].astype(float);norm=np.sqrt(np.diag(gram).clip(1e-30));cos=gram/norm[:,None]/norm[None,:]
        for rep in ('zh','python','en_reordered'):
            other=np.array([lookup[rows[i]['source_group'],rep] for i in en]);a=cos[en][:,en];b=cos[other][:,other]
            grams[part+'_'+rep+'_EN']=a;grams[part+'_'+rep+'_other']=b
            for split in ('train','validation','test','mixed_holdout'):
                ix=[i for i,e in enumerate(en) if rows[e]['split']==split];aa=a[ix][:,ix];bb=b[ix][:,ix];upper=np.triu_indices(len(ix),1)
                aa=aa[upper];bb=bb[upper]
                isometry.append({'part':part,'representation':rep,'split':split,'cases':len(ix),'pairwise_relations':len(aa),
                  'max_absolute_unit_gradient_gram_difference':float(np.max(abs(aa-bb))),
                  'rms_gram_difference':float(np.sqrt(np.mean((aa-bb)**2))),
                  'pearson_pair_relation':float(np.corrcoef(aa,bb)[0,1]),
                  'interpretation':'A single exact Euclidean isometry of these normalized gradient vectors would require equalGram. Nonzero discrepancy rules out that exact finite-set claim, not every approximate representation map or semantic correspondence.'})
    npz(out/'cross_expression_full_gradient_relations.npz',**grams)
    result={'timestamp':stamp(),'source':snapshot(__file__),'effect_prediction_controls':controls,'isometry_necessary_condition':isometry,
      'new_control_scope':'Representation is available from the task text. This stronger baseline separates global expression-format means from within-expression predictive gains. No new predictor was fitted to heldout outcomes.',
      'gradient_scope':'Every native scalar gradient contributes via exact factors. Gradients depend on supervised target labels; geometric comparison is not an online answer-free semantic encoding claim.',
      'seconds':time.monotonic()-start}
    save(out/'forecast_audit.json',result);ledger('effect_and_cross_expression_audit',result['seconds']);print('FORECAST_AUDIT_DONE',len(controls),len(isometry),flush=True)

if __name__=='__main__':main()
