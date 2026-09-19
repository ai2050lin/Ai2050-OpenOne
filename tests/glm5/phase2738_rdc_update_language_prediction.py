"""Frozen-before-material natural predictors on untouched bilingual held groups."""
from rdc_update_common import *

def main():
    import torch
    from rdc_update_graph import pack,kernels
    from rdc_law_native import parameter
    from phase2736_rdc_update_prediction import decoders,reports
    out=BASE/'language_prediction';start=time.monotonic()
    if (out/'result.json').exists():return
    frozen=read(BASE/'graph/frozen.json');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    rows=[dict(r,anchors=[p],field_anchor=a,cohort=r['cohort']+'/'+r['answer_style']+f'/anchor{a}')
          for r in gzread(BASE/'language_material.json.gz') if r['split']=='language_test' for a,p in enumerate(r['anchors'])]
    assert len(rows)==320
    old=[r for r in gzread(PRIOR/'natural_discovery.json.gz') if r['split']=='train']
    with np.load(BASE/'graph/head_mapping.npz') as z:coef=torch.tensor(z['coefficients'],device='cuda',dtype=torch.float32)
    left,targets,audit=pack(rows,coef);right,_,_=pack(old,coef);kk=kernels(left,right);del left,right
    resultrows=[];pairs=[]
    for b,targets0 in targets.items():
        selected=frozen['selected'][str(b)];yy=torch.tensor(targets0[:,:2560],device='cuda');errors={}
        w={k:parameter(f'model.layers.{b}.mlp.{name}_proj.weight') for k,name in [('g','gate'),('u','up'),('d','down')]}
        for name in (selected['kernel'],'query'):
            bank=f'b{b}_{name}_{selected["df"]}.npz';assert sha(BASE/'graph/banks'/bank)==frozen['banks_sha256'][bank]
            with np.load(BASE/'graph/banks'/bank) as z:
                center=torch.tensor(z['center'],device='cuda');p=kk[name]/float(z['scale'])@torch.tensor(z['coefficients'],device='cuda')+center
            prediction=decoders(p,w)[selected['decoder']];err=(prediction-yy).square().sum(-1).cpu().numpy();den=(yy-center[:2560]).square().sum(-1).cpu().numpy();errors[name]=err
            npz(out/f'b{b}_{name}.npz',prediction=prediction.cpu().numpy(),actual=yy.cpu().numpy(),squared_error=err,baseline_squared_error=den)
            resultrows.append({'block':b,'kernel':name,'decoder':selected['decoder'],'reports':reports(rows,err,den)})
        for cohort in sorted({r['cohort'] for r in rows}):
            ix=[i for i,r in enumerate(rows) if r['cohort']==cohort];delta=(errors[selected['kernel']][ix]-errors['query'][ix])/den[ix].clip(1e-12)
            pairs.append({'block':b,'cohort':cohort,'held_expressions':len(ix),'clustered_difference':clustered(delta,[rows[i]['source_group'] for i in ix]),
              'pooled_relative_MSE_difference':float((errors[selected['kernel']][ix]-errors['query'][ix]).sum()/den[ix].sum())})
        del yy,w
    compressed(out/'row_identity.json.gz',[{k:r[k] for k in ('sample_id','source_group','cohort','anchors','field_anchor')} for r in rows])
    result={'timestamp':stamp(),'source':snapshot(__file__),'held_expressions':160,'boundaries':320,'records':resultrows,'selected_minus_query':pairs,'seconds':time.monotonic()-start,
      'frozen_sha256':sha(BASE/'graph/frozen.json'),'scope':'Original2736 natural-training banks frozen before640language material existed. No refit or language-label input. Five controlled families and two anchor types are distribution shifts; this is not exact cross-language semantic alignment.'}
    save(out/'result.json',result);ledger('frozen_natural_rules_on_bilingual_held_families',result['seconds']);print('LANGUAGE_FORECAST_DONE',result['seconds'],flush=True)

if __name__=='__main__':main()
