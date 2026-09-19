"""Independent selected-rule inference, kernel algebra, frozen inputs and objective audit."""
import hashlib
from rdc_joint_common import *
from rdc_joint_features import load_features
from rdc_joint_kernels import Rule,gram_for,CURRENT,TEMPORAL


def main():
    frozen=read(BASE/'frozen.json');checks=[]
    for name,digest in frozen['files'].items():assert sha(BASE/name)==digest,name
    checks.append({'name':'all_frozen_files','count':len(frozen['files']),'passed':True})
    raw,y,post,meta=load_features()
    ii=np.array(read(BASE/'rules/indices.json')['test'][:8])
    for scope,label in [(s,frozen['choices'][s+'_MSE']) for s in ('current','temporal')]+[(s,frozen['choices'][s+'_KL']) for s in ('current','temporal')]+[('temporal',frozen['choices']['temporal_history_KL'])]:
        kl=label.startswith('KL_');name=label[3:] if kl else label
        evaluator=Rule(scope,name,kl=kl)
        data={k[len(scope)+1:]:v[ii] for k,v in raw.items() if k.startswith(scope+'_')}
        p=evaluator(data)
        folder=BASE/('probability_training' if kl else 'rules')/scope/name
        with np.load(folder/'predictions.npz') as z:ref=z['test'][:len(ii)]
        diff=float(np.max(np.abs(p-ref)))
        assert diff<5e-4,(scope,label,diff)
        checks.append({'name':'independent_full_coordinate_rule_'+scope+'_'+label,'max_abs_difference':diff,'passed':True})
        del evaluator
    rng=np.random.default_rng(2720)
    h,e,c=[rng.normal(size=(5,d)) for d in (3,4,2)]
    hd,ed,cd=h@h.T,e@e.T,c@c.T
    explicit=np.stack([np.einsum('i,j,k->ijk',hh,ee,cc).ravel() for hh,ee,cc in zip(h,e,c)])
    assert np.allclose(explicit@explicit.T,hd*ed*cd,rtol=1e-12,atol=1e-12)
    checks.append({'name':'exact_trilinear_kernel_matches_every_explicit_feature_toy_coordinate','passed':True})
    pred=BASE/'rules/current'
    with np.load(pred/'current_linear/predictions.npz') as a,np.load(pred/'raw_mean/predictions.npz') as b:
        assert all(np.array_equal(a[k],b[k]) for k in a.files)
    checks.append({'name':'zero_raw_history_mixture_exactly_plain_current','passed':True})
    for item in read(BASE/'review.json')['evidence']:
        assert sha(ROOT/item['path'])==item['sha256']
    checks.append({'name':'all21_reviewed_old_artifacts_unchanged','passed':True})
    prefix=read(BASE/'memo_prefix.json')
    memo=(ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md').read_bytes()
    assert hashlib.sha256(memo[:prefix['bytes']]).hexdigest()==prefix['sha256']
    checks.append({'name':'original_MEMO_byte_prefix_unchanged','bytes':prefix['bytes'],'passed':True})
    report={'timestamp':stamp(),'passed':True,'checks':checks,'source':snapshot(Path(__file__)),
        'current_winner':read(BASE/'rules/result.json')['current']['validation_MSE_winner'],
        'temporal_winner':read(BASE/'rules/result.json')['temporal']['validation_MSE_winner'],
        'KL_choices':frozen['choices'],'limits':'Independent inference and exact toy algebra do not prove generalization or native mechanistic identity. KL optimized coefficients are not assumed to preserve ridge effective df.'}
    save(BASE/'verification/phase2720_integrity.json',report)
    print('JOINT_INTEGRITY',report,flush=True)
    fit=read(BASE/'rules/result.json');prob=read(BASE/'probability_training/result.json')
    summary={'timestamp':stamp(),'choices':frozen['choices'],'MSE':{},'KL_state_tradeoff':[]}
    for scope in ('current','temporal'):
        summary['MSE'][scope]={name:{'df':r['effective_df'],'mix':r['mix'],'ridge':r['ridge'],
            'validation_normalized_MSE':r['validation_normalized_MSE'],'test_MSE':r['test']['anchor_MSE'],
            'test_layer':r.get('test_by_target_layer',{}),
            'matched_df_test':fit[scope]['matched_effective_df'].get(name,{}).get('test',{})} for name,r in fit[scope]['candidates'].items()}
    for r in prob['state']:
        if r['split']=='test' and r['route'] in ('current_linear','current_quadratic','KL_current_linear','KL_current_quadratic','embedding_bilinear','KL_embedding_bilinear','history_trilinear','KL_history_trilinear'):
            pr=next(p for p in prob['probability'] if p['scope']==r['scope'] and p['split']=='test' and p['route']==r['route'])
            summary['KL_state_tradeoff'].append({'scope':r['scope'],'route':r['route'],'H36_MSE':r['state']['anchor_MSE'],'KL':pr['KL'],'agreement':pr['argmax_agreement'],'observed_NLL':pr['predicted_observed_token_NLL'],'native_observed_NLL':pr['native_observed_token_NLL']})
    save(BASE/'verification/phase2720_summary.json',summary)
    print('JOINT_OBJECTIVE_TRADEOFF',summary['KL_state_tradeoff'],flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
