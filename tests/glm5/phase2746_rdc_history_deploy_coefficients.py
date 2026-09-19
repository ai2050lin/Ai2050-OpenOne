"""Exact full-dual to full-primal conversion for the two frozen self-fed routes."""
from rdc_construction_common import *
from rdc_construction_storage import FIELD_STORE,verify_storage
from phase2746_rdc_history_prediction_contract import OUT


def main():
    start=time.monotonic();frozen=read(OUT/'frozen.json');names=[frozen['autonomous_direct_baseline_route'],frozen['autonomous_primary_native_constrained_route']]
    routes=[next(r for r in frozen['routes'] if r['name']==name) for name in names]
    assert len({(r['input_index'],r['control_index'],r['lambda']) for r in routes})==1
    route=routes[0];meta=read(OUT/'fit'/(route['input_variant']+'.json'));assert sha(BASE/meta['field_path'])==meta['field_sha256']
    with np.load(BASE/meta['field_path']) as z:
        train=z['train_Z'].astype(float);sw=z['sqrt_train_weights'].astype(float);u=z['dual_eigenvectors'].astype(float);ev=z['dual_eigenvalues'].astype(float)
        projected=z['projected_targets'][route['control_index']].astype(float);mean=z['feature_mean'];scale=z['feature_scale'];targetmean=z['target_means'][route['control_index']]
    resolved=projected/(ev+route['lambda'])[:,None]
    left=(train*sw[:,None]).T@u/train.shape[1]
    coefficients=left@resolved
    # Hash-selected training examples plus dense synthetic standardized inputs
    # test only this algebraic deployment conversion, never select model rules.
    rng=np.random.default_rng(27460);x=np.concatenate([train[::19],rng.standard_normal((32,train.shape[1]))],0)
    reference=((x@(train*sw[:,None]).T/train.shape[1])@u)@resolved+targetmean
    primal=x@coefficients+targetmean
    candidate=x.astype(np.float32)@coefficients.astype(np.float32)+targetmean.astype(np.float32)
    relative=lambda value:float(np.linalg.norm(value-reference)/max(np.linalg.norm(reference),1e-30))
    assert relative(primal)<1e-10 and relative(candidate)<1e-5
    arrays={'coefficients_FP32':coefficients.astype(np.float32),'feature_mean_FP64':mean.astype(float),
        'feature_scale_FP64':scale.astype(float),'target_mean_FP32':targetmean.astype(np.float32)}
    file=FIELD_STORE/'history_deployment/coefficients.npz';verify_storage(sum(a.nbytes for a in arrays.values()));npz(file,**arrays)
    result={'timestamp':stamp(),'all_passed':True,'source':snapshot(__file__),'frozen_sha256':sha(OUT/'frozen.json'),
        'fit_sha256':meta['field_sha256'],'routes':routes,'full_primal_shape':list(coefficients.shape),
        'original_full_dual_dimensions':list(train.shape),'FP64_primal_vs_dual_relative_L2':relative(primal),
        'FP32_primal_vs_dual_relative_L2':relative(candidate),'conversion_check_points':len(x),
        'field_path':file.relative_to(BASE).as_posix(),'field_sha256':sha(file),'field_bytes':file.stat().st_size,
        'retains_all_features_and_all_targets':True,'new_fit_or_dimension_reduction':False,'seconds':time.monotonic()-start}
    save(OUT/'deployment/coefficients.json',result);ledger('phase2746_deployment_conversion',result['seconds'])
    print('DEPLOYMENT_COEFFICIENTS',result,flush=True)


if __name__=='__main__':main()
