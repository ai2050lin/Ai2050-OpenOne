"""Pre-fit checks of complete kernels, weighting and native decoder commutation."""
from rdc_law_common import *
from rdc_law_predict import kernel_torch,ridge_lambda,normalized_features,decode
from phase2729_rdc_law_prediction import training_weights


def main():
    import torch
    torch.set_num_threads(2);rng=np.random.default_rng(2729)
    meta=[{'cohort':c,'source_group':s} for c,s in [('a','x'),('a','x'),('a','y'),('b','z'),('b','z'),('b','z')]]
    ww=training_weights(meta,np.arange(6))
    assert np.allclose(ww,[.75,.75,1.5,1,1,1])
    raw={k:rng.normal(size=(17,8)) for k in ('q','embedding','history','routed')}
    raw.update(position=np.arange(17),**{'class':np.arange(17)%4})
    f,rulers=normalized_features(raw,np.arange(10),np.ones(10))
    ff,_=normalized_features(raw,rulers=rulers)
    assert all(np.array_equal(f[k],ff[k]) for k in f)
    t={k:torch.as_tensor(v) for k,v in f.items()}
    kernels=[]
    for name in ('early_linear','additive_history','multiplicative_history','routed_history','task_conditioned'):
        k=kernel_torch(name,t,t);e=torch.linalg.eigvalsh(k)
        assert float(e.min())>-1e-10
        lam,df=ridge_lambda(e,6)
        kernels.append({'name':name,'min_eigenvalue':float(e.min()),'lambda':lam,'effective_df':df})
    S=rng.normal(size=(7,10));a=rng.normal(size=(10,13));wd=rng.normal(size=(8,13))
    error=float(np.max(abs((S@a)@wd.T-S@(a@wd.T))))
    assert error<1e-12
    assert not (BASE/'prediction/result.json').exists()
    immutable(BASE/'prediction/protocol_refinement_before_fit.json',{
        'timestamp':stamp(),'source':snapshot(ROOT/'tests/glm5/phase2729_rdc_law_prediction.py'),
        'reason':'Pre-execution code review found ambiguous two-stage df wording and source-group weighting mismatch; no fit or validation/test predictions had run.',
        'weighting':'Equal six cohort totals; equal source_group totals within cohort; divide across all query anchors of that source. Multiple sample windows from same source do not increase its training weight.',
        'selection_clarification':'The comparative per-kernel table selects df with validation relative MSE. Separate deployment banks select each decoder kernel/df jointly across the full validation grid: relative MSE at16 and full-vocabulary KL at35. This is not the two-stage MSE-then-KL selection suggested by original wording.',
        'test_control':'Main test remains unused for all selection. Confirmation not captured. No expansion of candidates or tuning on outcomes.'})
    save(BASE/'verification/prediction_math.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'passed':True,
        'source_weights':ww.tolist(),'kernels':kernels,'linear_commutation_max_abs':error,
        'scope':'Synthetic CPU float64 implementation checks, not native-language experimental results.'})
    print('LAW_PREDICTION_MATH_PASS',error,flush=True)


if __name__=='__main__':main()
