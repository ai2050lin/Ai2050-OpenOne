"""Prospective CPU algebra checks; no real-model experiment or semantic claim."""
from rdc_construction_common import *
from rdc_question_selectivity import group_center,contrast_transform,objective,SelectivityRidge,SelectivityKernel,contrast_scores


def main():
    # Preserve stage ordering. Authoring this test is preparation; execution
    # and result registration wait for the complete Phase2747 handoff.
    assert read(BASE/'phase2747/delivery_manifest.json')['phase2747_complete']
    start=time.monotonic();rng=np.random.default_rng(2748001);checks=[]
    for n,d,q in [(12,5,7),(25,9,3),(9,20,11)]:
        groups=np.arange(n)//3;weights=rng.uniform(.2,2,n);weights/=weights.sum()
        x=rng.normal(size=(n,d));y=rng.normal(size=(n,q))
        p=np.eye(n)
        for group in np.unique(groups):
            ix=np.where(groups==group)[0]
            p[np.ix_(ix,ix)]-=np.tile(weights[ix]/weights[ix].sum(),(len(ix),1))
        assert np.allclose(group_center(y,groups,weights),p@y,atol=1e-13)
        assert np.allclose(group_center(np.ones((n,2)),groups,weights),0,atol=1e-13)
        for a in [0.,1.,4.,16.]:
            w=np.diag(weights);m=w+a*p.T@w@p;b=contrast_transform(np.eye(n),groups,weights,a)
            assert np.allclose(b.T@b,m,atol=2e-13)
            assert np.isclose(objective(y,groups,weights,a),np.square(b@y).sum(),atol=1e-11)
            fit=SelectivityRidge(a,.17).fit(x,y,groups,weights)
            xx=(x-fit.x_mean)/fit.x_scale;yy=y-fit.y_mean
            exact=np.linalg.solve(xx.T@m@xx+.17*np.eye(d),xx.T@m@yy)
            assert np.allclose(fit.coefficients,exact,rtol=1e-10,atol=1e-11)
            prediction=fit.predict(x)
            assert np.allclose(prediction,xx@exact+fit.y_mean,atol=1e-11)
            assert fit.audit['eigenvalues_retained']==n
            tests=contrast_scores(prediction,y,groups,weights)
            direct=float(np.einsum('n,nd,nd->',weights,p@(prediction-y),p@(prediction-y))/q)
            assert abs(tests['within_context_response_MSE']-direct)<1e-12
            # Same model sees no group labels at prediction; adding a constant
            # to all training targets changes the intercept only.
            shift=rng.normal(size=q);shifted=SelectivityRidge(a,.17).fit(x,y+shift,groups,weights)
            assert np.allclose(shifted.predict(x),prediction+shift,atol=1e-11)
            checks.append({'shape':[n,d,q],'strength':a,'transformed_objective_primal_dual_predictions_and_offset_exact':True})
    kernel_checks=[]
    for n,dq,dc,dy in [(12,3,4,5),(9,5,2,3)]:
        q=rng.normal(size=(n,dq));c=rng.normal(size=(n,dc));y=rng.normal(size=(n,dy))
        groups=np.arange(n)//3;w=rng.uniform(.2,2,n);w/=w.sum()
        for alpha in [0.,4.]:
            for interaction in [0.,1.,3.]:
                model=SelectivityKernel(alpha,.17,interaction).fit(q,c,y,groups,w)
                qq=model.train_query;cc=model.train_context
                def explicit(a,b):
                    parts=[a/np.sqrt(dq),b/np.sqrt(dc)]
                    if interaction:parts.append(np.einsum('ni,nj->nij',a,b).reshape(len(a),-1)*np.sqrt(interaction/(dq*dc)))
                    return np.concatenate(parts,axis=1)
                features=explicit(qq,cc)
                direct=SelectivityRidge(alpha,.17,standardize=False).fit(features,y,groups,w)
                query=rng.normal(size=(5,dq));context=rng.normal(size=(5,dc))
                test=explicit((query-model.query_mean)/model.query_scale,(context-model.context_mean)/model.context_scale)
                assert np.allclose(model.predict(query,context),direct.predict(test),rtol=1e-10,atol=1e-11)
                if interaction:
                    for i in range(dq):
                        for j in range(dc):
                            expected=direct.coefficients[dq+dc+i*dc+j]*np.sqrt(interaction/(dq*dc))
                            assert np.allclose(model.bilinear_fitted_coefficient(i,j),expected,rtol=1e-10,atol=1e-11)
                kernel_checks.append({'shape':[n,dq,dc,dy],'strength':alpha,'interaction':interaction,
                    'explicit_full_cartesian_features_equal_dual_predictions_and_all_pair_coefficients':True})
    folder=BASE/'phase2748/unit';folder.mkdir(parents=True,exist_ok=True)
    result={'timestamp':stamp(),'source':snapshot(__file__),'algorithm':snapshot(Path(__file__).with_name('rdc_question_selectivity.py')),
        'all_passed':True,'checks':checks,'kernel_checks':kernel_checks,'seconds':time.monotonic()-start,
        'scope':'Known weighted least-squares identities on small random CPU arrays only. '
                'No claim of native language prediction, semantic selectivity, new mathematics or Phase2748 completion.'}
    path=folder/('selectivity_'+str(time.time_ns())+'.json');save(path,result)
    print('SELECTIVITY_ALGEBRA_CHECKS',len(checks),flush=True)


if __name__=='__main__':main()
