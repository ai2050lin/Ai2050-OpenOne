"""Synthetic software checks, not language-model evidence."""
from rdc_conditional_common import *
from rdc_conditional_estimators import FullKernel
from phase2704_rdc_predictive_gates import gate_fit,sigmoid


def main():
    rng=np.random.default_rng(2703);x=rng.normal(size=(64,12));w=rng.normal(size=(12,3));y=x@w+.2
    tr,va,te=np.arange(32),np.arange(32,48),np.arange(48,64)
    fit=FullKernel(x,tr,va,te);p,m=fit.fit(y);assert m['mse']<1e-7
    g=rng.normal(size=(64,20));u=rng.normal(size=(64,20));a=g*sigmoid(g)*u;u[:,0]=0;a[:,0]=0
    group=np.arange(64)%4
    p,c,z=gate_fit(g,u,a,tr,group,True)
    for k in range(4):
        ix=tr[group[tr]==k];b=g[ix]*u[ix]
        assert np.abs((b*(b*c[k]-a[ix])).sum(0)).max()<1e-10
    # Cached spectral prediction agrees with direct solve on the chosen ridge.
    k=fit.kernel(tr,tr);alpha=np.linalg.solve(k+m['ridge']*np.eye(len(tr)),y[tr]);direct=fit.kernel(te,tr)@alpha
    pp,_=fit.fit(y);assert np.max(np.abs(direct-pp))<1e-6
    assert all(v==1 for v in z.values())
    save(CAMPAIGN/'estimator_checks.json',{'timestamp':stamp(),'synthetic_only':True,'linear_recovery_mse':m['mse'],'cached_vs_direct_max':float(np.abs(direct-pp).max()),'weighted_normal_equation':True,'zero_energy_fallback':True})
    print('ESTIMATOR_CHECKS_PASS',flush=True)


if __name__=='__main__':main()
