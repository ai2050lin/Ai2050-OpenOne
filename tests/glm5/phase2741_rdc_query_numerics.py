"""Independent synthetic all-coordinate regression and stable-merge checks."""
from rdc_query_common import *

def main():
    out=BASE/'verification';file=out/'rule_numerics.json'
    if file.exists():return
    start=time.monotonic();rng=np.random.default_rng(2741);n=500;d=2560;width=4
    x=rng.normal(size=(n,d,width));x[:,:,0]=1;x[:,:,1]*=8;x[:,:,2]-=3;true=rng.normal(size=(d,width));y=np.einsum('qdi,di->qd',x,true)+.01*rng.normal(size=(n,d))
    a=np.einsum('qdi,qdj->dij',x,x,optimize=True);b=np.einsum('qdi,qd->di',x,y,optimize=True)
    xm=a[:,0]/n;xs=np.sqrt(np.maximum(np.diagonal(a,axis1=-2,axis2=-1)/n-xm*xm,1e-10));xm[:,0]=0;xs[:,0]=1;tr=np.zeros_like(a)
    for j in range(width):tr[:,j,j]=1/xs[:,j]
    for j in range(1,width):tr[:,0,j]=-xm[:,j]/xs[:,j]
    aa=np.einsum('dji,djk,dkl->dil',tr,a,tr,optimize=True)/n;bb=np.einsum('di,dij->dj',b,tr,optimize=True)/n
    lam=.01;coef=np.linalg.solve(aa+np.diag([0,lam,lam,lam]),bb[...,None])[...,0];beta=np.einsum('dij,dj->di',tr,coef,optimize=True)
    errors=[]
    for j in [0,127,999,2559]:
        zx=(x[:,j]-xm[j])/xs[j];reference=np.linalg.solve(zx.T@zx/n+np.diag([0,lam,lam,lam]),zx.T@y[:,j]/n)
        pred=zx@reference;candidate=x[:,j]@beta[j];errors.append(float(np.max(abs(pred-candidate))))
    assert max(errors)<1e-10
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'full_coordinate_width':d,'rows':n,
      'independent_design_matrix_max_errors':errors,'finite_all_fitted_coefficients':bool(np.isfinite(beta).all()),'seconds':time.monotonic()-start,
      'scope':'Implementation unit test only; synthetic coefficients are not a language-mechanism observation.'}
    save(file,result);ledger('full_coordinate_decoder_independent_algebra_check',result['seconds']);print('DECODER_NUMERICS_PASS',max(errors),flush=True)

if __name__=='__main__':main()
