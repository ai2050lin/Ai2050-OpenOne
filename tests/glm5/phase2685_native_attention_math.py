"""Full-source, full-head-coordinate local arithmetic; never donor transport.

These FP64 analysis functions are not the native BF16 execution and do not
predict changes beyond the observed attention layer.
"""
import numpy as np

def rms(x,gamma,epsilon):
    x=np.asarray(x,dtype=np.float64)
    return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+epsilon)*gamma

def rotate_half(x):
    a,b=np.split(x,2,axis=-1)
    return np.concatenate((-b,a),axis=-1)

def rope(x,cos,sin):
    return x*cos[:,None,:]+rotate_half(x)*sin[:,None,:]

def routing(q,k,v,mask,scale,positions):
    """Every source, every head dimension. Only query rows may be requested."""
    assert q.ndim==k.ndim==v.ndim==3 and k.shape==v.shape
    n,h,d=q.shape;kv=k.shape[1];assert h%kv==0 and k.shape[0]==n
    full_k=np.repeat(k,h//kv,axis=1);full_v=np.repeat(v,h//kv,axis=1)
    score=np.einsum('qhd,shd->qhs',q[list(positions)],full_k)*scale+mask
    ex=np.exp(score-score.max(axis=-1,keepdims=True));prob=ex/ex.sum(axis=-1,keepdims=True)
    return prob,np.einsum('qhs,shd->qhd',prob,full_v)

def local_scalar_path(x,q_linear,k_linear,v_linear,q_gamma,k_gamma,epsilon,cos,sin,mask,scale,positions,kind,output_row,input_coordinate,delta):
    """A single learned scalar acts at ALL actual tokens, not one token only.

    x is unchanged because it is this layer's upstream input. Q/K head RMS
    denominators are recomputed endogenously, not frozen to observed values.
    """
    assert kind in ('q','k','v') and np.isfinite(delta)
    a={'q':np.asarray(q_linear,dtype=np.float64).copy(),'k':np.asarray(k_linear,dtype=np.float64).copy(),'v':np.asarray(v_linear,dtype=np.float64).copy()}
    a[kind].reshape(len(x),-1)[:,output_row]+=float(delta)*np.asarray(x,dtype=np.float64)[:,input_coordinate]
    q=rope(rms(a['q'],q_gamma,epsilon),cos,sin);k=rope(rms(a['k'],k_gamma,epsilon),cos,sin)
    prob,heads=routing(q,k,a['v'],mask,scale,positions)
    return {'q':q,'k':k,'v':a['v'],'probability':prob,'head_output':heads}

def self_test():
    # Fixed arithmetic fixtures, not random statistics or pretrained evidence.
    reports=[]
    for n in (3,7,11):
        h,kv,d,inp=4,2,8,13
        x=((np.arange(n*inp)%19)-9).reshape(n,inp)/16
        matrices={kind:((np.arange(heads*d*inp)%23)-11).reshape(heads*d,inp)/32 for kind,heads in (('q',h),('k',kv),('v',kv))}
        gamma=np.arange(1,d+1,dtype=float)/8;eps=1e-6
        # Exact quarter-turn position fixtures avoid attributing trig identities
        # to language structure; native experiments use captured model cos/sin.
        cos=np.repeat((np.arange(n)%2==0)[:,None],d,axis=1).astype(float)
        sin=np.repeat((np.arange(n)%2==1)[:,None],d,axis=1).astype(float)
        pos=(0,n-1);mask=np.zeros((2,h,n));mask[0,:,1:]=-np.inf
        linear={kind:(x@w.T).reshape(n,h if kind=='q' else kv,d) for kind,w in matrices.items()}
        baseline=local_scalar_path(x,linear['q'],linear['k'],linear['v'],gamma,gamma,eps,cos,sin,mask,d**-.5,pos,'q',0,0,0)
        assert (baseline['probability'][0,:,1:]==0).all()
        for kind in ('q','k','v'):
            for row in (0,len(matrices[kind])-1):
                for col in (0,inp-1):
                    for delta in (-.125,.125):
                        changed=matrices[kind].copy();changed[row,col]+=delta
                        direct=dict(linear);direct[kind]=(x@changed.T).reshape(linear[kind].shape)
                        q=rope(rms(direct['q'],gamma,eps),cos,sin);k=rope(rms(direct['k'],gamma,eps),cos,sin)
                        p,o=routing(q,k,direct['v'],mask,d**-.5,pos)
                        predicted=local_scalar_path(x,linear['q'],linear['k'],linear['v'],gamma,gamma,eps,cos,sin,mask,d**-.5,pos,kind,row,col,delta)
                        err=max(float(np.abs(p-predicted['probability']).max()),float(np.abs(o-predicted['head_output']).max()))
                        assert err<1e-12 and np.allclose(p.sum(-1),1,rtol=0,atol=1e-14)
                        if kind=='v':assert np.array_equal(p,baseline['probability'])
                        reports.append({'tokens':n,'kind':kind,'row':row,'input':col,'delta':delta,'max_error':err})
    assert len(reports)==72
    return {'synthetic_cases':72,'max_error':max(r['max_error'] for r in reports),'all_checks_passed':True,
            'pretrained_model_evidence':False,'records':reports}

if __name__=='__main__':print(self_test())
