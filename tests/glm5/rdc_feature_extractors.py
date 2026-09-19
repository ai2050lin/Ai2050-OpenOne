"""Exact full-coordinate kernels with train-only scaling and validation-only ridge choice."""
from rdc_feature_common import np

ALGORITHMS = ('A0_mean','A0_distance','A1_linear','A2_quadratic','A2_cubic','A3_ordered_pair','A4_conditional')
LAMBDAS = (1e-6, 1e-3, .1)

def normalize_blocks(blocks, train):
    scales = [max(float(np.sqrt(np.mean(np.sum(np.asarray(x[train],dtype=np.float64)**2,axis=1)))),1e-12) for x in blocks]
    return [np.asarray(x,dtype=np.float64)/s for x,s in zip(blocks,scales)], scales

def kernel(blocks, left, right, name):
    dots = [x[left] @ x[right].T for x in blocks]
    linear = sum(dots)/len(dots)
    if name in ('A1_linear','A5_native'): return 1+linear
    if name == 'A2_quadratic': return (1+linear)**2
    if name == 'A2_cubic': return (1+linear)**3
    if name == 'A3_ordered_pair': return 1+linear+dots[0]*dots[1]
    if name == 'A4_conditional': return 1+linear+dots[0]*dots[1]+dots[0]*dots[1]*dots[2]
    raise ValueError(name)

def metrics(y, prediction, classification):
    out={'mse':float(np.mean((y-prediction)**2)),'n':len(y)}
    if classification: out['accuracy']=float(np.mean(np.argmax(y,axis=1)==np.argmax(prediction,axis=1)))
    return out

def fit_predict(blocks, y, train, val, test, name, classification=False):
    y=np.asarray(y,dtype=np.float64)
    if y.ndim==1: y=y[:,None]
    train,val,test=[np.asarray(a,dtype=int) for a in (train,val,test)]
    assert len(set(train)&set(val))==len(set(train)&set(test))==len(set(val)&set(test))==0
    assert len(train) and len(val) and len(test)
    normalized,scales=normalize_blocks(blocks,train)
    params={'name':name,'scales':scales,'train':train,'val':val,'test':test}
    if name=='A0_mean':
        pred=np.repeat(y[train].mean(0)[None],len(test),axis=0)
        return metrics(y[test],pred,classification),pred,params
    if name=='A0_distance':
        z=np.concatenate(normalized,axis=1)
        d=np.sum(z[test]**2,axis=1)[:,None]+np.sum(z[train]**2,axis=1)[None]-2*z[test]@z[train].T
        pred=y[train[np.argmin(d,axis=1)]]
        return metrics(y[test],pred,classification),pred,params
    gram=kernel(normalized,train,train,name)
    e,q=np.linalg.eigh((gram+gram.T)*.5)
    assert e.min()>-1e-7*max(1,float(e.max())), 'Kernel PSD check failed'
    e=np.maximum(e,0)
    vy=kernel(normalized,val,train,name); ty=kernel(normalized,test,train,name)
    qty=q.T@y[train]
    candidates=[]
    for ridge in LAMBDAS:
        alpha=q@(qty/(e[:,None]+ridge))
        loss=float(np.mean((vy@alpha-y[val])**2))
        candidates.append((loss,ridge,alpha))
    loss,ridge,alpha=min(candidates,key=lambda r:(r[0],-r[1]))
    pred=ty@alpha
    params.update(alpha=alpha,ridge=ridge,validation_mse=loss,
                  z_train=np.concatenate([x[train] for x in normalized],axis=1)/np.sqrt(len(blocks)),
                  raw_scale_vector=np.concatenate([np.repeat(s*np.sqrt(len(blocks)),x.shape[1]) for s,x in zip(scales,blocks)]))
    return dict(metrics(y[test],pred,classification),ridge=ridge,validation_mse=loss),pred,params

def quadratic_row(z,alpha,scale,j,start,count,target=0):
    """Lazy raw-coordinate coefficients; x^T M x uses ordered (j,k) entries."""
    weights=alpha[:,target]
    stop=min(start+count,z.shape[1])
    m=(weights*z[:,j]) @ z[:,start:stop]/(scale[j]*scale[start:stop])
    linear=2*np.sum(weights*z[:,j])/scale[j]
    return float(np.sum(weights)),float(linear),m
