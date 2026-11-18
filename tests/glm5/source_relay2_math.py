"""Auditable forecast algebra and family-level statistics; no model inference."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_relay2_design as D

def linear_fit(z,y):
    a=np.c_[np.ones(len(z)),z].astype(np.float64);p=np.eye(a.shape[1]);p[0,0]=0
    return np.linalg.solve(a.T@a+p,a.T@y).astype(np.float32)

def transform_fit(x,y,kind,key,width=128):
    center=x.mean(0);scale=np.maximum(x.std(0),1e-5);z=(x-center)/scale
    result={'center':center.astype('f'),'scale':scale.astype('f')}
    if kind=='pca':
        from sklearn.utils.extmath import randomized_svd
        _,_,vt=randomized_svd(z,n_components=width,n_iter=2,random_state=int(D.rng(key).integers(2**31-1)))
        result['projection']=vt.T.astype('f');latent=z@result['projection'];result['latent_scale']=np.maximum(latent.std(0),1e-5).astype('f')
        return result,latent/result['latent_scale']
    if kind=='random':idx=np.sort(D.rng(key).choice(x.shape[1],width,replace=False))
    else:
        yn=(y-y.mean(0))/np.maximum(y.std(0),1e-5);score=np.square(z.T@yn/max(len(x)-1,1)).sum(1)
        idx=np.sort(np.argsort(score,kind='stable')[-width:]);result['selection_score']=score.astype('f')
    result['idx']=idx;return result,z[:,idx]

def transform(x,p):
    if 'idx' in p:return (x[...,p['idx']]-p['center'][p['idx']])/p['scale'][p['idx']]
    return ((x-p['center'])/p['scale'])@p['projection']/p['latent_scale']

def unpack(store,prefix):return {k[len(prefix)+1:]:store[k] for k in store.files if k.startswith(prefix+'_')}
def pack(store,prefix,data):
    for k,v in data.items():store[prefix+'_'+k]=v
def predict(z,coef):return coef[0]+z@coef[1:]

def softmax(x):
    e=np.exp(x-np.max(x,axis=-1,keepdims=True));return e/e.sum(-1,keepdims=True)

def bf16_round_torch(x):
    # Straight-through derivative; forward value has the exact BF16 rounding boundary.
    return x+(x.to(dtype=__import__('torch').bfloat16).float()-x).detach()

def rotate_torch(x,cos,sin):
    import torch
    x=bf16_round_torch(x);rot=torch.cat([-x[...,x.shape[-1]//2:],x[...,:x.shape[-1]//2]],dim=-1)
    return bf16_round_torch(bf16_round_torch(x*cos)+bf16_round_torch(rot*sin))

def distribution_torch(q,k,cos,sin,valid):
    qr=rotate_torch(q,cos[:,-1],sin[:,-1]);kr=rotate_torch(k,cos,sin)
    score=(qr[:,None]*kr).sum(-1)/np.sqrt(q.shape[-1]);score=score.masked_fill(~valid,float('-inf'))
    return score.softmax(-1),score

def family_means(rows,values):
    fam=sorted({r['family'] for r in rows});values=np.asarray(values)
    return np.array([values[[i for i,r in enumerate(rows) if r['family']==f]].mean(0) for f in fam])

def interval(rows,values,key):
    means=family_means(rows,values);sample=D.rng(key).integers(0,len(means),(2000,len(means)))
    return np.quantile(means[sample].mean(1),[.025,.975],axis=0).tolist()

def paired(rows,gain,key):
    means=family_means(rows,gain);random=D.rng(key);signs=random.choice([-1,1],size=(9999,len(means)))
    perm=(signs*means).mean(-1);obs=means.mean()
    return {'mean_gain':float(obs),'ci95_family':interval(rows,gain,key+'/ci'),
        'permutation_p_one_sided':float((1+(perm>=obs).sum())/(len(perm)+1)),'families':len(means)}

def holm(ps):
    order=np.argsort(ps);out=np.zeros(len(ps));last=0
    for i,k in enumerate(order):last=max(last,(len(ps)-i)*ps[k]);out[k]=min(last,1)
    return out.tolist()

def error_rows(actual,pred,G):return np.einsum('ni,ij,nj->n',actual-pred,G,actual-pred,optimize=True)

def block_tokens(r,arm):
    if arm=='base':return []
    if arm=='mask_allpast':return list(range(len(r['tokens'])-1))
    if arm in ['mask_question','mask_instruction','mask_assistant_tail']:return r['groups'][arm.removeprefix('mask_')]
    if arm=='mask_records':return r['groups']['records']
    chosen=[r['target_record']] if arm=='mask_correct' else [j for j in range(6) if j!=r['target_record']]
    assert arm in ['mask_correct','mask_wrong'],arm
    return sorted({t for j in chosen for t in r['records'][str(j)]['span']})
