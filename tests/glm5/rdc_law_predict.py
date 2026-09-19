"""Full-coordinate early-state/history kernels and native-coordinate prediction banks."""
from rdc_law_common import *

KERNELS=('early_linear','additive_history','multiplicative_history','routed_history','task_conditioned',
         'shuffled_history_0','shuffled_history_1','shuffled_history_2')
DECODERS=('direct_mlp','predicted_x_native','product_of_predicted_factors','predicted_joint_product')
DF_GRID=(32,128,384)


def normalized_features(raw,train_indices=None,weights=None,rulers=None):
    """All coordinates retained, per-vector RMS followed by training-only global scaling."""
    data={}
    for key in ('q','embedding','history','routed'):
        a=raw[key].astype(float)
        if key in ('q','embedding'):a=a/np.maximum(np.sqrt(np.mean(a*a,axis=-1,keepdims=True)),1e-12)
        data[key]=a
    if rulers is None:
        assert train_indices is not None and weights is not None
        ww=weights/weights.sum();rulers={}
        for key,a in data.items():
            mean=(a[train_indices]*ww[:,None]).sum(0)
            scale=float(np.sqrt(np.sum(ww*np.mean((a[train_indices]-mean)**2,axis=-1))))
            rulers[key]={'mean':mean,'scale':max(scale,1e-8)}
    for key in data:data[key]=(data[key]-rulers[key]['mean'])/rulers[key]['scale']
    data['position']=np.log1p(raw['position'].astype(float))/np.log(2048.)
    data['class']=np.asarray(raw['class'],dtype=np.int64)
    return data,rulers


def kernel_torch(name,left,right):
    """Exact feature kernels, including every cross-coordinate bilinear product implicitly."""
    width=left['q'].shape[-1]
    base=(left['q']@right['q'].T+left['embedding']@right['embedding'].T)/(2*width)
    base=base+left['position'][:,None]*right['position'][None,:]
    if name=='early_linear':return 1+base
    if name=='task_conditioned':
        same=(left['class'][:,None]==right['class'][None,:]).to(base.dtype)
        return 1+base+base*same
    key='routed' if name=='routed_history' else name if name.startswith('shuffled_history') else 'history'
    h=left[key]@right[key].T/width
    if name=='additive_history':return 1+base+h
    return 1+base+h+base*h


def ridge_lambda(eigenvalues,target_df):
    import torch
    e=eigenvalues.clamp_min(0)
    lo=float(e.max())*1e-14;hi=float(e.max())*1e8
    for _ in range(90):
        mid=(lo*hi)**.5
        if float((e/(e+mid)).sum())>target_df:lo=mid
        else:hi=mid
    value=(lo*hi)**.5
    actual=float((e/(e+value)).sum())
    assert abs(actual-target_df)<1e-5,(actual,target_df)
    return value,actual


def widths(width=2560,units=9728):
    return {'x':slice(0,width),'phi':slice(width,width+units),'u':slice(width+units,width+2*units),
        'activation':slice(width+2*units,width+3*units),'mlp':slice(width+3*units,2*width+3*units)}


def decode(pred,decoder,w,sl=None):
    import torch.nn.functional as F
    sl=widths() if sl is None else sl
    if decoder=='direct_mlp':return pred[:,sl['mlp']]
    if decoder=='predicted_x_native':
        xx=pred[:,sl['x']]
        return F.linear(F.silu(F.linear(xx,w['g']))*F.linear(xx,w['u']),w['d'])
    if decoder=='product_of_predicted_factors':return F.linear(pred[:,sl['phi']]*pred[:,sl['u']],w['d'])
    if decoder=='predicted_joint_product':return F.linear(pred[:,sl['activation']],w['d'])
    raise KeyError(decoder)


def decoder_target_indices(decoder):
    sl=widths()
    names={'direct_mlp':['mlp'],'predicted_x_native':['x'],'product_of_predicted_factors':['phi','u'],
           'predicted_joint_product':['activation']}[decoder]
    return np.concatenate([np.arange(sl[k].start,sl[k].stop) for k in names])


def save_bank(folder,info,coeff,center,decoder):
    indices=decoder_target_indices(decoder)
    npz(folder.with_suffix('.npz'),coefficients=coeff[:,indices].astype(np.float32),target_center=center[indices].astype(np.float32),target_indices=indices)
    save(folder.with_suffix('.json'),dict(info,decoder=decoder,arrays_sha=sha(folder.with_suffix('.npz'))))


class Bank:
    def __init__(self,path,device='cuda:0'):
        import torch
        self.info=read(path.with_suffix('.json'))
        with np.load(path.with_suffix('.npz')) as z:
            self.coeff=torch.as_tensor(z['coefficients'],device=device)
            self.center=torch.as_tensor(z['target_center'],device=device)
            self.target_indices=z['target_indices']
        self.decoder=self.info['decoder'];self.kernel=self.info['kernel'];self.device=device

    def predict(self,cross_kernel,w):
        import torch
        import torch.nn.functional as F
        result=cross_kernel.to(self.coeff.dtype)@self.coeff+self.center
        if self.decoder=='direct_mlp':return result
        if self.decoder=='predicted_x_native':return F.linear(F.silu(F.linear(result,w['g']))*F.linear(result,w['u']),w['d'])
        if self.decoder=='product_of_predicted_factors':
            a,u=result.chunk(2,dim=-1);return F.linear(a*u,w['d'])
        if self.decoder=='predicted_joint_product':return F.linear(result,w['d'])
        raise KeyError(self.decoder)
