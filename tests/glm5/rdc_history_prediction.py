"""Past-only cache reconstruction and native query-override candidate operators."""
from rdc_construction_common import *
from rdc_native_tail import block_call,config


def past_cache(z,step):
    # Arrays are lossless original BF16. At step0, prefix last token is current.
    # At step1, the whole original prefix is past. At step2, exactly one
    # appended generated-token entry is additionally past.
    ix=z['cache_blocks'].tolist().index(35)
    result=[]
    for kind in ['keys','values']:
        prefix=z['prefix_cache_'+kind][ix]
        if step==0:a=prefix[:,:-1]
        else:
            appended=z['appended_cache_'+kind][:step-1,ix].transpose(1,0,2)
            a=np.concatenate([prefix,appended],axis=1)
        result.append(a)
    return result


def tensor(a,dtype=None):
    import torch
    value=unbits(a) if a.dtype==np.uint16 else a
    return torch.from_numpy(np.ascontiguousarray(value)).to(device='cuda',dtype=dtype or torch.float32)


def position_factors(rotary,position):
    import torch
    dummy=torch.zeros((1,1,2560),device='cuda',dtype=torch.bfloat16)
    return tuple(v.float() for v in rotary(dummy,torch.tensor([[position]],device='cuda')))


def complete_block(layer,h,key,value,cos,sin,query=None):
    handle=None
    if query is not None:
        query=query.reshape(1,1,32,128)
        handle=layer.self_attn.q_norm.register_forward_hook(lambda m,a,o:query)
    try:return block_call(layer,h,key,value,cos,sin,35)
    finally:
        if handle is not None:handle.remove()


def source_permutation(sid,step,length):
    return np.random.default_rng(int(rank('history_source_permutation/'+sid+'/'+str(step))[:16],16)).permutation(length)


def normalized_query(query,gain):
    import torch
    q=query.reshape(32,128);gain=gain.reshape(1,128)
    supported=gain!=0
    latent=torch.where(supported,q/torch.where(supported,gain,torch.ones_like(gain)),torch.zeros_like(q))
    return gain*latent/latent.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-12)


def weights(rows):
    from collections import Counter
    families=sorted({r['family'] for r in rows})
    groups={f:Counter(r['source_group'] for r in rows if r['family']==f) for f in families}
    value=np.array([1/(len(families)*len(groups[r['family']])*groups[r['family']][r['source_group']]) for r in rows])
    assert abs(value.sum()-1)<1e-10
    return value
