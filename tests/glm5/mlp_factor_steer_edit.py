"""Apply fixed learned evidence vectors to native Head/MLP writers at source tokens."""
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch

import mlp_factor_steer_common as C
from mlp_factor_steer_fit import array_key
from mlp_factor_steer_materials import WRITE_ROLES


def positions(rows,device):
    return torch.tensor([[r['anchors'][k] for k in WRITE_ROLES] for r in rows],device=device)


def orientation(rows,device):
    return torch.tensor([1. if r['expected']=='mining' else -1. for r in rows],device=device)


def load_vectors(device):
    path=C.OUT/'directions.npz';assert C.file_sha(path)==C.read_gz(C.OUT/'factorial-result.json.gz')['directions_sha256']
    with np.load(path) as arrays:return {k:torch.from_numpy(arrays[k]).to(device) for k in arrays if k.endswith(('_H','_M'))}


@contextmanager
def write(model,pos,sign,recipe,vectors,alpha,mode='joint',reference=None):
    """Modify exact head slices or MLP outputs. Match the real joint L2 before BF16 addition."""
    heads=defaultdict(list)
    if mode!='mlp_only':
        for b,h in recipe['heads']:heads[b].append(h)
    mlps=recipe['mlps'] if mode!='head_only' else []
    use_matched=mode=='matched';i,role=(pos>=0).nonzero().T;j=pos[i,role];hooks=[];norms={}
    def apply(value,key,delta):
        if use_matched:
            rng=np.random.default_rng(C.SEED+key[0]*53+key[1]);d=delta.shape[-1]
            permutation=torch.as_tensor(rng.permutation(d),device=value.device)
            signs=torch.as_tensor(rng.choice(np.asarray([-1,1],np.float32),size=d),device=value.device)
            delta=delta[:,permutation]*signs
            target=reference[key]['applied']
            length=delta.norm(dim=-1)
            delta=delta*(target/length.clamp_min(1e-30))[:,None]
            empty=length==0
            if empty.any():delta[empty,0]=target[empty]
        result=(value.float()+delta).to(value.dtype)
        norms[key]={'applied':(result.float()-value.float()).norm(dim=-1),'pre_rounding':delta.norm(dim=-1)}
        return result
    try:
        for b,hlist in heads.items():
            def edit_head(_,args,b=b,hlist=hlist):
                x=args[0];out=x.clone();anchor=x[i,j].reshape(len(i),32,128).clone()
                for h in hlist:
                    delta=vectors[array_key(b,'H')][h,role]*sign[i,None]*alpha
                    anchor[:,h]=apply(anchor[:,h],(b,h),delta)
                out[i,j]=anchor.flatten(1)
                return (out,*args[1:])
            hooks.append(model.model.layers[b].self_attn.o_proj.register_forward_pre_hook(edit_head))
        for b in mlps:
            def edit_mlp(_,args,x,b=b):
                out=x.clone();delta=vectors[array_key(b,'M')][0,role]*sign[i,None]*alpha
                out[i,j]=apply(x[i,j],(b,32),delta);return out
            hooks.append(model.model.layers[b].mlp.register_forward_hook(edit_mlp))
        yield norms
    finally:
        for h in hooks:h.remove()


def prefill(model,x,pos,sign,recipe,vectors,alpha,mode='joint',reference=None):
    with write(model,pos,sign,recipe,vectors,alpha,mode,reference) as norms:
        cache=model(input_ids=x[:,:C.CUT],use_cache=True,logits_to_keep=1).past_key_values
    return cache,norms
