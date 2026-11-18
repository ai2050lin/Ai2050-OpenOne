"""Native full-shape B30 source edits and quantization-aware residual matching."""

import numpy as np

import paradigm_four_run as F


HEADS=(25,18)
CONTROL=20
LAYER=30
ROW_DOSE=.003
TOL=.03
COS_MIN=.90
# Dose search uses both native write norms and only materialized residuals.
GRID=tuple(i/24 for i in range(12,193))


def extract(model,block,x,o,query):
    t=model.t
    j=t.arange(len(x),device='cuda')
    projected=block.self_attn.o_proj(o.reshape(*x.shape[:-1],model.width).contiguous())
    after=x+projected
    normalized=block.post_attention_layernorm(after)
    g=block.mlp.gate_proj(normalized)
    u=block.mlp.up_proj(normalized)
    a=t.nn.functional.silu(g)*u
    return dict(after=after,g=g,u=u,a=a,output=after+block.mlp.down_proj(a),
                j=j,query=query)


def realized_match(base,left,right,t,directions):
    """Match realized BF16 displacement while retaining native write directions."""
    b=base.float()
    original=t.stack(((left.float()-b).norm(dim=-1),(right.float()-b).norm(dim=-1)),-1)
    original_d1,original_d2=directions
    d1=t.where((original[:,0]>0)[:,None],left.float()-b,original_d1)
    d2=t.where((original[:,1]>0)[:,None],right.float()-b,original_d2)
    norm1=d1.norm(dim=-1);norm2=d2.norm(dim=-1)
    goal=t.maximum(norm1,norm2)
    grid=t.as_tensor(GRID,device=b.device,dtype=t.float32)

    def levels(d,norm):
        unit=d/norm.clamp_min(1.e-12)[:,None]
        points=(b[:,None]+(grid[None,:,None]*goal[:,None,None])*unit[:,None]).to(base.dtype)
        shift=points.float()-b[:,None]
        actual=shift.norm(dim=-1)
        cosine=(shift*unit[:,None]).sum(-1)/actual.clamp_min(1.e-12)
        return points,actual,cosine

    p,a,ca=levels(d1,norm1)
    q,c,cc=levels(d2,norm2)
    larger=t.maximum(a[:,:,None],c[:,None,:])
    gaps=(a[:,:,None]-c[:,None,:]).abs()/larger.clamp_min(1.e-12)
    legal=(gaps<=TOL)&(t.minimum(a[:,:,None],c[:,None,:])>=ROW_DOSE)&(ca[:,:,None]>=COS_MIN)&(cc[:,None,:]>=COS_MIN)
    costs=(a[:,:,None]-goal[:,None,None]).abs()+(c[:,None,:]-goal[:,None,None]).abs()
    fallback=gaps+costs/goal[:,None,None].clamp_min(ROW_DOSE)
    decision=t.where(legal.any(dim=(1,2))[:,None,None],costs.masked_fill(~legal,t.inf),fallback)
    choice=decision.flatten(1).argmin(-1)
    ix=choice//len(GRID);iy=choice%len(GRID)
    j=t.arange(len(b),device=b.device)
    receipt=t.stack((original[:,0],original[:,1],a[j,ix],c[j,iy],gaps[j,ix,iy],legal[j,ix,iy],
                     ca[j,ix],cc[j,iy],norm1,norm2,original_d1.norm(dim=-1),original_d2.norm(dim=-1)),1)
    return p[j,ix],q[j,iy],receipt.cpu().numpy()


def source_cut(model,block,panel,x,q,k,v,o,head,source):
    query=panel['q'][:,0]
    t=model.t
    j=t.arange(len(x),device='cuda')
    slots=[F.M.REGIONS.index('format' if site=='request' else site) for site in source.split('+')]
    slot=F.span_masks(model,panel)[:,slots].any(1)
    values=v.clone()
    values[:,head//model.groups].masked_fill_(slot[...,None],0)
    blocked=model.attention(block,q,k,values,panel['mask'].cuda())
    edited=o.clone()
    edited[j,query,head]=blocked[j,query,head]
    branch=extract(model,block,x,edited,query)
    width=o.shape[-1]
    weight=block.self_attn.o_proj.weight[:,head*width:(head+1)*width].float()
    branch['direction']=t.nn.functional.linear(edited[j,query,head].float()-o[j,query,head].float(),weight)
    # Reinsert the original attention write at the actual tensor location.
    edited[j,query,head]=o[j,query,head]
    assert t.equal(edited,o)
    return branch


def from_attention(model,block,state,after):
    updated=state['after'].clone()
    j,q=state['j'],state['query']
    updated[j,q]=after
    return (updated+block.mlp(block.post_attention_layernorm(updated)))[j,q]


def neuron_restore(model,block,clean,cut,units,mode):
    t=model.t
    j,query=clean['j'],clean['query']
    idx=t.as_tensor(units,device='cuda')
    g=clean['g'] if mode in ('gate','both') else cut['g']
    u=clean['u'] if mode in ('up','both') else cut['u']
    a=cut['a'].clone()
    subset=(j[:,None],query[:,None],idx[None,:])
    a[subset]=t.nn.functional.silu(g[subset])*u[subset]
    result=cut['after']+block.mlp.down_proj(a)
    weight=block.mlp.down_proj.weight[:,idx].float()
    direction=t.nn.functional.linear(a[subset].float()-cut['a'][subset].float(),weight)
    return result,direction


def selected_gua(model,state,units):
    j,q=state['j'],state['query']
    return model.t.stack([state[key][j,q][:,units].float() for key in ('g','u','a')],-1).cpu().numpy()


def full_logits(model,panel,raw):
    t=model.t
    j=t.arange(len(raw),device='cuda')
    return model.logits(raw[j[:,None],panel['q']]).float()


def same_shape_next(model,block,base_input,state,panel,angles):
    t=model.t
    j=t.arange(len(base_input),device='cuda')
    query=panel['q'][:,0]
    changed=base_input.clone()
    changed[j,query]=state
    q,k,v=model.qkv(block,changed,angles)
    o=model.attention(block,q,k,v,panel['mask'].cuda())
    return model.residual(block,changed,o)[j,query]


def dose_percentiles(receipts):
    good=receipts[...,5].astype(bool)
    return dict(count=int(good.sum()),total=int(good.size),coverage=float(good.mean()),
                original_median=np.median(receipts[...,:2].reshape(-1,2),axis=0).tolist(),
                matched_median=np.median(receipts[...,2:4].reshape(-1,2),axis=0).tolist(),
                raw_p95=float(np.quantile(receipts[...,4],.95)),
                eligible_cosine_p05=np.quantile(receipts[...,6:8][good],.05,axis=0).tolist() if good.any() else None,
                extrapolation_median=np.median(receipts[...,2:4][good]/receipts[...,8:10][good],axis=0).tolist() if good.any() else None,
                eligible_p95=float(np.quantile(receipts[...,4][good],.95)) if good.any() else None)
