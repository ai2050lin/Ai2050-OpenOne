"""Smooth local derivatives anchored at native values, with explicit causal K/V.

These are parameter-computation identities, not learned semantic laws or the
derivative of the discontinuous BF16 rounding operation.
"""
import torch


def linear(weight,direction):
    return torch.einsum('od,...dk->...ok',weight.to(direction.dtype),direction)


def rms_jvp(value,direction,weight,eps):
    value=value.to(direction.dtype);inv=torch.rsqrt(value.square().mean(-1,keepdim=True)+eps)
    inner=(value.unsqueeze(-1)*direction).mean(-2,keepdim=True)
    return weight.to(direction.dtype).unsqueeze(-1)*inv.unsqueeze(-1)*(direction-value.unsqueeze(-1)*inner*inv.square().unsqueeze(-1))


def rotate_direction(value):
    a,b=value.chunk(2,dim=-2)
    return torch.cat((-b,a),dim=-2)


def last_attention_jvp(q,k,v,dq,dk_self,dv_self,last,scale):
    """q [B,H,D], K/V [B,H,T,D]; tangents [B,H,D,K]."""
    scores=torch.einsum('bhd,bhtd->bht',q,k)*scale
    visible=torch.arange(k.shape[2],device=k.device)[None,:]<=last[:,None]
    scores=scores.masked_fill(~visible[:,None],float('-inf'));p=scores.softmax(-1)
    ds=torch.einsum('bhdk,bhtd->bhtk',dq,k)*scale
    self_term=torch.einsum('bhd,bhdk->bhk',q,dk_self)*scale
    batch=torch.arange(q.shape[0],device=q.device)
    ds[batch,:,last,:]+=self_term
    dp=p.unsqueeze(-1)*(ds-(p.unsqueeze(-1)*ds).sum(2,keepdim=True))
    result=torch.einsum('bhtk,bhtd->bhdk',dp,v)+p[batch,:,last].unsqueeze(-1).unsqueeze(-1)*dv_self
    baseline=torch.einsum('bht,bhtd->bhd',p,v)
    return result,baseline


def silu_product_jvp(g,v,dg,dv):
    s=torch.sigmoid(g);a=g*s
    return (s*(1+g*(1-s))*v).unsqueeze(-1)*dg+a.unsqueeze(-1)*dv


def local_step(layer,before,cap,direction,positions,pe,roots=None):
    """First eight columns: full/no-attn/no-MLP/residual for dose0, then dose2.
    Any later columns follow full propagation (individual-unit linearization).

    All baseline states are natural observations. Historical prefix K/V are fixed
    by causality because interventions occur only at the final visible token.
    """
    batch=torch.arange(len(positions),device=before.device);last=before[batch,positions].float()
    d=direction.float();dtype=d.dtype
    dx=rms_jvp(last,d,layer.input_layernorm.weight,layer.input_layernorm.variance_epsilon)
    attn=layer.self_attn;hd=attn.head_dim;nq=attn.q_proj.out_features//hd;nkv=attn.k_proj.out_features//hd;repeat=nq//nkv
    qraw=cap['qraw'][batch,positions].reshape(len(batch),nq,hd).float()
    kraw=cap['kraw'][batch,positions].reshape(len(batch),nkv,hd).float()
    dq=linear(attn.q_proj.weight,dx).reshape(len(batch),nq,hd,-1)
    dk=linear(attn.k_proj.weight,dx).reshape(len(batch),nkv,hd,-1)
    dv=linear(attn.v_proj.weight,dx).reshape(len(batch),nkv,hd,-1)
    dq=rms_jvp(qraw,dq,attn.q_norm.weight,attn.q_norm.variance_epsilon)
    dk=rms_jvp(kraw,dk,attn.k_norm.weight,attn.k_norm.variance_epsilon)
    cos,sin=pe;cs=cos[0,positions].float()[:,None,:,None];sn=sin[0,positions].float()[:,None,:,None]
    dq=dq*cs+rotate_direction(dq)*sn;dk=dk*cs+rotate_direction(dk)*sn
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    q_all,k_all=apply_rotary_pos_emb(cap['qnorm'].transpose(1,2),cap['knorm'].transpose(1,2),cos,sin)
    q=q_all[batch,:,positions].float();k=k_all.float().repeat_interleave(repeat,dim=1)
    v=cap['vraw'].reshape(len(batch),before.shape[1],nkv,hd).transpose(1,2).float().repeat_interleave(repeat,dim=1)
    da,approx=last_attention_jvp(q,k,v,dq,dk.repeat_interleave(repeat,dim=1),dv.repeat_interleave(repeat,dim=1),positions,attn.scaling)
    da=linear(attn.o_proj.weight,da.reshape(len(batch),nq*hd,-1))
    da[:,:,[1,3,5,7]]=0
    dr=d+da
    residual=cap['mlp_residual'][batch,positions].float()
    dz=rms_jvp(residual,dr,layer.post_attention_layernorm.weight,layer.post_attention_layernorm.variance_epsilon)
    dg=linear(layer.mlp.gate_proj.weight,dz);dv_=linear(layer.mlp.up_proj.weight,dz)
    g=cap['G'][batch,positions].float();v_=cap['V'][batch,positions].float()
    dam=silu_product_jvp(g,v_,dg,dv_);dm=linear(layer.mlp.down_proj.weight,dam);dm[:,:,[2,3,6,7]]=0
    next_=dr+dm
    selected=None if roots is None else torch.stack([dg[:,roots],dv_[:,roots],dam[:,roots]],dim=2)
    approximate_output=torch.nn.functional.linear(approx.reshape(len(batch),-1),attn.o_proj.weight.float())
    native=cap['attention'][batch,positions].float()
    numerical=dict(attention_fp32_reconstruction_relative_L2=float((approximate_output-native).norm()/native.norm().clamp_min(1e-12)))
    return next_,selected,numerical
