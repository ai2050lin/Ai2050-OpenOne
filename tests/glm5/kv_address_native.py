"""Read and edit Q and K at the original Qwen3 answer-query boundary."""

from dataclasses import dataclass
from itertools import product
import math

import numpy as np

import kv_address_run as C


HEADS=C.STOCK_HEADS
SCALE=128**-.5


@dataclass(frozen=True)
class Arm:
    name: str
    mode: str
    head: int | None=None
    factor: str | None=None
    slot: int=0
    signal: bool=True
    repair: bool=False


def intervention_arms():
    return [Arm(f"{j}-{part}-{factor}",part,j,factor)
            for j,part,factor in product(range(3),("q","k"),C.FACTORS)]+[
                Arm(f"{j}-rope","rope",j) for j in range(3)]


def value_arms():
    return [Arm(f"{'balanced' if repair else 'original'}-{'sn'[slot]}-{'donor' if signal else 'rotate'}",
                "value",slot=slot,signal=signal,repair=repair)
            for repair,slot,signal in product((False,True),(0,1),(False,True))]


def reo(x,delta,freq):
    """Transplant post-RoPE vectors by changing position phase only."""
    t=__import__("torch")
    angle=delta.float()[...,None]*freq
    s,c=angle.sin(),angle.cos()
    left,right=x.float().chunk(2,-1)
    return t.cat((left*c-right*s,right*c+left*s),-1)


class Reader:
    def __init__(self,task,rows,obs):
        import torch as t
        from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS
        self.t=t
        self.g=task
        self.rows=rows
        self.obs=obs
        self.ridx=t.arange(len(rows),device="cuda")
        self.registry=ALL_ATTENTION_FUNCTIONS
        self.stock=self.registry["sdpa"]
        self.freq=task.model.model.rotary_emb.inv_freq.detach().float()
        self.round_dtype=t.bfloat16
        self.good=t.as_tensor([o["joint"] for o in obs],device="cuda",dtype=t.bool)
        assert self.good.any()
        reference=next(o for o in obs if o["joint"])
        self.index={k:t.tensor([(o if o["joint"] else reference)["index"][k] for o in obs],device="cuda")
                    for k in ("state","note","code")}
        self.query=self.index["code"]-1
        self.begin=int(self.query.min())
        self.end=int(self.query.max())
        self.endpoints = [task.letters[c] for c in "PQ"]
        _,self.inp=task.inputs(rows)
        self.width=self.inp.input_ids.shape[1]
        self.sequence=t.as_tensor([((o["tokens"]+[task.pad]*(self.end+1))[:self.end+1]) for o in obs],device="cuda")
        self.positions=t.stack([self.inp.attention_mask.sum(-1)+i for i in (self.query,self.index["state"],self.index["note"])],-1)
        self.slots=t.stack([self.index[k]+self.width for k in ("state","note")],-1)
        self.active=Arm("no","no")
        self.step_at=-1
        self.registry["sdpa"]=self.attention
        self.reference=None
        self.alt_value=None
        self.balance=None

    def close(self):
        self.registry["sdpa"]=self.stock

    def initialize(self,donor=False):
        t=self.t
        self.step_at=-1
        ret=self.g.first(self.inp)
        cache=ret.past_key_values
        mask=self.inp.attention_mask
        if not donor:
            assert t.equal(ret.logits[:,-1].argmax(-1)[self.good],self.sequence[self.good,0])
        ids=t.as_tensor([o["options"][r["alt"]] for r,o in zip(self.rows,self.obs,strict=True)],device="cuda")
        for i in range(self.begin):
            self.step_at=i
            token=self.sequence[:,i]
            if donor:
                token=t.where(self.good & (self.index["state"]==i),ids,token)
            cache,mask,logits=self.g.step(cache,mask,token)
            if not donor:
                assert t.equal(logits.argmax(-1)[self.good],self.sequence[self.good,i+1])
        if donor:
            self.alt_value=t.stack([cache.layers[l].values[self.ridx,h//4,self.slots[:,0]].float() for l,h in HEADS],1)
        else:
            self.seed=cache
            self.mask=mask
        self.step_at=-1

    def pair(self,module,h,q,k,mask):
        t=self.t
        dots=(q[:, h, 0, :].float()[:,None,:]*k[:,h//4].float()).sum(-1)*module.scaling
        if mask is not None:
            m=mask[:,0,0]
            dots=dots.masked_fill(~m,-t.inf) if m.dtype==t.bool else dots+m
        j=self.slots.clamp_max(k.shape[2]-1)
        alpha=dots.softmax(-1)
        return dots[self.ridx[:,None],j],alpha[self.ridx[:,None],j]

    def cached_values(self,l,h,v):
        return v[self.ridx[:,None],h//4,self.slots.clamp_max(v.shape[2]-1)].float()

    def key_balance(self,module,i,h,q,k,mask,ok):
        t=self.t
        l,alpha=self.pair(module,h,q,k,mask)
        # Center both pair logits on their shared log-mass, preserving other logits.
        target=t.logsumexp(l,dim=-1)-math.log(2)
        direction=q[:,h,0].float()
        coeff=(target[:,None]-l)/(module.scaling*direction.square().sum(-1)[:,None].clamp_min(1e-9))
        g=h//4
        slot=self.slots.clamp_max(k.shape[2]-1)
        prior=k[self.ridx[:,None],g,slot]
        new=(prior.float()+coeff[...,None]*direction[:,None]).to(k.dtype)
        k[self.ridx[:,None],g,slot]=t.where(ok[:,None,None],new,prior)
        self.balance_error[ok,i]=(self.pair(module,h,q,k,mask)[1]-alpha.sum(-1)[:,None]/2).abs()[ok]

    def value_update(self,module,i,h,q,k,v,mask,ok):
        t=self.t
        a=self.active
        pair=self.pair(module,h,q,k,mask)[1]
        sr=self.balance if a.repair else self.reference
        natural=sr["alpha"][:,i]
        low=natural.amin(-1)
        desired=t.minimum(natural.amax(-1),2*low)
        gain=(desired/pair[:,a.slot].clamp_min(1e-12)).clamp_max(2.05)
        vbase=self.reference["values"][:,i,0]
        d=self.alt_value[:,i]-vbase
        if not a.signal:
            d=t.roll(d,(13+7*HEADS[i][0])%127+1,dims=-1)
        g=h//4
        s=self.slots[:,a.slot].clamp_max(v.shape[2]-1)
        base=v[self.ridx,g,s]
        new=(base.float()+d*gain[:,None]).to(v.dtype)
        v[self.ridx,g,s]=t.where(ok[:,None],new,base)
        ideal=desired[:,None]*d
        analytic=pair[:,a.slot,None]*(new.float()-base.float())
        self.target[ok,i]=ideal[ok]
        self.analytic[ok,i]=analytic[ok]
        self.gain[ok,i]=(desired/pair[:,a.slot].clamp_min(1e-12))[ok]

    def attention(self,module,q,k,v,mask,**kw):
        native,weights=self.stock(module,q,k,v,mask,**kw)
        a=self.active
        at=[(i,h) for i,(l,h) in enumerate(HEADS) if module.layer_idx==l]
        if not at or self.step_at<self.begin or a.mode=="no":
            return native,weights
        ok=self.good & self.usable & (self.query==self.step_at)
        if not bool(ok.any()):
            return native,weights
        t=self.t
        pa={i:self.pair(module,h,q,k,mask) for i,h in at}
        if a.mode=="capture":
            for i,h in at:
                sel=self.slots.clamp_max(k.shape[2]-1)
                self.qkv[ok,i,0]=q[ok,h,0].float()
                self.qkv[ok,i,1:]=k[self.ridx[:,None],h//4,sel][ok].float()
                self.values[ok,i]=self.cached_values(module.layer_idx,h,v)[ok]
                self.alpha[ok,i]=pa[i][1][ok]
                self.gap[ok,i]=(pa[i][0][:,0]-pa[i][0][:,1])[ok]
            return native,weights
        q2,k2,v2=q,k,v
        select=[(i,h) for i,h in at if i==a.head] if a.head is not None else at
        if not select:
            return native,weights
        if a.mode in ("k","rope","balance") or a.repair:
            k2=k.clone()
        if a.mode=="q":
            q2=q.clone()
        if a.mode=="value":
            v2=v.clone()
        if a.mode in ("q","k","rope"):
            for i,h in select:
                if a.mode=="q":
                    prior=q[:,h,0]
                    changed=reo(self.donor[:,i,0],self.positions[:,0]-self.donor_pos[:,0],self.freq).to(q.dtype)
                    q2[:,h,0]=t.where(ok[:,None],changed,prior)
                else:
                    g=h//4
                    j=self.slots.clamp_max(k.shape[2]-1)
                    prior=k[self.ridx[:,None],g,j]
                    x,d=(prior,self.positions[:,[2,1]]-self.positions[:,1:]) if a.mode=="rope" else (
                        self.donor[:,i,1:],self.positions[:,1:]-self.donor_pos[:,1:])
                    changed=reo(x,d,self.freq).to(k.dtype)
                    k2[self.ridx[:,None],g,j]=t.where(ok[:,None,None],changed,prior)
        if (a.mode=="balance" or a.repair) and any(i==1 for i,_ in at):
            self.key_balance(module,1,HEADS[1][1],q2,k2,mask,ok)
        if a.mode=="value":
            for i,h in select:
                self.value_update(module,i,h,q2,k2,v2,mask,ok)
        changed,_=self.stock(module,q2,k2,v2,mask,**kw)
        group=t.zeros(32,device="cuda",dtype=t.bool)
        group[[h for _,h in select]]=True
        intermediate=None
        if a.mode=="value" and a.repair and k2 is not k:
            intermediate,_=self.stock(module,q2,k2,v,mask,**kw)
        reference=native if intermediate is None else intermediate
        for i,h in select:
            logits,alpha=self.pair(module,h,q2,k2,mask)
            self.alpha[ok,i]=alpha[ok]
            self.gap[ok,i]=(logits[:,0]-logits[:,1])[ok]
            self.actual[ok,i]=(changed[:,0,h].float()-reference[:,0,h].float())[ok]
        return t.where((ok[:,None]&group[None,:])[:,None,:,None],changed,native),weights

    def replay(self,arm,donor=None,donor_pos=None,usable=None):
        t=self.t
        self.active=arm
        self.donor,self.donor_pos=donor,donor_pos
        self.usable=t.ones_like(self.good) if usable is None else usable
        n=len(self.rows)
        def zero(*dims):
            return t.zeros((n,3,*dims),device="cuda",dtype=t.float32)
        self.qkv=zero(3,128)
        self.values=zero(2,128)
        self.alpha=zero(2)
        self.gap=zero()
        self.target=zero(128)
        self.analytic=zero(128)
        self.actual=zero(128)
        self.gain=zero()
        self.balance_error=zero(2)
        scores=t.zeros(n,2,device="cuda",dtype=t.float32)
        full=t.zeros(n,self.g.model.config.vocab_size,device="cuda",dtype=t.bfloat16)
        cache,mask=self.g.clone(self.seed),self.mask
        for j in range(self.begin,self.end+1):
            self.step_at=j
            cache,mask,log=self.g.step(cache,mask,self.sequence[:,j])
            ok=self.good&(self.query==j)
            full[ok]=log[ok]
            scores[ok]=log[ok][:,self.endpoints].float()
        self.step_at=-1
        return dict(qkv=self.qkv,values=self.values,alpha=self.alpha,gap=self.gap,scores=scores,full=full,
                    target=self.target,analytic=self.analytic,actual=self.actual,gain=self.gain,balance_error=self.balance_error)


def capture(task,rows,obs,sample=False):
    t=__import__("torch")
    r=Reader(task,rows,obs)
    field=C.R.Field(task,[0,1,2,3] if sample else [])
    try:
        r.initialize()
        p=r.replay(Arm("natural","capture"))
        field.stop()
        check=r.replay(Arm("no","no"))
        assert t.equal(p["full"],check["full"])
        target=t.tensor([task.letters[x["code"]] for x in rows],device="cuda")
        assert t.equal(p["full"].argmax(-1)[r.good],target[r.good])
        records={k:p[k].cpu().numpy() for k in ("qkv","values","alpha","gap","scores")}
        if sample:
            records["hidden"]=np.asarray(field.saved,dtype=np.uint16)
        return dict(records,pos=r.positions.cpu().numpy(),good=r.good.cpu().numpy(),freq=r.freq.cpu().numpy())
    finally:
        field.stop()
        r.close()
