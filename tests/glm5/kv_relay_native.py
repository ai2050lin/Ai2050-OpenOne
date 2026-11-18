"""Instrument and locally predict the real L20→L21 route on cached answer replay."""

from dataclasses import dataclass
from itertools import product

import numpy as np

import kv_address_native as Old
import kv_relay_run as C


HEADS=C.HEADS
CAP=2.05
CONDITIONS=tuple(product(range(2),(False,True)))


@dataclass(frozen=True)
class Arm:
    name: str
    mode: str
    cell: int=0
    source: bool=False
    mlp: str="live"
    query: str="live"
    key: str="live"
    trio: bool=False


def arms(i):
    return (
        Arm("source","edit",i,True),
        Arm("mlp-block","edit",i,True,mlp="natural"),
        Arm("mlp-restore","edit",i,mlp="edited"),
        Arm("units-block","edit",i,True,mlp="units-natural"),
        Arm("units-restore","edit",i,mlp="units-edited"),
        Arm("null-units-block","edit",i,True,mlp="null-natural"),
        Arm("null-units-restore","edit",i,mlp="null-edited"),
        Arm("q-block","edit",i,True,query="natural"),
        Arm("q-restore","edit",i,query="edited"),
        Arm("k-block","edit",i,True,key="natural"),
        Arm("k-restore","edit",i,key="edited"),
        Arm("qk-block","edit",i,True,query="natural",key="natural"),
        Arm("qk-restore","edit",i,query="edited",key="edited"),
        Arm("trio","edit",i,True,trio=True),
        Arm("trio-mlp","edit",i,True,mlp="natural",trio=True),
        Arm("trio-q19","edit",i,True,query="h19",trio=True),
        Arm("trio-q","edit",i,True,query="natural",trio=True),
        Arm("trio-qk","edit",i,True,query="natural",key="natural",trio=True),
    )


class Relay(Old.Reader):
    def __init__(self,g,rows,obs,include):
        super().__init__(g,rows,obs)
        t=self.t
        self.live=t.as_tensor(include,device="cuda",dtype=t.bool)
        assert bool((self.index["state"][self.live]<self.begin).all())
        self.active=Arm("no","no")
        self.l20,self.l21=[g.model.model.layers[i] for i in (20,21)]
        n=len(rows)
        self.history_size=self.width+self.end+1
        mk=lambda *d:t.zeros((n,*d),dtype=t.bfloat16,device="cuda")
        self.x=mk(2560)
        self.a=mk(2560)
        self.z=mk(9728)
        self.m=mk(2560)
        self.prev=mk(2560)
        self.anchors=mk(4,2560)
        self.source_heads=mk(4,128)
        self.cs=mk(2,128)
        self.early_alpha=t.zeros((n,2),device="cuda")
        self.early_value=mk(2,128)
        self.received=mk(2,2,128)
        self.bank=mk(2,self.history_size,128)
        self.visible=t.zeros((n,self.history_size),device="cuda",dtype=t.bool)
        self.handles=[
            self.l20.input_layernorm.register_forward_pre_hook(self.input20),
            self.l20.post_attention_layernorm.register_forward_pre_hook(self.mid20),
            self.l20.mlp.down_proj.register_forward_pre_hook(self.units20),
            self.l20.mlp.register_forward_hook(self.out20),
            self.l21.input_layernorm.register_forward_pre_hook(self.arrive21),
            self.l21.self_attn.register_forward_pre_hook(self.rope21,with_kwargs=True),
        ]
        self.units=None
        self.forecast={}

    def close(self):
        for h in self.handles:
            h.remove()
        super().close()

    def mask_live(self):
        return self.live&(self.query==self.step_at) if self.step_at>=self.begin and self.active.mode!="no" else None

    def input20(self,mod,args):
        ok=self.mask_live()
        if ok is not None and self.active.mode=="natural":
            self.x[ok]=args[0][ok,0]

    def mid20(self,mod,args):
        ok=self.mask_live()
        if ok is not None and self.active.mode=="natural":
            self.a[ok]=args[0][ok,0]

    def units20(self,mod,args):
        ok=self.mask_live()
        if ok is None:
            return None
        if self.active.mode=="natural":
            self.z[ok]=args[0][ok,0]
        elif self.active.mlp.startswith(("units-","null-")):
            original=args[0].clone()
            source=self.z if self.active.mlp.endswith("-natural") else self.z_edit[:,self.active.cell]
            units=self.null_units if self.active.mlp.startswith("null-") else self.units
            idx=self.ridx[ok]
            original[idx[:,None],0,units]=source[idx[:,None],units]
            return (original,)
        return None

    def out20(self,mod,args,out):
        ok=self.mask_live()
        if ok is None:
            return None
        if self.active.mode=="natural":
            self.m[ok]=out[ok,0]
        elif self.active.mlp in ("natural","edited"):
            target=self.m if self.active.mlp=="natural" else self.m_edit[:,self.active.cell]
            return self.t.where(ok[:,None,None],target[:,None],out)
        return None

    def arrive21(self,mod,args):
        ok=self.mask_live()
        if ok is None:
            return
        if self.active.mode=="natural":
            self.prev[ok]=args[0][ok,0]
        else:
            self.arrive[ok]=args[0][ok,0]

    def rope21(self,mod,args,kwargs):
        ok=self.mask_live()
        if ok is None or self.active.mode!="natural":
            return
        cs=kwargs["position_embeddings"]
        for i,x in enumerate(cs):
            self.cs[ok,i]=x[ok,0]

    def read_alpha(self, q,bank,mask):
        x=(q.float()[...,None,:]*bank.float()).sum(-1)*Old.SCALE
        x=x.masked_fill(~mask[:,None],-self.t.inf)
        x=self.t.where(self.live[:,None,None],x,0.)
        normalized=x.softmax(-1)
        idx=self.slots[:,None,:].expand(-1,2,-1)
        take=normalized.gather(-1,idx)
        gap=x.gather(-1,idx)
        return take*self.live[:,None,None],gap[...,0]-gap[...,1],normalized

    def earlier(self,module,h,q,k,v,mask,ok,cell):
        t=self.t
        slot,signal=CONDITIONS[cell]
        old=v[self.ridx,h//4,self.slots[:,slot].clamp_max(v.shape[2]-1)]
        donor=self.alt_value[:,2]-self.early_value[:,0].float()
        if not signal:
            donor=t.roll(donor,(13+7*20)%127+1,dims=-1)
        target=t.minimum(self.early_alpha.amax(-1),2*self.early_alpha.amin(-1))
        live=self.pair(module,h,q,k,mask)[1][:,slot]
        scale=(target/live.clamp_min(1e-12)).clamp_max(CAP)
        val=(old.float()+scale[:,None]*donor).to(v.dtype)
        vnew=v.clone()
        vnew[self.ridx,h//4,self.slots[:,slot].clamp_max(v.shape[2]-1)]=t.where(ok[:,None],val,old)
        return vnew,donor,target,live,val.float()-old.float()

    def later_value(self,module,h,idx,q,k,v,mask,ok):
        t=self.t
        slot,signal=CONDITIONS[self.active.cell]
        s=self.slots[:,slot].clamp_max(v.shape[2]-1)
        old=v[self.ridx,h//4,s]
        d=self.alt_value[:,idx]-self.receiver_values[:,idx,0].float()
        if not signal:
            d=t.roll(d,(13+7*21)%127+1,dims=-1)
        natural=self.alpha0[:,idx]
        target=t.minimum(natural.amax(-1),2*natural.amin(-1))
        live=self.pair(module,h,q,k,mask)[1][:,slot]
        gain=target/live.clamp_min(1e-12)
        val=(old.float()+gain.clamp_max(CAP)[:,None]*d).to(old.dtype)
        v[self.ridx,h//4,s]=t.where(ok[:,None],val,old)
        self.want[ok,idx]=(target[:,None]*d)[ok]
        self.analytic[ok,idx]=(live[:,None]*(val.float()-old.float()))[ok]
        self.gain[ok,idx]=gain[ok]

    def attention(self,module,q,k,v,mask,**kw):
        native,weights=self.stock(module,q,k,v,mask,**kw)
        ok=self.mask_live()
        l=module.layer_idx
        if l not in (20,21) or ok is None or not bool(ok.any()):
            return native,weights
        t=self.t
        a=self.active
        if l==20:
            h=5
            if a.mode=="natural":
                self.early_alpha[ok]=self.pair(module,h,q,k,mask)[1][ok]
                self.early_value[ok]=self.cached_values(l,h,v)[ok].to(v.dtype)
                for j in range(4):
                    vn,*_=self.earlier(module,h,q,k,v,mask,ok,j)
                    changed,_=self.stock(module,q,k,vn,mask,**kw)
                    mixed=native.clone()
                    mixed[ok,0,h]=changed[ok,0,h]
                    derived=module.o_proj(mixed.reshape(len(ok),1,-1).contiguous())
                    self.anchors[ok,j]=(self.x[ok]+derived[ok,0])
                    self.source_heads[ok,j]=changed[ok,0,h]
                return native,weights
            if not a.source:
                return native,weights
            vn,d,target,live,dv=self.earlier(module,h,q,k,v,mask,ok,a.cell)
            changed,_=self.stock(module,q,k,vn,mask,**kw)
            mixed=native.clone()
            mixed[ok,0,h]=changed[ok,0,h]
            self.want[ok,2]=(target[:,None]*d)[ok]
            self.analytic[ok,2]=(live[:,None]*dv)[ok]
            self.actual[ok,2]=(changed[ok,0,h].float()-native[ok,0,h].float())
            self.gain[ok,2]=(target/live.clamp_min(1e-12))[ok]
            return mixed,weights

        h=[19,25]
        group=[j//4 for j in h]
        if a.mode=="natural":
            nk=k[:,group]
            self.bank[ok,:,:nk.shape[2]]=nk[ok]
            masks=mask[:,0,0] if mask is not None else self.t.ones_like(nk[:,0,:,0],dtype=t.bool)
            self.visible[ok,:nk.shape[2]]=(masks if masks.dtype==t.bool else masks>=0)[ok]
            self.received[ok,:,0]=q[ok][:,h,0]
            self.received[ok,:,1]=nk[ok,:,-1]
            self.receiver_values[ok]=t.stack([self.cached_values(l,j,v) for j in h],1)[ok].to(v.dtype)
            return native,weights

        # Verify only the newly appended Key can change on this query.
        cached=k[:,group,:-1]
        assert t.equal(cached[ok],self.bank[ok,:,:k.shape[2]-1])
        observed=t.stack((q[:,h,0],k[:,group,-1]),dim=2)
        self.qk_before[ok]=observed[ok]
        q2,k2,v2=q,k,v
        if a.query!="live":
            qs=self.received[:,:,0] if a.query in ("natural","h19") else self.forecast["full"]["qk"][:,a.cell,:,0]
            q2=q.clone()
            for i in ((0,) if a.query=="h19" else range(2)):
                q2[ok,h[i],0]=qs[ok,i]
        if a.key!="live":
            ks=self.received[:,:,1] if a.key=="natural" else self.forecast["full"]["qk"][:,a.cell,:,1]
            k2=k.clone()
            for i in range(2):
                k2[ok,group[i],-1]=ks[ok,i]
        has_address=q2 is not q or k2 is not k
        if a.trio:
            v2=v.clone()
            for i,j in enumerate(h):
                self.later_value(module,j,i,q2,k2,v2,mask,ok)
        if has_address or a.trio:
            ref=self.stock(module,q2,k2,v,mask,**kw)[0] if has_address else native
            changed=self.stock(module,q2,k2,v2,mask,**kw)[0] if a.trio else ref
            self.actual[ok,:2]=(changed[ok][:,0,h].float()-ref[ok][:,0,h].float())
        else:
            changed=native
        alpha=[]
        for i,j in enumerate(h):
            _,a1=self.pair(module,j,q2,k2,mask)
            alpha.append(a1)
        self.alpha_now[ok]=t.stack(alpha,1)[ok]
        self.qk_after[ok]=t.stack((q2[:,h,0],k2[:,group,-1]),dim=2)[ok]
        if changed is native:
            return native,weights
        mixed=native.clone()
        for j in h:
            mixed[ok,0,j]=changed[ok,0,j]
        return mixed,weights

    def reset(self):
        t=self.t
        n=len(self.rows)
        self.receiver_values=t.zeros((n,2,2,128),device="cuda",dtype=t.bfloat16)
        self.arrive=t.zeros_like(self.a)
        self.qk_before=t.zeros_like(self.received)
        self.qk_after=t.zeros_like(self.received)
        self.alpha_now=t.zeros((n,2,2),device="cuda")
        self.want=t.zeros((n,3,128),device="cuda")
        self.analytic=t.zeros_like(self.want)
        self.actual=t.zeros_like(self.want)
        self.gain=t.zeros((n,3),device="cuda")

    def step_run(self,arm,reset=True):
        t=self.t
        self.active=arm
        n=len(self.rows)
        if reset:
            self.reset_edit()
        cache,mask=self.g.clone(self.seed),self.mask
        full=t.zeros(n,self.g.model.config.vocab_size,dtype=t.bfloat16,device="cuda")
        for i in range(self.begin,self.end+1):
            self.step_at=i
            cache,mask,log=self.g.step(cache,mask,self.sequence[:,i])
            sel=self.live&(self.query==i)
            full[sel]=log[sel]
        self.step_at=-1
        return full

    def reset_edit(self):
        for k in ("arrive","qk_before","qk_after","alpha_now","want","analytic","actual","gain"):
            getattr(self,k).zero_()

    def natural(self):
        self.reset()
        self.initialize()
        self.initialize(donor=True)
        self.base_score=self.step_run(Arm("natural","natural"))
        shadow=self.step_run(Arm("shadow","no"))
        assert self.t.equal(self.base_score,shadow)
        self.alpha0,self.gap0,_=self.alpha_at_qk(self.received)
        self.direct_sanity()

    def project(self,y):
        t=self.t
        n=len(y)
        x=self.l21.input_layernorm(y[:,None])
        m=self.l21.self_attn
        q=m.q_norm(m.q_proj(x).view(n,1,32,128))[:,0,[19,25]]
        k=m.k_norm(m.k_proj(x).view(n,1,8,128))[:,0,[4,6]]
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        q,k=apply_rotary_pos_emb(q[:,:,None],k[:,:,None],self.cs[:,0,None],self.cs[:,1,None])
        return t.stack((q[:,:,0],k[:,:,0]),dim=2)

    def alpha_at_qk(self,combined):
        mm=self.bank.clone()
        mm[self.ridx[:,None],self.t.arange(2,device="cuda"),(self.slots[:,0]*0+self.width+self.query)[:,None]]=combined[:,:,1]
        return self.read_alpha(combined[:,:,0],mm,self.visible)

    def direct_sanity(self):
        t=self.t
        assert t.equal((self.a+self.m)[self.live],self.prev[self.live])
        assert t.equal(self.project(self.prev)[self.live],self.received[self.live])
        exp=t.tensor([self.g.letters[r["code"]] for r in self.rows],device="cuda")
        assert t.equal(self.base_score.argmax(-1)[self.live],exp[self.live])

    def prepare(self,units=None,random=None):
        t=self.t
        self.units=None if units is None else t.as_tensor(units,device="cuda",dtype=t.long)[None]
        random=None if random is None else t.as_tensor(random,device="cuda",dtype=t.long)
        self.null_units=None if random is None else random[None]
        mlp=self.l20.mlp
        paths={"full":[],"direct":[],"mlp":[]}
        if units is not None:
            paths.update(sparse=[],random=[])
        zz=[]
        ms=[]
        for j in range(4):
            h=self.anchors[:,j]
            x=self.l20.post_attention_layernorm(h[:,None])
            z=(mlp.act_fn(mlp.gate_proj(x))*mlp.up_proj(x))[:,0]
            m=mlp.down_proj(z[:,None])[:,0]
            zz.append(z)
            ms.append(m)
            states=dict(full=h+m,direct=h+self.m,mlp=self.a+m)
            if units is not None:
                for name,ids in (("sparse",self.units.flatten()),("random",random)):
                    mixed=self.z.clone()
                    mixed[:,ids]=z[:,ids]
                    states[name]=h+mlp.down_proj(mixed[:,None])[:,0]
            for name,r in states.items():
                qk=self.project(r)
                aa,gap,_=self.alpha_at_qk(qk)
                paths[name].append(dict(qk=qk,alpha=aa,gap=gap))
        self.z_edit=t.stack(zz,1)
        self.m_edit=t.stack(ms,1)
        self.forecast={a:{k:t.stack([j[k] for j in v],1) for k in v[0]} for a,v in paths.items()}

    def edited(self,arm):
        full=self.step_run(arm)
        return dict(score=full[:,self.endpoints].float().cpu().numpy(),alpha=self.alpha_now.cpu().numpy().copy(),
                    before=self.qk_before.float().cpu().numpy(),after=self.qk_after.float().cpu().numpy(),
                    arrive=self.arrive.cpu().float().numpy(),dose=self.dose() if arm.trio else None)

    def dose(self):
        return {n:getattr(self,n).cpu().numpy().copy() for n in ("want","analytic","actual","gain")}
