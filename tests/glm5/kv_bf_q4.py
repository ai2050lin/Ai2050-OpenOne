"""Cross input and output rounding interventions at the Qwen3-4B H25 write."""

from dataclasses import dataclass

import numpy as np

import kv_relay_native as Native
import kv_bf_run as C


METHODS=("native","once","late-round","late")


@dataclass(frozen=True)
class Arm(Native.Arm):
    precision: str="native"


class Reader(Native.Relay):
    def __init__(self,*a):
        super().__init__(*a)
        self.delta=self.t.zeros((len(self.rows),128),dtype=self.t.float32,device="cuda")
        self.output_hook=self.l21.self_attn.o_proj.register_forward_hook(self.adjust)

    def close(self):
        self.output_hook.remove()
        super().close()

    def precise_arm(self):
        return self.active.mode=="edit" and getattr(self.active,"precision","native")!="native"

    def later_value(self,module,h,idx,q,k,v,mask,ok):
        if idx!=1 or not self.precise_arm():
            return super().later_value(module,h,idx,q,k,v,mask,ok)
        t=self.t
        self.delta.zero_()
        slot,signal=Native.CONDITIONS[self.active.cell]
        s=self.slots[:,slot].clamp_max(v.shape[2]-1)
        old=v[self.ridx,h//4,s]
        d=self.alt_value[:,idx]-self.receiver_values[:,idx,0].float()
        if not signal:
            d=t.roll(d,(13+7*21)%127+1,dims=-1)
        natural=self.alpha0[:,idx]
        target=t.minimum(natural.amax(-1),2*natural.amin(-1))
        live=self.pair(module,h,q,k,mask)[1][:,slot]
        gain=target/live.clamp_min(1e-12)
        scaled=gain.clamp_max(Native.CAP)[:,None]*d
        if self.active.precision=="late-round":
            scaled=(old.float()+scaled).to(old.dtype).float()-old.float()
        self.delta[ok]=(live[:,None]*scaled)[ok]
        self.want[ok,idx]=(target[:,None]*d)[ok]
        self.analytic[ok,idx]=self.delta[ok]
        self.gain[ok,idx]=gain[ok]

    def attention(self,module,q,k,v,mask,**kw):
        z,weights=super().attention(module,q,k,v,mask,**kw)
        if module.layer_idx!=21 or not self.precise_arm():
            return z,weights
        ok=self.mask_live()
        if ok is None or not bool(ok.any()):
            return z,weights
        position=z[ok,0,25].float()
        if self.active.precision=="once":
            z=z.clone()
            after=(position+self.delta[ok]).to(z.dtype)
            z[ok,0,25]=after
            self.actual[ok,1]=after.float()-position
        else:
            self.actual[ok,1]=self.delta[ok]
        return z,weights

    def adjust(self,m,args,out):
        if not self.precise_arm() or self.active.precision=="once":
            return None
        ok=self.mask_live()
        if ok is None or not bool(ok.any()):
            return None
        t=self.t
        w=m.weight[:,25*128:26*128].float()
        shift=self.delta@w.T
        return t.where(ok[:,None,None],(out.float()+shift[:,None]).to(out.dtype),out)


def variants(cell):
    for method in METHODS:
        yield Arm(method,"edit",cell,True,query="natural",trio=True,precision=method)
    yield Arm("dynamic","edit",cell,True,trio=True)


def run_batch(task,rows,obs,good,arm_names=None):
    r=Reader(task,rows,obs,good)
    try:
        r.natural()
        out={}
        for j in range(4):
            for arm in variants(j):
                if arm_names is not None and arm.name not in arm_names:
                    continue
                x=r.edited(arm)
                out.setdefault(arm.name,[]).append(dict(alpha=x["alpha"],scores=x["score"],dose=x["dose"]))
        records=dict(
            ids=np.array([x["id"] for x in rows]),good=good,
            native_score=r.base_score[:,r.endpoints].float().cpu().numpy(),
            natural_alpha=r.alpha0.cpu().numpy(),
            value_state=r.receiver_values[:,:,0].float().cpu().numpy(),
            value_note=r.receiver_values[:,:,1].float().cpu().numpy(),
            value_donor=r.alt_value[:,:2].float().cpu().numpy())
        return records,out
    finally:
        r.close()
