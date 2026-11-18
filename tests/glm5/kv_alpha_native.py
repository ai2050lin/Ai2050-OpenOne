"""Native BF16 Value changes and per-head effective write measurement."""

from dataclasses import dataclass
from itertools import product

import numpy as np

import kv_route_run as R
from kv_route_native import Replay, Arm, sample_native


HEADS = R.HEADS
GROUPS = {"h19":HEADS[:1], "trio":HEADS}
SNAPSHOTS = ("state","note","donor")
DIM = 128


@dataclass
class WeightedArm(Arm):
    regime: str = "raw"


def arms():
    return [
        WeightedArm(f"{g}-{regime}-{slot}-{kind}",kind,heads,slot,"answer",regime)
        for (g,heads),regime,slot,kind in product(
            GROUPS.items(),("raw","low","high","write"),("state","note"),("swap","null"))
    ]


class WeightedReplay(Replay):
    def __init__(self,g,ctx):
        super().__init__(g,ctx)
        torch=self.torch
        n=len(ctx["rows"])
        self.baseline_alpha=torch.zeros((n,len(HEADS),2),device="cuda",dtype=torch.float32)
        self.write_records={}
        self.live_rows={}
        self.read_frame={}
        self.current_vectors=None
        self.current_stats=None

    def prepare_arm(self,arm):
        t=self.torch
        n=len(self.good)
        self.current_vectors=t.zeros((n,len(HEADS),DIM),device="cuda",dtype=t.float32)
        self.current_stats=t.zeros((n,len(HEADS),4),device="cuda",dtype=t.float32)
        result = self.path(arm)
        self.write_records[arm.name]=self.current_vectors.cpu().numpy()
        self.live_rows[arm.name]=self.current_stats.cpu().tolist()
        return result

    def alpha_at_slots(self,module,h,q,k,mask):
        t=self.torch
        index=self.ctx["decoy"].clamp_max(k.shape[2]-1)
        dots=(q[:,h,0,:].float()[:,None,:]*k[:,h//4].float()).sum(dim=-1)*module.scaling
        if mask is not None:
            broadcast=mask[:,0,0]
            dots=dots.masked_fill(~broadcast,-t.inf) if broadcast.dtype==t.bool else dots+broadcast
        p=dots.softmax(-1)
        return t.stack((p[:,self.ctx["source"]],p[self.ridx,index]),-1)

    def source_value(self,v,source):
        if isinstance(source,int):
            return v[:,:,source]
        return v[self.ridx,:,source]

    def attention(self,module,q,k,v,mask,**kw):
        native,weights=self.stock(module,q,k,v,mask,**kw)
        l=module.layer_idx
        looking = self.good & (self.clock["answer"]==self.at)
        a=self.active_arm
        if not self.at or (not self.probe and not a.kind):
            return native,weights
        matched=[(i,h) for i,(ll,h) in enumerate(HEADS) if ll==l
                 and (self.probe or h in self.mask_for_layer.get(l,()))]
        if not matched or not bool(looking.any()):
            return native,weights
        t=self.torch
        if self.probe:
            for i,h in matched:
                self.baseline_alpha[looking,i]=self.alpha_at_slots(module,h,q,k,mask)[looking]
            return native,weights

        is_note=a.source=="note"
        slot_index=int(is_note)
        source=self.ctx["decoy"].clamp_max(v.shape[2]-1) if is_note else self.ctx["source"]
        if is_note:
            assert bool((self.ctx["decoy"][looking]<v.shape[2]-1).all())
        old=self.source_value(v,source)
        retained=self.note[l] if is_note else self.state[l]
        assert t.equal(old[looking],retained[looking])
        alt=v.clone() if a.regime!="write" else None
        direct=native.clone() if a.regime=="write" else None
        for i,h in matched:
            g=h//4
            live=self.alpha_at_slots(module,h,q,k,mask)
            alpha=live[:,slot_index]
            x=self.state[l][:,g].float()
            d=self.ctx["v_alt"][l][:,g].float()-x
            if a.kind=="null":
                d=t.roll(d,shifts=(13+7*l)%127+1,dims=-1)
            reference=self.baseline_alpha[:,i]
            small,big=reference.amin(-1),reference.amax(-1)
            target=(small if a.regime=="low" else t.minimum(big,2*small)) if a.regime in ("low","high") else alpha
            scale=(target/alpha.clamp_min(1e-12)).clamp_max(2.05) if a.regime in ("low","high") else t.ones_like(alpha)
            base=old[:,g]
            candidate=(base.float()+scale[:,None]*d).to(base.dtype)
            if direct is not None:
                proposed=(native[:,0,h].float()+alpha[:,None]*d).to(native.dtype)
                direct[:,0,h]=t.where(looking[:,None],proposed,native[:,0,h])
            else:
                original=alt[self.ridx,g,source]
                alt[self.ridx,g,source]=t.where(looking[:,None],candidate,original)
            self.current_stats[looking,i]=t.stack((live[:,0],live[:,1],target,scale),-1)[looking]
        if direct is None:
            changed,_=self.stock(module,q,k,alt,mask,**kw)
            changed=t.where(self.query_mask(looking,[h for _,h in matched]),changed,native)
        else:
            changed=direct
        for i,h in matched:
            self.current_vectors[looking,i]=(changed[:,0,h].float()-native[:,0,h].float())[looking]
        self.count[looking]+=len(matched)
        return changed,weights

    def source_arrays(self):
        def collect(panel):
            return self.torch.stack([panel[l][:,h//4].float() for l,h in HEADS],dim=1).cpu().numpy()
        return dict(zip(SNAPSHOTS,map(collect,(self.state,self.note,self.ctx["v_alt"])),strict=True))


def measure(g,rows,baseline,key,sample=False):
    ctx=g.natural(rows,retain=True,sample=range(4) if sample else ())
    assert ctx["obs"]==baseline
    reader=WeightedReplay(g,ctx)
    try:
        sham=reader.state_at_clocks()
        assert reader.path(Arm("natural"))==sham
        if sample:
            sample_native(ctx,reader,key)
        sample_files={f"native-{key}.npz":R.sha(R.OUT/f"native-{key}.npz")} if sample else {}
        paths={a.name:reader.prepare_arm(a) for a in arms()}
        stats=reader.live_rows
        file=R.Base.Previous.check_write(R.OUT/f"writes-{key}.npz")
        np.savez_compressed(file,**reader.source_arrays(),sham_alpha=reader.baseline_alpha.cpu().numpy(),**reader.write_records)
        return dict(ids=[r["id"] for r in rows],natural=sham,paths=paths,stats=stats,exact_all_vocab=True,
                    fields=sample_files,write_file=file.name,write_sha=R.sha(file),
                    prompt_hash=ctx["prompt_hash"],after_state=ctx["after_state"],
                    source=ctx["source"],note=ctx["decoy"].cpu().tolist(),answer_clock=reader.clock["answer"].cpu().tolist())
    finally:
        reader.close()
