"""Native source and clock interventions, same-generation replay, and subset rescue."""
from dataclasses import dataclass
from itertools import combinations
import sys
import time

import kv_route_run as R


@dataclass
class Arm:
    name: str
    kind: str=""
    group: tuple | None=()
    source: str="state"
    clock: str="answer"


SUBSETS={ "".join(map(str,x)):tuple(R.HEADS[j] for j in x)
          for k in (1,2,3) for x in combinations(range(3),k)}


def specs():
    arms=[Arm("natural"),Arm("whole","swap",None),Arm("remaining","swap",tuple((l,h) for l in range(36)
                   for h in range(32) if (l,h) not in R.HEADS)),Arm("whole_null","null",None)]
    for source in ("state","note"):
        for clock in ("early","answer"):
            if (source,clock)==("state","answer"):
                continue
            arms.extend(Arm(f"{mode}-{source}-{clock}",mode,R.HEADS,source,clock) for mode in ("swap","null"))
    for key,group in SUBSETS.items():
        arms.extend(Arm(f"{kind}-{key}",kind,group) for kind in ("swap","null","zero","zero_null","recover","recover_null"))
    return arms


class Replay:
    def __init__(self,g,ctx):
        import torch
        from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS
        self.g=g
        self.ctx=ctx
        self.registry=ALL_ATTENTION_FUNCTIONS
        self.stock=self.registry["sdpa"]
        self.torch=torch
        self.ridx=torch.arange(len(ctx["rows"]),device="cuda")
        self.good=torch.as_tensor([x["joint"] for x in ctx["obs"]],device="cuda",dtype=torch.bool)
        assert self.good.any()
        early_default=next(x["note_index"]+1 for x in ctx["obs"] if x["joint"])
        late_default=next(x["answer_index"]-1 for x in ctx["obs"] if x["joint"])
        self.clock={label:torch.as_tensor(indices,device="cuda") for label,indices in (
            ("early",[x["note_index"]+1 if x["joint"] else early_default for x in ctx["obs"]]),
            ("answer",[x["answer_index"]-1 if x["joint"] else late_default for x in ctx["obs"]]))}
        self.minimum={k:int(v.min()) for k,v in self.clock.items()}
        self.last=int(self.clock["answer"].max())
        self.answer_ids=[g.letters[z] for z in ("P","Q")]
        self.state=[layer.values[:,:,ctx["source"]].clone() for layer in ctx["c1"].layers]
        self.note=ctx["v_note"]
        self.fingerprint=R.state_digest(ctx["c1"].layers[-1].values)
        self.arrays={k:[] for k in ("state_swap","state_null","note_swap","note_null","zero","zero_null","recover_null")}
        self.error={k:0. for k in ("state_null","note_swap","note_null","zero_null","recover_null")}
        for l,(x,n,z) in enumerate(zip(self.state,self.note,ctx["v_alt"],strict=True)):
            d=z.float()-x.float()
            shift=(13+7*l)%127+1
            rot=lambda t:torch.roll(t,shifts=shift,dims=-1)
            self.arrays["state_swap"].append(z)
            for key,origin,delta,expected in (
                    ("state_null",x,rot(d),d),("note_swap",n,d,d),("note_null",n,rot(d),d),
                    ("zero_null",x,rot(-x.float()),-x.float()),
                    ("recover_null",torch.zeros_like(x),rot(x.float()),x.float())):
                altered=(origin.float()+delta).to(origin.dtype)
                self.arrays[key].append(altered)
                goal=expected.norm(dim=-1)
                actual=(altered.float()-origin.float()).norm(dim=-1)
                error=((goal-actual).abs()/goal.clamp_min(1e-8))[self.good]
                self.error[key]=max(self.error[key],float(error.max()))
            self.arrays["zero"].append(torch.zeros_like(x))
        assert max(self.error.values())<=.03,self.error
        self.sham={}
        self.active_arm=Arm("natural")
        self.at=0
        self.mask_for_layer={}
        self.probe=False
        self.quality={}
        self.registry["sdpa"]=self.attention

    def close(self):
        self.registry["sdpa"]=self.stock
        assert R.state_digest(self.ctx["c1"].layers[-1].values)==self.fingerprint
        assert all(self.torch.equal(x,l.values[:,:,self.ctx["source"]]) for x,l in zip(self.state,self.ctx["c1"].layers,strict=True))

    def mask_for(self,heads):
        result={}
        for l,h in heads:
            result.setdefault(l,[]).append(h)
        return result

    def probability(self,l,h,q,k,mask):
        torch=self.torch
        dots=(q[:,h,0,:].float()[:,None,:]*k[:,h//4].float()).sum(-1)*128**-.5
        if mask is not None:
            m=mask[:,0,0]
            if m.dtype==torch.bool:
                dots=dots.masked_fill(~m,-torch.inf)
            else:
                dots=dots+m.float()
        return dots.softmax(-1)

    def attention(self,module,q,k,v,mask,**kw):
        torch=self.torch
        native,weights=self.stock(module,q,k,v,mask,**kw)
        l=module.layer_idx
        a=self.active_arm
        if self.probe:
            for ll,h in R.HEADS:
                if l!=ll:
                    continue
                for clock,step in self.clock.items():
                    active=self.good & (step==self.at)
                    if bool(active.any()):
                        p=self.probability(l,h,q,k,mask)
                        safe_idx=self.ctx["decoy"].clamp_max(k.shape[2]-1)
                        rec=self.quality.setdefault(f"{l}.{h}-{clock}",torch.zeros((len(self.good),2),device="cuda"))
                        probabilities=torch.stack((p[:,self.ctx["source"]],p[self.ridx,safe_idx]),dim=-1)
                        rec[active]=probabilities[active]
        if not a.kind or not self.at:
            return native,weights
        restore=a.kind in ("recover","recover_null")
        hs=self.restore_bg.get(l,()) if restore else self.mask_for_layer.get(l,())
        if not hs:
            return native,weights
        active=self.good & (self.clock[a.clock]==self.at)
        if not bool(active.any()):
            return native,weights
        source=self.ctx["source"] if a.source=="state" else self.ctx["decoy"].clamp_max(v.shape[2]-1)
        if a.source=="note":
            assert bool((self.ctx["decoy"][active]<v.shape[2]-1).all())
        def rerun(value):
            alt=v.clone()
            ori=alt[self.ridx,:,source]
            alt[self.ridx,:,source]=torch.where(active[:,None,None],value,ori)
            result,_=self.stock(module,q,k,alt,mask,**kw)
            return result
        if restore:
            changed=rerun(self.arrays["zero"][l])
            use=self.mask_for_layer.get(l,())
            repaired=(native if a.kind=="recover" else rerun(self.arrays["recover_null"][l])) if use else None
            if use:
                changed=torch.where(self.query_mask(active,use),repaired,changed)
            self.restored[active]+=len(use)
        else:
            key=f"{a.source}_{a.kind}" if a.kind in ("swap","null") else a.kind
            changed=rerun(self.arrays[key][l])
        self.count[active]+=len(hs)
        return torch.where(self.query_mask(active,hs),changed,native),weights

    def query_mask(self,active,heads):
        torch=self.torch
        mask=torch.zeros(32,dtype=torch.bool,device="cuda")
        mask[list(heads)]=True
        return (active[:,None]&mask[None,:])[:,None,:,None]

    def ready(self,a):
        torch=self.torch
        self.active_arm=a
        self.mask_for_layer=self.mask_for(a.group) if a.group is not None else {l:list(range(32)) for l in range(36)}
        self.restore_bg=self.mask_for(R.HEADS)
        self.count=torch.zeros(len(self.good),device="cuda",dtype=torch.int64)
        self.restored=torch.zeros_like(self.count)

    def state_at_clocks(self):
        """Clone the original cache through exact replay of only natural non-answer tokens."""
        torch=self.torch
        c=self.ctx
        cache=self.g.clone(c["c1"])
        mask=c["mask"]
        self.probe=True
        self.ready(Arm("natural"))
        scores=torch.zeros((len(self.good),2),device="cuda")
        best=torch.zeros(len(self.good),device="cuda",dtype=torch.int64)
        for i in range(1,self.last+1):
            for clock,t in self.minimum.items():
                if i==t:
                    self.sham[clock]=(self.g.clone(cache),mask.clone(),i)
            self.at=i
            token=torch.where(i<=self.clock["answer"],c["history"][i],self.g.pad)
            cache,mask,logits=self.g.step(cache,mask,token)
            selected=(self.clock["answer"]==i)&self.good
            assert torch.equal(logits[selected],c["steps"][i][selected])
            scores[selected]=logits[selected][:,self.answer_ids].float()
            best[selected]=logits[selected].argmax(-1)
        self.probe=False
        self.at=0
        return dict(scores=scores.cpu().tolist(),top=best.cpu().tolist(),read_count=self.count.cpu().tolist(),
                    restore_count=self.restored.cpu().tolist())

    def path(self,a,free=False):
        torch=self.torch
        self.ready(a)
        cache,mask,start=self.sham[a.clock]
        cache=self.g.clone(cache)
        history=list(self.ctx["history"][:start+1])
        if free:
            assert bool((self.clock["answer"][self.good]==start).all()) if start==self.minimum["answer"] else True
        ended=torch.stack([self.g.ended(t) for t in history]).any(0)
        scores=torch.zeros((len(self.good),2),device="cuda")
        best=torch.zeros(len(self.good),device="cuda",dtype=torch.int64)
        for i in range(start,R.CAP-1 if free else self.last+1):
            self.at=i
            token=history[-1] if free else torch.where(i<=self.clock["answer"],self.ctx["history"][i],self.g.pad)
            cache,mask,logits=self.g.step(cache,mask,token)
            selected=(self.clock["answer"]==i)&self.good
            scores[selected]=logits[selected][:,self.answer_ids].float()
            best[selected]=logits[selected].argmax(-1)
            if free:
                token=self.g.next_token(logits,ended)
                history.append(token)
                ended=ended|self.g.ended(token)
                if bool(ended.all()):
                    break
        self.at=0
        reads=3 if a.kind in ("recover","recover_null") else (len(a.group) if a.group is not None else 36*32) if a.kind else 0
        assert torch.equal(self.count,self.good.long()*reads),(a,self.count.cpu().tolist())
        assert torch.equal(self.restored,self.good.long()*(len(a.group) if a.kind.startswith("recover") else 0))
        if free:
            return torch.stack(history,dim=1).cpu().tolist()
        return dict(scores=scores.cpu().tolist(),top=best.cpu().tolist(),read_count=self.count.cpu().tolist(),
                    restore_count=self.restored.cpu().tolist())


def sample_native(ctx,reader,suite):
    import numpy as np
    import torch
    def bits(a):
        return a[:4].contiguous().view(torch.uint16).cpu().numpy()
    arrays={name:np.stack([bits(x) for x in vals]) for name,vals in
            (("original",reader.state),("donor",ctx["v_alt"]),("note",ctx["v_note"]))}
    arrays["hidden"]=np.asarray(ctx["field"],dtype=np.uint16)
    np.savez(R.Base.Previous.check_write(R.OUT/f"native-{suite}.npz"),ids=[r["id"] for r in ctx["rows"][:4]],**arrays)


def one(g,rows,baseline,suite,batch,tested):
    import torch
    ctx=g.natural(rows,retain=True,sample=range(4) if batch==1 else ())
    assert ctx["obs"]==baseline
    reader=Replay(g,ctx)
    try:
        sham=reader.state_at_clocks()
        if batch==1:
            sample_native(ctx,reader,suite)
        paths={a.name:reader.path(a) for a in tested if a.name!="natural"}
        paths["natural"]=sham
        # Independently verify the checkpointed free baseline from immediately before the emitted code.
        assert reader.path(Arm("natural"))==sham
        free={}
        homogeneous=bool((reader.clock["answer"]==reader.minimum["answer"]).all())
        if homogeneous:
            for a in tested:
                if a.name in ("natural","whole","whole_null","remaining","swap-012","null-012"):
                    tokens=reader.path(a,free=True)
                    free[a.name]=[R.parse(r,g.tok,t,g.eos,o["value_ids"],g.letters)
                                  for r,t,o in zip(rows,tokens,baseline,strict=True)]
            assert [x["tokens"] for x in free["natural"]]==[x["tokens"] for x in baseline]
        else:
            # The source/action never sees the natural answer. Variable layouts retain fixed-prefix
            # logits, with any free answer comparison recorded as unavailable for the full batch.
            free=dict(unavailable="mixed answer offsets, cannot clamp a uniform prefix before greedy code")
        return dict(ids=[r["id"] for r in rows],paths=paths,free=free,
                    exact_code_logits=True,after_state=ctx["after_state"],prompt_hash=ctx["prompt_hash"],
                    dose=reader.error,clock={k:v.cpu().tolist() for k,v in reader.clock.items()},
                    source=ctx["source"],distractor=ctx["decoy"].cpu().tolist(),
                    attention={k:v.cpu().tolist() for k,v in reader.quality.items()})
    finally:
        reader.close()


def discover(runs,rows):
    from statistics import mean
    by={r["id"]:r for r in rows}
    def score(i):
        result=[]
        for blob in runs:
            for rid,x,y in zip(blob["ids"],blob["paths"][f"swap-{i}"]["scores"],blob["paths"][f"null-{i}"]["scores"],strict=True):
                a=int(by[rid]["alt_code"]=="Q")
                result.append((x[a]-x[1-a])-(y[a]-y[1-a]))
        return mean(result)
    ranked=sorted(range(3),key=lambda z:(-score(z),z))
    return dict(focus_index=ranked[0],ranking=[dict(i=i,point=R.HEADS[i],difference=score(i)) for i in ranked],
                prechosen=R.HEADS,scope="refines emphasis only, all individual and subset contrasts remain frozen")


def execute():
    import torch
    torch.set_num_threads(4)
    plan=R.check()
    jobs=specs()
    with R.Base.Previous.GpuLease(),torch.inference_mode():
        model=R.load_model("q4")
        g=R.Task(model)
        ledger=[]
        try:
            for suite in R.M.SUITES:
                if suite!="discovery":
                    selected=R.read("selection.json")
                    assert selected["status"]=="frozen" and selected["plan_sha"]==R.sha(R.OUT/"plan.json")
                rows,behavior,passed=R.behavior(g,suite)
                if not passed:
                    ledger.append(dict(suite=suite,status="behavior gate failed",shards=[]))
                    if suite=="discovery":
                        R.write("selection.json",dict(status="failed behavior gate"))
                        break
                    continue
                shards=[]
                batches=list(zip(R.Base.groups(rows,R.BATCH),R.Base.groups(behavior,R.BATCH),strict=True))
                start=time.monotonic()
                R.Base.Previous.stage_log("q4",suite+"-interventions",0,len(batches),start)
                for i,(group,baseline) in enumerate(batches,1):
                    batch=one(g,group,baseline,suite,i,jobs)
                    name=f"native-results-{suite}-{i:02d}.json"
                    R.write(name,batch)
                    shards.append(dict(name=name,sha=R.sha(R.OUT/name)))
                    R.Base.Previous.stage_log("q4",suite+"-interventions",i,len(batches),start)
                ledger.append(dict(suite=suite,status="complete",shards=shards))
                if suite=="discovery":
                    selection=discover([R.read(x["name"]) for x in shards],rows)
                    R.write("selection.json",dict(status="frozen",**selection,plan_sha=R.sha(R.OUT/"plan.json"),
                                                  discovery_shards=shards))
        finally:
            model.close()
    R.write("confirmation.json",dict(ledger=ledger,arms=[a.__dict__ for a in jobs],selection_sha=R.sha(R.OUT/"selection.json")))


if __name__=="__main__":
    R.setup()
    execute()
