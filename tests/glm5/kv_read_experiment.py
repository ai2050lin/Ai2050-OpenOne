"""Native Qwen SDPA replay, discovery scans, and frozen head/time confirmation."""
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import kv_read_run as R

TIMES=("first","interior","answer")
FIELDS=("keys","values")


@dataclass
class Arm:
    name: str
    field: str = ""
    kind: str = ""
    points: tuple | None = None


class Reader:
    def __init__(self,generation,context):
        import torch
        from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS
        from own_state_intervene import state_digest
        self.torch=torch
        self.g=generation
        self.ctx=context
        self.source=context["source"]
        self.registry=ALL_ATTENTION_FUNCTIONS
        self.native=self.registry["sdpa"]
        self.registry["sdpa"]=self.interface
        self.arm=Arm("natural")
        self.now=0
        self.where={}
        self.tags={}
        records=context["records"]
        self.good=torch.as_tensor([r["joint"] for r in records],device="cuda",dtype=torch.bool)
        self.answer=torch.as_tensor([r["code_index"]-1 if r["joint"] else 3 for r in records],device="cuda")
        assert bool((self.answer>=2).all())
        self.maximum=int(self.answer.max())
        self.answer_ids=[generation.answer_ids[k] for k in ("P","Q")]
        self.alternate=torch.as_tensor([int(r["alt_code"]=="Q") for r in context["rows"]],device="cuda")
        self.orig={}
        self.scramble={}
        self.zero_scramble={}
        self.dose={}
        self.cache_hash=state_digest(context["cache"].layers[-1].values)
        for f in FIELDS:
            self.orig[f]=[getattr(l,f)[:,:,self.source].clone() for l in context["cache"].layers]
            self.scramble[f],self.zero_scramble[f]=[],[]
            rates={"swap":0.,"zero":0.}
            for i,(base,target) in enumerate(zip(self.orig[f],context["donor"][f],strict=True)):
                floating=base.float()
                for kind,wanted,collection in (("swap",target,self.scramble),("zero",torch.zeros_like(target),self.zero_scramble)):
                    delta=wanted.float()-floating
                    background=(floating+torch.roll(delta,shifts=(11+17*i)%127+1,dims=-1)).to(base.dtype)
                    collection[f].append(background)
                    a=delta.norm(dim=-1)
                    b=(background.float()-floating).norm(dim=-1)
                    received=((b-a).abs()/a.clamp_min(1e-12))[self.good]
                    if received.numel():
                        rates[kind]=max(rates[kind],float(received.max()))
            self.dose[f]=rates
        assert max(j for v in self.dose.values() for j in v.values())<=.03

    def close(self):
        from own_state_intervene import state_digest
        self.registry["sdpa"]=self.native
        assert state_digest(self.ctx["cache"].layers[-1].values)==self.cache_hash
        for f in FIELDS:
            assert all(self.torch.equal(x,getattr(l,f)[:,:,self.source])
                       for x,l in zip(self.orig[f],self.ctx["cache"].layers,strict=True))

    def schedule(self,arm):
        self.arm=arm
        self.where={}
        for l,h,t in arm.points or ():
            self.where.setdefault(l,[]).append((h,t))

    def use(self,when):
        if when=="all":
            return self.good & (self.now<=self.answer)
        if when=="first":
            return self.good & (self.now==1) & (self.answer>1)
        if when=="answer":
            return self.good & (self.now==self.answer)
        if when=="interior":
            return self.good & (self.now>1) & (self.now<self.answer)
        raise ValueError(when)

    def use_heads(self,l):
        torch=self.torch
        if self.arm.points is None:
            return self.use("all")[:,None,None,None]
        heads=torch.zeros((len(self.good),32),dtype=torch.bool,device="cuda")
        for h,t in self.where.get(l,[]):
            if h==-1:
                heads|=self.use(t)[:,None]
            else:
                heads[:,h]|=self.use(t)
        return heads[:,None,:,None]

    def interface(self,module,q,k,v,mask,**kw):
        torch=self.torch
        a=self.arm
        l=module.layer_idx
        stock,weights=self.native(module,q,k,v,mask,**kw)
        if not a.field or not self.now or (a.points is not None and l not in self.where and a.kind!="restore"):
            return stock,weights
        selected=self.use_heads(l)
        active=self.use("all") if a.kind=="restore" else selected[:,0,:,0].any(-1)
        if not bool(active.any()):
            return stock,weights
        real=k if a.field=="keys" else v
        if a.kind=="swap":
            new=self.ctx["donor"][a.field][l]
        elif a.kind=="scramble":
            new=self.scramble[a.field][l]
        elif a.kind=="zero_scramble":
            new=self.zero_scramble[a.field][l]
        else:
            assert a.kind in ("zero","restore")
            new=torch.zeros_like(real[:,:,self.source])
        altered=real.clone()
        altered[:,:,self.source]=torch.where(active[:,None,None],new,real[:,:,self.source])
        recalculated,_=self.native(module,q,altered if a.field=="keys" else k,
                                   altered if a.field=="values" else v,mask,**kw)
        if a.kind=="restore":
            result=torch.where(selected,stock,recalculated)
        else:
            result=torch.where(selected,recalculated,stock)
        return result,weights

    def path(self,arm,free=False,exact=False):
        torch=self.torch
        ctx=self.ctx
        self.schedule(arm)
        history=list(ctx["history"][:2])
        cache=self.g.clone(ctx["cache"])
        mask=ctx["mask"]
        picked=torch.zeros((len(self.good),2),device="cuda",dtype=torch.float32)
        chosen=torch.zeros(len(self.good),device="cuda",dtype=torch.int64)
        done=self.g.ended(history[1])|self.g.ended(history[0])
        for i in range(1,R.CAP-1 if free else self.maximum+1):
            self.now=i
            token=history[-1] if free else ctx["history"][i]
            if not free and i>1:
                token=torch.where(i<=self.answer,token,torch.full_like(token,self.g.pad))
            cache,mask,logits=self.g.step(cache,mask,token)
            at=(self.answer==i)&self.good
            if exact:
                assert torch.equal(logits[at],ctx["logits"][i][at]),(i,int(at.sum()))
            picked=torch.where(at[:,None],logits[:,self.answer_ids].float(),picked)
            chosen=torch.where(at,logits.argmax(-1),chosen)
            if free:
                nxt=self.g.next_token(logits,done)
                history.append(nxt)
                done=done|self.g.ended(nxt)
                if bool(done.all()):
                    break
        self.now=0
        if free:
            return torch.stack(history,dim=1).cpu().tolist()
        return dict(scores=picked.cpu().tolist(),top=chosen.cpu().tolist())


def mean_margin(data,rows):
    return sum(scores[int(r["alt_code"]=="Q")]-scores[int(r["code"]=="Q")]
               for scores,r in zip(data["scores"],rows,strict=True))/len(rows)


def matched(points):
    used={(l,h) for l,h,_ in points}
    result=[]
    for l,h,t in points:
        alt=next(j for a in range(1,32) if (l,(j:=(h+a)%32)) not in used)
        used.add((l,alt))
        result.append((l,alt,t))
    return tuple(result)


def inverse(points):
    return tuple((l,h,t) for l,h,chosen in points for t in TIMES if t!=chosen)


def discover(g,p):
    assert not (R.OUT/"selection.json").exists()
    rows,baseline,passed=R.measure(g,"discovery","q4")
    if not passed:
        R.write("selection.json",dict(status="blocked by discovery behavior",plan_sha=R.sha(R.OUT/"plan.json")))
        return
    context=g.natural(rows,True,field_indices=range(4))
    assert context["records"]==baseline
    saved_sample(context,"discovery")
    reader=Reader(g,context)
    log=[]
    t0=time.monotonic()
    try:
        natural=reader.path(Arm("baseline"),exact=True)
        assert reader.path(Arm("replay"),exact=True)==natural
        assert reader.path(Arm("replay-free"),free=True)==[r["tokens"] for r in baseline]
        base=mean_margin(natural,rows)
        choices={}
        for f in FIELDS:
            def probe(name,points):
                data=reader.path(Arm(name,f,"swap",points))
                effect=mean_margin(data,rows)-base
                log.append(dict(name=name,field=f,points=points,delta=effect,raw=data))
                if len(log)%8==0:
                    R.Base.Previous.stage_log("q4","discovery-scan",len(log),226,t0)
                return effect
            probe("all",None)
            for time_name in TIMES:
                probe("all-"+time_name,tuple((l,-1,time_name) for l in range(36)))
            layers=sorted(range(36),key=lambda l:(-probe("layer",((l,-1,"all"),)),l))[:p["selection"]["layers_per_side"]]
            heads=sorted(((l,h) for l in layers for h in range(32)),
                         key=lambda q:(-probe("head",((q[0],q[1],"all"),)),*q))[:p["selection"]["heads_per_side"]]
            picked=[]
            for l,h in heads:
                ranked=sorted(TIMES,key=lambda t:(-probe("time",((l,h,t),)),TIMES.index(t)))
                picked.append([l,h,ranked[0]])
            choices[f]=dict(layers=layers,points=picked)
        R.write("discovery-results.json",dict(ids=[r["id"] for r in rows],baseline=natural,arms=log,norm=reader.dose))
        R.write("selection.json",dict(status="frozen",choices=choices,
                                     discovery_sha=R.sha(R.OUT/"discovery-results.json"),plan_sha=R.sha(R.OUT/"plan.json")))
        R.Base.Previous.stage_log("q4","discovery-scan",len(log),len(log),t0)
    finally:
        reader.close()


def saved_sample(context,name):
    import numpy as np
    from own_state_audit import f32
    import torch
    if context["field"]:
        arrays={}
        for f in FIELDS:
            before=[getattr(x,f)[:4,:,context["source"]].contiguous().view(torch.uint16).cpu().numpy()
                    for x in context["cache"].layers]
            after=[x[:4].contiguous().view(torch.uint16).cpu().numpy() for x in context["donor"][f]]
            arrays[f]=np.stack([before,after])
        arrays["field"]=np.asarray(context["field"],dtype=np.uint16)
        assert np.isfinite(f32(arrays["field"])).all()
        np.savez(R.Base.Previous.check_write(R.OUT/f"native-{name}.npz"),ids=[r["id"] for r in context["rows"][:4]],**arrays)


def arm_list(choices):
    result=[Arm("natural")]
    for f in FIELDS:
        chosen=tuple(tuple(x) for x in choices[f]["points"])
        for name,kind,points in (("all_swap","swap",None),("all_scramble","scramble",None),
                                 ("selected","swap",chosen),("selected_scramble","scramble",chosen),
                                 ("other_time","swap",inverse(chosen)),
                                 ("selected_zero","zero",chosen),("selected_zero_null","zero_scramble",chosen),
                                 ("all_zero","zero",None),("restore","restore",chosen),
                                 ("restore_matched","restore",matched(chosen))):
            result.append(Arm(f"{f}-{name}",f,kind,points))
        for i,(l,h,t) in enumerate(chosen):
            result.extend((Arm(f"{f}-head{i}","keys" if f=="keys" else "values","swap",((l,h,t),)),
                           Arm(f"{f}-head{i}_null",f,"scramble",((l,h,t),)),
                           Arm(f"{f}-head{i}_other",f,"swap",inverse(((l,h,t),)))))
    return result


def confirm(g):
    frozen=R.read("selection.json")
    if frozen["status"]!="frozen":
        R.write("confirmation.json",dict(status="discovery not admitted",suites=[]))
        return
    assert frozen["discovery_sha"]==R.sha(R.OUT/"discovery-results.json")
    assert frozen["plan_sha"]==R.sha(R.OUT/"plan.json")
    choices=frozen["choices"]
    specs=arm_list(choices)
    suites=[]
    for suite in R.Material.SUITES[1:]:
        rows,observations,passed=R.measure(g,suite,"q4")
        if not passed:
            suites.append(dict(suite=suite,status="behavior gate failed",shards=[]))
            continue
        results=[]
        start=time.monotonic()
        work=list(zip(R.Base.groups(rows,R.BATCH),R.Base.groups(observations,R.BATCH),strict=True))
        R.Base.Previous.stage_log("q4",f"{suite}-intervention",0,len(work),start)
        for number,(group,baseline) in enumerate(work,1):
            context=g.natural(group,True,field_indices=range(4) if number==1 else ())
            assert context["records"]==baseline
            saved_sample(context,suite)
            reader=Reader(g,context)
            try:
                paths={arm.name:reader.path(arm,exact=arm.name=="natural") for arm in specs}
                assert reader.path(Arm("replay"),exact=True)==paths["natural"]
                free={}
                for arm in specs:
                    if arm.name in ("natural","keys-all_swap","values-all_swap","values-all_scramble","values-selected","values-selected_scramble"):
                        tokens=reader.path(arm,free=True)
                        free[arm.name]=[R.parsed(r,g.tok,a,g.eos,b["value_ids"],g.answer_ids)
                                        for r,a,b in zip(group,tokens,baseline,strict=True)]
                assert [x["tokens"] for x in free["natural"]]==[x["tokens"] for x in baseline]
                name=f"confirm-{suite}-{number:02d}.json"
                R.write(name,dict(ids=[r["id"] for r in group],paths=paths,free=free,norm=reader.dose,
                                  replay_exact=True,source_index=context["source"],prefill_hashes=context["hashes"],
                                  after_state_logit_hash=context["site_hash"],sites=[r["code_index"] for r in baseline]))
                results.append(dict(name=name,sha=R.sha(R.OUT/name)))
            finally:
                reader.close()
            R.Base.Previous.stage_log("q4",f"{suite}-intervention",number,len(work),start)
        suites.append(dict(suite=suite,status="complete",shards=results))
    R.write("confirmation.json",dict(status="complete",suites=suites,selection_sha=R.sha(R.OUT/"selection.json"),
                                    arms=[x.__dict__ for x in specs]))


def run():
    import torch
    torch.set_num_threads(4)
    p=R.plan()
    with R.Base.Previous.GpuLease(),torch.inference_mode():
        engine=R.open_model("q4")
        try:
            g=R.Task(engine)
            if not (R.OUT/"selection.json").exists():
                discover(g,p)
            assert not (R.OUT/"confirmation.json").exists()
            confirm(g)
        finally:
            engine.close()


if __name__=="__main__":
    R.setup()
    run()
