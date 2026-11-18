"""Frozen same-cache intervention and negative controls for the revised task."""
import hashlib
from pathlib import Path
import sys
import time

import own_state_explicit as E
R=E.R


def freeze():
    assert not (R.OUT/"causal-plan.json").exists()
    plan=R.check_plan()
    R.write("causal-plan.json",dict(behavior_plan_sha=R.digest(R.OUT/"plan.json"),
                                  source_sha=R.digest(Path(__file__)),generator_sha=R.digest(Path(E.__file__)),
                                  implementation="No layer fit or selection: all original value layers, one generated position",
                                  units="Each family gives a paired mean over its 8 arms. Natural failed rows score 0 in all arms.",
                                  decision={
                                      "minimum_families":96,
                                      "token": "family mean alternate-code success >=.25, positive each of the 8 cells, adjusted one-sided sign-flip p<=.0125",
                                      "value": "V-only minus matched control >=.10, both mapping differences positive, all 8 cell differences nonnegative, adjusted one-sided sign-flip p<=.0125",
                                      "control": "matched control alternate-code <=.25; loss of well-formed replies vs replay <=10 percentage points",
                                      "mechanical": "100% across batches of replay token/logit exact, original cache immutable, norm matched <=3% within every applied row",
                                      "dependency": "native evidence only if token and V gates both pass in that task; report separately per task"},
                                  sampling="fixed first two confirm families of each admitted task; all layers, raw first-pass checkpoints for each generation step",
                                  max_write_peak=400*2**20))


def stable_bytes(tensor):
    import torch
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def state_digest(tensor):
    return hashlib.sha256(stable_bytes(tensor)).hexdigest()


class Field:
    def __init__(self,generation,indices):
        self.indices=indices
        self.saved=[]
        self.sample={}
        self.handles=[]
        if not indices:
            return
        model=generation.model.model
        modules=[model.embed_tokens,*model.layers,model.norm]
        def collect(i,m,args,out):
            import torch
            if isinstance(out,tuple):
                out=out[0]
            self.sample[i]=out[self.indices,-1].detach().contiguous().view(torch.uint16).cpu().numpy().copy()
            if i==len(modules)-1:
                self.saved.append([self.sample[j] for j in range(len(modules))])
                self.sample={}
        for i,m in enumerate(modules):
            def hook(m,args,out,i=i):
                collect(i,m,args,out)
            self.handles.append(m.register_forward_hook(hook))

    def stop(self):
        for handler in self.handles:
            handler.remove()


def branch(g,rows,expected,index):
    import numpy as np
    import torch
    encoded,inputs=g.inputs(rows)
    if any(r["input_ids"]!=enc[0] or r["native_values"]!=enc[1] for r,enc in zip(expected,encoded,strict=True)):
        raise ValueError("Actual input tokens differ from admitted run")
    selected=[i for i,r in enumerate(rows) if r["index"]<2]
    field=Field(g,selected)
    original=g.first(inputs)
    g1=original.logits[:,-1].argmax(-1)
    pre=g.clone(original.past_key_values)
    pre_hash=g.full_hash(pre)
    full,mask,g1_logits=g.step(original.past_key_values,inputs.attention_mask,g1)
    natural_after=g.clone(full)
    done=g.ended(g1)
    g2=g.next_token(g1_logits,done)
    locked_done=done|g.ended(g2)
    natural=g.advance(full,mask,[g1,g2],locked_done)
    del full,original
    field.stop()
    field_file=None
    if field.saved:
        field_file=f"field-{index:03d}.npz"
        np.savez(R.Previous.check_write(R.OUT/field_file),ids=[rows[i]["id"] for i in selected],
                 hidden_bits=np.asarray(field.saved,dtype=np.uint16))
    recovered=g.records(rows,encoded,natural)
    exact_original=all(a["natural"]["tokens"]==b["natural"]["tokens"] for a,b in zip(expected,recovered,strict=True))
    replay_cache,replay_mask,replay_logits=g.step(g.clone(pre),inputs.attention_mask,g1)
    replay_g2=g.next_token(replay_logits,done)
    replay=g.advance(replay_cache,replay_mask,[g1,replay_g2],done|g.ended(replay_g2))
    del replay_cache
    checks=dict(admitted_tokens_match=exact_original,exact_tokens=natural==replay,
                bitwise_logits=torch.equal(g1_logits,replay_logits),natural_logits_sha=state_digest(g1_logits),
                replay_logits_sha=state_digest(replay_logits),original_prompt_kv_sha=pre_hash,field=field_file)
    if not all(checks[k] for k in ("admitted_tokens_match","exact_tokens","bitwise_logits")):
        R.write(f"replay-failure-{index:03d}.json",dict(checks=checks,ids=[r["id"] for r in rows],natural=natural,replay=replay))
        raise RuntimeError("Same-generation unedited branch failed. No affected interventions admitted.")

    apply=torch.tensor([b["natural"]["joint"] for b in recovered],dtype=torch.bool,device="cuda")
    donors=torch.tensor([enc[1][r["alt"]] for enc,r in zip(encoded,rows,strict=True)],device="cuda")
    pending=torch.where(apply,donors,g1)
    altered,dmask,altered_logits=g.step(g.clone(pre),inputs.attention_mask,pending)
    src=[layer.values[:,:,-1:,:].detach().clone() for layer in altered.layers]
    before=[layer.values[:,:,-1:,:].detach().clone() for layer in natural_after.layers]
    first_done=g.ended(pending)
    other_g2=g.next_token(altered_logits,first_done)
    swapped=g.advance(altered,dmask,[pending,other_g2],first_done|g.ended(other_g2))
    del altered
    source_immutable=g.full_hash(pre)==pre_hash
    del pre

    replace=g.clone(natural_after)
    random=g.clone(natural_after)
    real_power=torch.zeros(len(rows),device="cuda")
    null_power=torch.zeros_like(real_power)
    native_bytes=[]
    for l,(base,target) in enumerate(zip(before,src,strict=True)):
        orig=base.float()
        delta=target.float()-orig
        scrambled=(orig+torch.roll(delta,shifts=(13+11*l)%(delta.shape[-1]-1)+1,dims=-1)).to(base.dtype)
        replace.layers[l].values[:,:,-1:,:]=target
        random.layers[l].values[:,:,-1:,:]=scrambled
        real_power+=delta.flatten(1).square().sum(1)
        null_power+=(scrambled.float()-orig).flatten(1).square().sum(1)
        native_bytes.append(torch.stack([base.squeeze(2),target.squeeze(2)],dim=1).contiguous().view(torch.uint16).cpu().numpy())
    target_norm=real_power.sqrt().cpu().tolist()
    matched_norm=null_power.sqrt().cpu().tolist()
    norm_error=[abs(a-b)/max(a,1e-20) for a,b in zip(target_norm,matched_norm)]
    checks["source_kv_immutable"]=source_immutable
    checks["norm_match"]=all(err<=R.GATES["max_norm_relative_error"] for err in norm_error)
    checks["natural_state_cache_immutable"]=all(torch.equal(layer.values[:,:,-1:,:],base) for layer,base in zip(natural_after.layers,before,strict=True))
    if not all(checks[k] for k in ("source_kv_immutable","norm_match","natural_state_cache_immutable")):
        R.write(f"control-failure-{index:03d}.json",dict(checks=checks,ids=[r["id"] for r in rows],norm_error=norm_error))
        raise RuntimeError("KV immutability or norm gate failed")
    reuse_v=g.advance(replace,mask,[g1,g2],locked_done)
    del replace
    reuse_null=g.advance(random,mask,[g1,g2],locked_done)
    slot_file=f"value-slot-{index:03d}.npz"
    np.savez(R.Previous.check_write(R.OUT/slot_file),ids=[r["id"] for r in rows],
             value_bits=np.stack(native_bytes,axis=1),actual_donor_token=pending.cpu().numpy())
    arms={}
    for label,tokens in (("replay",replay),("state_token_swap",swapped),("value_patch",reuse_v),("matched_null",reuse_null)):
        arms[label]=[r["natural"] for r in g.records(rows,encoded,tokens)]
    data=[]
    for i,(r,o) in enumerate(zip(rows,recovered,strict=True)):
        data.append(dict(id=r["id"],family=r["family"],task=r["task"],factors=r["factors"],
                         applied=bool(apply[i]),natural=o["natural"],
                         alt_code=r["alt_code"],target_norm=target_norm[i],control_norm=matched_norm[i],
                         **{label:branch[i] for label,branch in arms.items()}))
    return dict(checks=checks,rows=data,slot_file=slot_file,slot_sha=R.digest(R.OUT/slot_file))


def run():
    import torch
    R.check_plan()
    plan=R.read("causal-plan.json")
    assert plan["source_sha"]==R.digest(Path(__file__))
    assert plan["generator_sha"]==R.digest(Path(E.__file__))
    assert plan["behavior_plan_sha"]==R.digest(R.OUT/"plan.json")
    assert not (R.OUT/"causal-run.json").exists()
    admitted=R.read("q4-confirm.json")
    if not admitted["admitted"]:
        R.write("causal-run.json",dict(status="not admitted",batches=[],scope=admitted["admitted"]))
        return
    rows=[r for r in R.read("materials-confirm.json") if r["task"] in admitted["admitted"]]
    by_id={r["id"]:r for r in admitted["records"]}
    batches=list(R.groups(rows,R.GPU_BATCH))
    start=time.monotonic()
    complete=[]
    torch.set_num_threads(4)
    with R.Previous.GpuLease(),torch.inference_mode():
        identity=R.Previous.identity("q4")
        assert identity==R.read("q4-model.json")["files"]
        engine=R.load_model("q4")
        try:
            g=R.Greedy(engine)
            R.Previous.stage_log("q4","same-cache-intervention",0,len(batches),start)
            for i,group in enumerate(batches,1):
                record=branch(g,group,[by_id[r["id"]] for r in group],i)
                name=f"causal-{i:03d}.json"
                R.write(name,record)
                complete.append(dict(name=name,sha=R.digest(R.OUT/name)))
                R.Previous.stage_log("q4","same-cache-intervention",i,len(batches),start)
        finally:
            engine.close()
    R.write("causal-run.json",dict(status="complete",batches=complete,scope=admitted["admitted"],
                                   behavior_sha=R.digest(R.OUT/"q4-confirm.json"),
                                   execution_seconds=time.monotonic()-start))


if __name__=="__main__":
    E.select()
    R.configure()
    {"freeze":freeze,"run":run}[sys.argv[1]]()
