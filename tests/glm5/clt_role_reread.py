"""Qwen3-14B, source-specific early V and late head-local K interventions.

Run from run_routine_test.py using the corresponding frozen archive plan.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

ROOT = Path(os.environ["PYTHONPATH"].split(os.pathsep)[0]) if Path(__file__).name=="test.py" else Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"tests/glm5"), str(ROOT)]
from clt_role_reread_materials import SIZES, SEED, MODEL, identities
from deploy.storage_policy import check_write
from deploy.test_archive_client import save
from shared.dist_protocol.test_archive import digest
from source_binding_design import read, sha

ARMS = ("base", "cut_a", "control_a", "cut_b", "control_b", "sham", "restore")
EPS = 1e-9


def context():
    work = Path(os.environ.get("AI2050_TEST_WORK_DIR",""))
    questions = Path(os.environ.get("AI2050_TEST_QUESTIONS",""))
    if not work.is_absolute() or not questions.is_file():
        raise RuntimeError("A registered routine run is required")
    plan, preflight, review = [read(questions.parent/n) for n in ("protocol.json","preflight.json","review.json")]
    if (preflight.get("plan_sha256")!=digest(plan) or not preflight.get("preflight_id") or not review.get("server_run_id")
            or Path(review["scratch"]).resolve()!=work.resolve() or plan["entrypoint_sha256"]!=sha(Path(__file__))
            or plan["dataset"]["questions_file_sha256"]!=sha(questions) or plan["acquisition"]["code_sha256"]!=digest(identities())):
        raise ValueError("archive or code identity mismatch")
    return work, questions, plan, review["server_run_id"]


def tensor_hash(tensor):
    import torch
    raw = tensor.contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(raw.tobytes()).hexdigest()


def fit_axis(samples, labels, seed):
    x = np.asarray(samples, dtype=np.float64)
    y = np.asarray(labels)
    means = [x[y==i].mean(axis=0) for i in (0,1)]
    difference = means[1]-means[0]
    size = np.linalg.norm(difference)
    if not np.isfinite(size) or size < EPS:
        raise ValueError("unidentifiable discovery axis")
    unit = difference/size
    noise = np.random.default_rng(seed).standard_normal(x.shape[1])
    noise -= noise.dot(unit)*unit
    return {"mean":((means[0]+means[1])/2).tolist(), "direction":unit.tolist(),
            "random":(noise/np.linalg.norm(noise)).tolist()}


def fit(rows, native):
    # Before the ledger each queried name is duplicated across four late/rule conditions.
    models = {"early":[], "keys":[], "method":"centered native rank-one source projections"}
    for archive in range(2):
        use = [i for i,r in enumerate(rows) if r["active"]==archive]
        unique = [i for i in use if rows[i]["late"]==0 and rows[i]["mode"]==0]
        models["early"].append(fit_axis(native["early_v"][unique], [rows[i]["early"] for i in unique], SEED+archive))
        slots = []
        for slot in range(2):
            slots.append(fit_axis(native["keys"][use,slot,:], [int(rows[i]["late"]==slot) for i in use], SEED+2+2*archive+slot))
        models["keys"].append(slots)
    return models


def projection(value, model):
    center, direction, random = [value.new_tensor(model[k]) for k in ("mean","direction","random")]
    centered = value-center
    return centered.dot(direction)*direction, centered.dot(random)*random, float(centered.dot(direction))


def matched(target, control, norm):
    n1, n2 = float(norm(target)), float(norm(control))
    common = min(n1,n2)
    return target*(common/max(n1,EPS)), control*(common/max(n2,EPS))


def evaluate_batch(rt, rows, candidates, progress):
    t = rt.t
    original_block = rt.block
    details = [dict() for _ in rows]
    fields = {k:[] for k in ("early_v","keys","h18","q26","q28")}

    def block(b):
        layer = original_block(b)
        if b not in (25,26,28):
            return layer
        head = {25:18,26:36,28:21}[b]
        attn = layer.self_attn
        group = head//(rt.heads//rt.kv)
        start = head*rt.hd
        cache = {}
        offset = 0

        def prepare(module, args, kwargs):
            cache.update(pe=kwargs["position_embeddings"], mask=kwargs["attention_mask"])

        def remember(key):
            def hook(module, args, result):
                cache[key] = result
            return hook

        def inspect(module, args):
            nonlocal offset
            actual = args[0]
            n, length, _ = actual.shape
            q = attn.q_norm(cache["q"].reshape(n,length,rt.heads,rt.hd)).transpose(1,2)
            k = attn.k_norm(cache["k"].reshape(n,length,rt.kv,rt.hd)).transpose(1,2)
            q,k = rt.mod.apply_rotary_pos_emb(q,k,*cache["pe"])
            q,k = q[:,head,-1].float(), k[:,group].float()
            v = cache["v"].reshape(n,length,rt.kv,rt.hd)[:,:,group].float()
            score = (k*q[:,None]).sum(-1)*attn.scaling
            score = score.masked_fill(~cache["mask"][:,0,-1],-float("inf"))
            weights = score.softmax(-1)
            altered = actual.clone() if candidates else actual
            for j in range(n):
                r = rows[offset+j]
                d = details[offset+j]
                arm = r["arm"]
                active = r["active"]
                pad = length-len(r["tokens"])
                query = pad+r["query"][-1]
                indices = [t.tensor([pad+x for x in span],device=k.device) for span in r["records"]]
                pair = indices[2*active:2*active+2]
                output = actual[j,-1,start:start+rt.hd]
                calc = weights[j] @ v[j]
                d[f"reconstruction_{b}"] = float((calc-output.float()).abs().max()/(1+output.float().abs().max()))
                if b==25:
                    value = v[j,query]
                    fields["early_v"].append(value.cpu().numpy())
                    alpha = weights[j,query]
                    d["early_alpha"] = float(alpha)
                    d["early_value_hash"] = tensor_hash(value)
                    d["write_hash"] = tensor_hash(alpha*value)
                    if candidates:
                        change, random, projection_score = projection(value,candidates["early"][active])
                        d["early_readout"] = projection_score
                        if arm in ("cut_a","control_a","restore","sham"):
                            out_weight = module.weight[:,start:start+rt.hd].float()
                            change, random = matched(alpha*change, alpha*random, lambda x:(out_weight@x).norm())
                            delta = random if arm=="control_a" else change
                            if arm=="sham":
                                delta = delta*0
                            edited = (output.float()-delta).to(output.dtype)
                            d["dose_a"] = float((out_weight@(edited.float()-output.float())).norm())
                            altered[j,-1,start:start+rt.hd] = edited
                            if arm=="restore":
                                altered[j,-1,start:start+rt.hd] = output
                else:
                    fields["q"+str(b)].append(q[j].cpu().numpy())
                    mass = [float(weights[j,ix].sum()) for ix in indices]
                    d["mass"+str(b)] = mass
                    if b==28:
                        continue
                    means = t.stack([k[j,ix].mean(0) for ix in pair])
                    fields["keys"].append(means.cpu().numpy())
                    d["body_kv_hash"] = tensor_hash(t.cat([k[j,ix] for ix in indices]+[v[j,ix] for ix in indices]))
                    d["q26_hash"] = tensor_hash(q[j])
                    if candidates:
                        parts = [projection(means[s], candidates["keys"][active][s]) for s in range(2)]
                        d["late_readout"] = [x[2] for x in parts]
                        if arm in ("cut_b","control_b","restore","sham"):
                            # Subtract a constant from every logit in a record. This is exactly
                            # q dot a change to each K in this consumer; shared KV heads stay native.
                            shifts = [t.stack([q[j].dot(x[h])*attn.scaling for x in parts]) for h in range(2)]
                            counts = q.new_tensor([len(ix) for ix in pair])
                            change, random = matched(*shifts,lambda x:(counts*x.square()).sum().sqrt())
                            delta = random if arm=="control_b" else change
                            if arm=="sham":
                                delta = delta*0
                            changed = score[j].clone()
                            for s,ix in enumerate(pair):
                                changed[ix] -= delta[s]
                            d["dose_b"] = float(((score[j]-changed)[t.cat(pair)]).norm())
                            updated = changed.softmax(-1)
                            # Apply only the local delta to the original SDPA output.
                            altered[j,-1,start:start+rt.hd] = (output.float()+(updated-weights[j])@v[j]).to(output.dtype)
                            if arm=="restore":
                                altered[j,-1,start:start+rt.hd] = output
                            else:
                                d["mass26"] = [float(updated[ix].sum()) for ix in indices]
            offset += n
            return (altered,)

        attn.register_forward_pre_hook(prepare,with_kwargs=True)
        for key in ("q","k","v"):
            getattr(attn,key+"_proj").register_forward_hook(remember(key))
        attn.o_proj.register_forward_pre_hook(inspect)
        return layer

    def step(b,layer,x,y,batch,length,offset=0,**kwargs):
        if b==17:
            for j,r in enumerate(batch):
                value = y[j,length-len(r["tokens"])+r["query"][-1]]
                fields["h18"].append(value.contiguous().view(t.uint16).cpu().numpy())
                details[offset+j]["h18_hash"] = tensor_hash(value)
        if b>=0 and (b+1)%8==0 and offset+len(batch)==len(rows):
            progress(b+1)

    rt.block = block
    try:
        _, logits = rt.forward(rows,step)
    finally:
        rt.block = original_block
    for d,r,logit in zip(details,rows,logits,strict=True):
        d.update({k:r[k] for k in ("id","family","mode","active","early","late","target","arm")})
        d.update(scores=logit[r["choice_ids"]].tolist(), top_id=int(logit.argmax()),
                 correct=int(logit.argmax())==r["choice_ids"][r["target"]], logits_hash=tensor_hash(logit))
    return details, {k:np.stack(values) for k,values in fields.items()}


def core_gate(rows):
    rates, routing = {}, {}
    for early in (0,1):
        for late in (0,1):
            for mode in (0,1):
                cell = [r for r in rows if (r["early"],r["late"],r["mode"])==(early,late,mode)]
                rates[f"{early}{late}{mode}"] = float(np.mean([r["correct"] for r in cell]))
    for mode in (0,1):
        conflicts = [r for r in rows if r["early"]!=r["late"] and r["mode"]==mode]
        routing[str(mode)] = float(np.mean([r["mass26"][r["target"]]>r["mass26"][r["target"]^1] for r in conflicts]))
    groups = {}
    for r in rows:
        groups.setdefault((r["family"],r["early"]),set()).add((r["h18_hash"],r["early_value_hash"]))
    errors = {k:max(r[k] for r in rows) for k in ("reconstruction_25","reconstruction_26","reconstruction_28")}
    technical = all(len(v)==1 for v in groups.values()) and max(errors.values())<=.025
    return dict(cells=rates,conflict_route=routing,technical=technical,max_pre_o_errors=errors,
                passed=technical and min(rates.values())>=.85 and min(routing.values())>=.70)


def log_contrast(r, metric, row):
    hi = 2*r["active"]+row
    if metric=="output":
        return r["scores"][hi]-r["scores"][hi^1]
    m = r["mass"+metric]
    return float(np.log(max(m[hi],1e-30))-np.log(max(m[hi^1],1e-30)))


def confirmation(rows):
    by_id = {(r["id"],r["arm"]):r for r in rows}
    base = [r for r in rows if r["arm"]=="base"]
    families = sorted({r["family"] for r in base})
    rng = np.random.default_rng(SEED+99)
    draws = rng.integers(len(families),size=(5000,len(families)))
    def interval(pairs):
        values = np.zeros(len(families))
        counts = np.zeros(len(families))
        for f,v in pairs:
            values[f] += v
            counts[f] += 1
        if np.any(counts==0):
            raise ValueError("incomplete family contrast")
        values /= counts
        return [float(values.mean()),*np.quantile(values[draws].mean(1),[.005,.995]).tolist()]
    tests, preservation = {}, {}
    for metric in ("26","output","28"):
        for mode in (0,1):
            subset = [r for r in base if r["mode"]==mode]
            for channel,sign in (("a",-1),("b",1)):
                effects = []
                for r in subset:
                    if r["early"]==r["late"]:
                        continue
                    cut,ctl = [by_id[r["id"],arm+"_"+channel] for arm in ("cut","control")]
                    effects.append((r["family"],sign*(log_contrast(cut,metric,r["early"])-log_contrast(ctl,metric,r["early"]))))
                tests[f"{metric}/{mode}/{channel}"] = interval(effects)
            chan = "a" if mode else "b"
            kept, natural = [], []
            for r in subset:
                role = r["late"] if mode else r["early"]
                b = log_contrast(r,metric,role)
                retained = log_contrast(by_id[r["id"],"cut_"+chan],metric,role)
                natural.append((r["family"],b))
                kept.append((r["family"],retained-.8*b))
            preservation[f"{metric}/{mode}"] = dict(baseline=interval(natural),excess_above_80pct=interval(kept))
    doses = {}
    for ch in ("a","b"):
        pairs = [(by_id[r["id"],"cut_"+ch]["dose_"+ch],by_id[r["id"],"control_"+ch]["dose_"+ch]) for r in base]
        doses[ch] = float(np.mean([a>0 and abs(a-b)<=.1*a for a,b in pairs]))
    edges = all(by_id[r["id"],a]["body_kv_hash"]==r["body_kv_hash"] and
                by_id[r["id"],a]["early_alpha"]==r["early_alpha"] for r in base for a in ("cut_a","control_a"))
    edges &= all(by_id[r["id"],a]["write_hash"]==r["write_hash"] and by_id[r["id"],a]["q26_hash"]==r["q26_hash"]
                 and by_id[r["id"],a]["body_kv_hash"]==r["body_kv_hash"] for r in base for a in ("cut_b","control_b"))
    restore = all(by_id[r["id"],a]["logits_hash"]==r["logits_hash"] and by_id[r["id"],a]["mass26"]==r["mass26"]
                  for r in base for a in ("sham","restore"))
    decode = {}
    for archive in (0,1):
        subset = [r for r in base if r["active"]==archive]
        decode[f"early/{archive}"] = float(np.mean([(r["early_readout"]>0)==bool(r["early"]) for r in subset]))
        decode[f"late/{archive}"] = float(np.mean([np.argmax(r["late_readout"])==r["late"] for r in subset]))
    qualified = bool(core_gate(base)["passed"] and min(decode.values())>=.80 and min(doses.values())>=.95 and edges and restore)
    def passed(metric):
        return qualified and all(tests[f"{metric}/{m}/{c}"][1]>0 for m in (0,1) for c in ("a","b")) and all(
            min(v[1] for v in preservation[f"{metric}/{m}"].values())>0 for m in (0,1))
    return dict(conflict_effects=tests,positive_controls=preservation,local_readout=decode,
                dose_match=doses,intact_edges=edges,sham_and_restore=restore,technical_and_behavior=qualified,
                conditional_route_supported=passed("26"),output_supported=passed("26") and passed("output"))


def execute():
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    from source_binding_resources import resources
    from source_binding_runtime import Runtime
    work, path, plan, run = context()
    selection = plan["analysis"]["selection"]
    stage = selection["stage"]
    start = time.monotonic()
    print(json.dumps({"stage":"verify local weight snapshot","completed":0,"run":run}),flush=True)
    if snapshot(MODEL)!=plan["acquisition"]["model_snapshot"]:
        raise ValueError("Qwen weight identity changed")
    material = read(path)
    arms = ARMS if stage=="confirm" else ("base",)
    if len(material)!=SIZES[stage]*8 or len({r["id"] for r in material})!=len(material):
        raise ValueError("incomplete crossed families")
    # Keep every arm at the same batch shape; one original eight-cell family per native microbatch.
    expanded = [dict(r,arm=a) for f in range(SIZES[stage]) for a in arms for r in material[8*f:8*f+8]]
    records, artifacts, discovery_keys, discovery_values = [], [], [], []
    chunk = 8*len(arms) if stage=="confirm" else 32
    count = (len(expanded)+chunk-1)//chunk
    def log(batch,block):
        done = 40*batch+block
        elapsed = time.monotonic()-start
        print(json.dumps(dict(stage=stage,step=done,total=40*count,elapsed=elapsed,
            eta=elapsed*(40*count-done)/done if done else None,**resources())),flush=True)
    with GpuLease():
        rt = Runtime("qwen14")
        rt.t.backends.cuda.matmul.allow_tf32 = False
        with rt.t.inference_mode(), (work/"case_results.jsonl").open("x",encoding="utf-8",buffering=1) as stream:
            for batch,lo in enumerate(range(0,len(expanded),chunk)):
                check_write(work,16*2**20,operation="原生读出与干预")
                log(batch,0)
                trials, native = evaluate_batch(rt,expanded[lo:lo+chunk],selection.get("candidate"),lambda b:log(batch,b))
                for trial in trials:
                    stream.write(json.dumps(dict(trial,run_id=run))+"\n")
                output = work/f"native-{batch:04}.npz"
                np.savez_compressed(output,case_id=np.asarray([r["id"] for r in trials]),arm=np.asarray([r["arm"] for r in trials]),**native)
                artifacts.append({"path":output.name,"role":"native_field","retention":"important"})
                save(work/"artifacts.json",{"schema":"research_test_files.v1","files":artifacts})
                records.extend(trials)
                if stage=="discovery":
                    discovery_keys.append(native["keys"])
                    discovery_values.append(native["early_v"])
    gate = core_gate([r for r in records if r["arm"]=="base"])
    report = dict(stage=stage,run_id=run,families=SIZES[stage],cases=len(material),arm_forwards=len(records),
                  gate=gate,stage_pass=gate["passed"])
    if stage=="discovery" and gate["passed"]:
        candidate = fit(material,dict(keys=np.concatenate(discovery_keys),early_v=np.concatenate(discovery_values)))
        save(work/"candidate.json",candidate)
        artifacts.append({"path":"candidate.json","role":"evidence","retention":"important"})
    if stage=="confirm":
        report["confirm"] = confirmation(records)
        report["stage_pass"] = report["confirm"]["conditional_route_supported"]
    sources = work/"research-scripts.zip"
    with zipfile.ZipFile(sources,"x",compression=zipfile.ZIP_DEFLATED) as archive:
        for rel in identities():
            archive.write(ROOT/rel,arcname=rel)
    artifacts.append({"path":sources.name,"role":"script","retention":"core"})
    save(work/"artifacts.json",{"schema":"research_test_files.v1","files":artifacts})
    save(work/"conclusion.json",report)
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    execute()
