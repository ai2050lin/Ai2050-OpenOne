"""Frozen Qwen3-14B native interventions at h18 and B26H36."""
import argparse
from datetime import datetime, timezone
import gc
import itertools
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "result/clt_b26_native_v1"
TMP = ROOT / "tests/glm5_temp/clt_b26_native_v1"
sys.path[:0] = [str(ROOT), str(HERE)]
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TEMP=str(TMP), TMP=str(TMP), TMPDIR=str(TMP))

import numpy as np
from deploy.storage_policy import checked_dir, check_write
import source_binding_design as D
from tools.power_analysis import check_phase

SEED = 202611170936
BLOCK, HEAD, KV = 26, 36, 7
CAP = 0.065419
FACTORS = list(itertools.product(range(2), repeat=4))
GROUPS = ("h_row", "h_id", "q_row", "q_id", "k_id", "k_row")
ARMS = ["base", "restore", "rope", "rope_control", "value", "value_control"] + [
    arm for group in GROUPS for arm in (group, group + "_control")
]


def save(name, data):
    file = OUT / name
    check_write(file, 4 << 20, operation="B26 evidence")
    assert not file.exists(), file
    temp = file.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(file)


def freeze():
    from transformers import AutoTokenizer
    from shared.dist_protocol.natural_register_runner import snapshot

    checked_dir(OUT, 512 << 20, operation="B26 experiment budget")
    checked_dir(TMP)
    assert not (OUT / "prereg.json").exists()
    model = D.MODELS["qwen14"]
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    old = HERE / "result/conditional_lookup_native_v1"
    revision = snapshot(model)
    assert revision == D.read(old / "prereg.json")["model_snapshot"]
    used = {name for split in ("pilot", "discovery", "confirm")
            for row in D.read(old / (split + "-materials.json")) for name in row["names"]}
    rng = np.random.default_rng(SEED)
    splits = {}
    for split, count in (("discovery", 64), ("confirm", 128)):
        rows = []
        for family in range(count):
            while True:
                names = ["".join(rng.choice(list("bdfgklmnprstvz")) + rng.choice(list("aeiou"))
                                 for _ in range(3)).capitalize() for _ in range(2)]
                if len(set(names)) == 2 and not used.intersection(names) and len({
                    len(tok.encode(" " + n, add_special_tokens=False)) for n in names
                }) == 1:
                    break
            used.update(names)
            values = list(map(str, rng.choice(D.WORDS, 4, replace=False)))
            ids = [tok.encode(v, add_special_tokens=False) for v in values]
            assert all(len(v) == 1 for v in ids)
            titles = ["Orion", "Vega"][::1 if family % 2 == 0 else -1]
            for cell, (r0, r1, identity, archive) in enumerate(FACTORS):
                content = "Two independent archives assign codes to the same entities. Return only the requested code word.\n"
                spans, code_spans, source_ids = [], [], []
                for block, row0 in enumerate((r0, r1)):
                    content += f"Archive {titles[block]}:\n"
                    for line in range(2):
                        source = line ^ row0
                        word = values[2 * block + line]
                        start = len(content)
                        content += f"{names[source]} has code "
                        code_spans.append((len(content), len(content) + len(word)))
                        content += word + "."
                        spans.append((start, len(content)))
                        source_ids.append(source)
                        content += "\n"
                content += "Entity to look up: "
                query = (len(content), len(content) + len(names[identity]))
                content += names[identity] + f"\nArchive to use: {titles[archive]}\nReturn only the code word."
                prompt = tok.apply_chat_template([dict(role="user", content=content)], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False) + "Answer:\n"
                enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
                offset = prompt.index(content)

                def positions(span):
                    left, right = span
                    return [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < offset + right and b > offset + left]

                records = list(map(positions, spans))
                assert len(set(sum(records, []))) == len(sum(records, []))
                for i in (0, 2):
                    p,q = records[i:i+2]
                    delta = q[0]-p[0]
                    assert p[-1]+delta < len(enc["input_ids"]) and q[0]-delta >= 0
                rr = [r0 ^ identity, r1 ^ identity]
                target = 2 * archive + rr[archive]
                rows.append(dict(id=f"{split}-{family:03}-{cell:02}", family=family, cell=cell, b=archive,
                                 identity=identity, r=rr, source_ids=source_ids, names=names,
                                 content=content, prompt=prompt, tokens=enc["input_ids"],
                                 records=records, code_pos=[max(positions(x)) for x in code_spans],
                                 entity_pos=max(positions(query)), target_record=target,
                                 gold_id=ids[target][0], choice_ids=[v[0] for v in ids],
                                 representative=cell == family % len(FACTORS)))
        for i in range(0, len(rows), 2):
            a, b = rows[i:i+2]
            assert a["tokens"][:a["entity_pos"]+1] == b["tokens"][:b["entity_pos"]+1]
        name = split + "-materials.json"
        save(name, rows)
        splits[split] = dict(families=count, n=len(rows), sha=D.sha(OUT / name))
    power = check_phase(.15, 128, p_d=.25, alpha=.025)
    assert power["pass"]
    deps = ["clt_b26.py", "clt_b26_prereg.md", "source_binding_runtime.py",
            "source_binding_resources.py", "source_binding_design.py"]
    save("prereg.json", dict(utc=datetime.now(timezone.utc).isoformat(), code={n: D.sha(HERE / n) for n in deps},
                            materials=splits, power=power, arms=ARMS, prior_candidate=D.sha(old / "candidate.json"),
                            provenance=dict(git_sha=D.git("rev-parse", "HEAD"), git_dirty=True, seed=SEED,
                                            model_path=str(model), model_revision=revision)))
    print("Design and material hashes frozen", flush=True)


def load_run(split):
    pre = D.read(OUT / "prereg.json")
    for name, digest in pre["code"].items():
        assert D.sha(HERE / name) == digest, name
    seal = None
    if split == "confirm":
        seal = D.read(OUT / "candidate.json")
        assert seal["prereg_sha"] == D.sha(OUT / "prereg.json")
        for name, digest in seal["files"].items():
            assert D.sha(OUT / name) == digest, name
    file = OUT / (split + "-materials.json")
    assert D.sha(file) == pre["materials"][split]["sha"]
    return pre, seal, D.read(file)


def execute(split):
    import torch as t
    import torch.nn.functional as F
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    from source_binding_resources import resources
    from source_binding_runtime import Runtime

    pre, seal, rows = load_run(split)
    prov = pre["provenance"]
    print("Verifying original model files", flush=True)
    assert snapshot(D.MODELS["qwen14"]) == prov["model_revision"]
    checked_dir(OUT, 128 << 20)
    checked_dir(TMP)
    rt = Runtime("qwen14")
    assert (rt.heads, rt.kv, rt.hd) == (40, 8, 128)
    start = time.time()
    nat = {key: [] for key in ("h", "q", "k", "source", "scores")}
    outputs = {arm: [] for arm in ARMS} if seal else {}
    parameters = {}
    selections = seal["indices"] if seal else {}
    write_path = OUT / (split + "-progress.jsonl")

    def progress(completed, total, block):
        check_write(OUT, 32 << 20, operation="B26 progress")
        elapsed = time.time() - start
        event = dict(split=split, block=block, completed=completed, total=total, elapsed=elapsed,
                     eta=elapsed / max(1, completed) * (total - completed),
                     cuda_peak=t.cuda.max_memory_allocated(), **resources())
        with write_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)
        assert elapsed < 40 * 60, "compute budget exceeded"

    def run_batch(shard, intervention):
        arms = ARMS if intervention else ["base"]
        ids, pos, (cos, sin), mask = rt.context(shard)
        hidden = {"base": rt.embed(ids)}
        width = ids.shape[1]
        pads = [width - len(row["tokens"]) for row in shard]
        ep = t.tensor([pad + row["entity_pos"] for pad, row in zip(pads, shard)], device="cuda")
        cp = t.tensor([[pad+i for i in row["code_pos"]] for pad, row in zip(pads, shard)], device="cuda")
        record = t.zeros((len(shard), 4, width), device="cuda", dtype=t.bool)
        for j, (pad, row) in enumerate(zip(pads, shard)):
            for k, seq in enumerate(row["records"]):
                record[j, k, [pad + i for i in seq]] = True
        gold = t.tensor([r["target_record"] for r in shard], device="cuda")
        trace = {arm: [] for arm in arms}
        coordinate_doses = {}

        def edit_h(values, arm):
            if arm not in ("restore",) and not arm.startswith("h_"):
                return
            doses = []
            for j, vector in enumerate(values):
                md = selections["h_row" if arm == "restore" else arm]
                original = vector[ep[j], md].clone()
                doses.append(original.float().norm())
                vector[ep[j], md] = 0
                if arm == "restore":
                    vector[ep[j], md] = original
            coordinate_doses[arm] = t.stack(doses).cpu().tolist()

        for b in range(rt.cfg.num_hidden_layers):
            if b == 18 and intervention:
                for arm in arms[1:]:
                    hidden[arm] = hidden["base"].clone()
                    edit_h(hidden[arm], arm)
            if b == 18 and not intervention:
                nat["h"].append(hidden["base"][t.arange(len(shard), device="cuda"), ep].cpu().view(t.uint16).numpy())
            layer = rt.block(b)
            attn = layer.self_attn
            collection = {}
            if b == BLOCK:
                qslice, kslice = slice(HEAD*128,(HEAD+1)*128), slice(KV*128,(KV+1)*128)
                if not parameters:
                    parameters.update(q=attn.q_proj.weight[qslice].cpu().view(t.uint16).numpy(),
                                      k=attn.k_proj.weight[kslice].cpu().view(t.uint16).numpy(),
                                      v=attn.v_proj.weight[kslice].cpu().view(t.uint16).numpy(),
                                      o=attn.o_proj.weight[:,qslice].cpu().view(t.uint16).numpy(),
                                      qnorm=attn.q_norm.weight.cpu().view(t.uint16).numpy(),
                                      knorm=attn.k_norm.weight.cpu().view(t.uint16).numpy())

                def projection(kind, mod, inputs, out):
                    state = inputs[0]
                    pick = qslice if kind == "q" else kslice
                    arm = collection["arm"]
                    group = {"q": "q_row", "k": "k_id"}[kind] if arm == "restore" else arm
                    if group.startswith(kind + "_"):
                        coords = selections[group]
                        active = state[:, -1:, :] if kind == "q" else state
                        change = F.linear(active[..., coords].float(), mod.weight[pick, coords].float())
                        if kind == "k":
                            change = change * collection["record"].any(1)[...,None]
                        target = out[:, -1:, pick] if kind == "q" else out[:, :, pick]
                        previous = target.clone() if arm == "restore" else None
                        altered = (target.float() - change).to(target.dtype)
                        collection["dose"] = (altered.float()-target.float()).flatten(1).norm(dim=-1)
                        target.copy_(altered)
                        if previous is not None:
                            target.copy_(previous)
                    if kind == "q" and not intervention:
                        nat["q"].append(state[:, -1].contiguous().cpu().view(t.uint16).numpy())
                    if kind == "k" and not intervention:
                        ix = t.arange(len(state), device="cuda")[:, None]
                        nat["k"].append(state[ix, collection["cp"]].contiguous().cpu().view(t.uint16).numpy())

                def qnorm(_, args, result):
                    collection["q"] = result[:, -1:, HEAD:HEAD+1].transpose(1, 2).clone()

                def knorm(_, args, result):
                    if collection["arm"] in ("rope", "rope_control"):
                        target = result[:, :, KV]
                        pos_table = collection["phase_positions"]
                        ix = t.arange(len(target), device="cuda")[:, None]
                        c, s = collection["cos"], collection["sin"]
                        shifted_c, shifted_s = c[ix,pos_table].float(), s[ix,pos_table].float()
                        c, s = c.float(), s.float()
                        dc, ds = shifted_c*c+shifted_s*s, shifted_s*c-shifted_c*s
                        changed = target.float()*dc + rt.mod.rotate_half(target.float())*ds
                        new = t.where((pos_table!=t.arange(width,device="cuda"))[...,None], changed.to(target.dtype), target)
                        collection["dose"] = (new.float()-target.float()).flatten(1).norm(dim=-1)
                        target.copy_(new)
                    collection["k"] = result[:, :, KV:KV+1].transpose(1,2).clone()

                def vhook(_, args, value):
                    collection["v"] = value[:, :, kslice].float()

                def o_pre(mod, args):
                    actual = args[0][:,-1,qslice]
                    qr, _ = rt.mod.apply_rotary_pos_emb(collection["q"], collection["q"], collection["cos"][:,-1:], collection["sin"][:,-1:])
                    kr, _ = rt.mod.apply_rotary_pos_emb(collection["k"], collection["k"], collection["cos"], collection["sin"])
                    raw = (qr[:,0].float()*kr[:,0].float()).sum(-1)*attn.scaling
                    alpha = raw.masked_fill(~collection["mask"][:,0,-1],float("-inf")).softmax(-1)
                    vec = alpha[...,None] * collection["v"]
                    contributions = t.einsum("ntd,nrt->nrd", vec, collection["record"].float())
                    mass = t.einsum("nt,nrt->nr", alpha, collection["record"].float())
                    ix = t.arange(len(raw),device="cuda")
                    g = collection["gold"]
                    native_error = (actual.float()-vec.sum(1)).abs().amax(1)
                    projected = F.linear(contributions, mod.weight[:,qslice].float())
                    norms = projected.norm(dim=-1)
                    arm = collection["arm"]
                    if arm in ("value", "value_control", "restore"):
                        slot = g ^ int(arm=="value_control")
                        paired = t.minimum(norms[ix,g],norms[ix,g^1])
                        gain = paired / norms[ix,slot].clamp_min(1e-20)
                        old = actual.clone() if arm=="restore" else None
                        new = (actual.float()-gain[:,None]*contributions[ix,slot]).to(actual.dtype)
                        collection["dose"] = F.linear((new.float()-actual.float()),mod.weight[:,qslice].float()).norm(dim=-1)
                        actual.copy_(new)
                        if old is not None:
                            actual.copy_(old)
                    summary = t.cat([mass, norms, native_error[:,None],
                                     collection.get("dose",t.zeros_like(native_error))[:,None]], dim=-1)
                    trace[arm].append(summary.cpu().numpy())

                handles = [
                    attn.q_proj.register_forward_hook(lambda m,i,o: projection("q",m,i,o)),
                    attn.k_proj.register_forward_hook(lambda m,i,o: projection("k",m,i,o)),
                    attn.q_norm.register_forward_hook(qnorm), attn.k_norm.register_forward_hook(knorm),
                    attn.v_proj.register_forward_hook(vhook), attn.o_proj.register_forward_pre_hook(o_pre),
                ]

            for arm, states in hidden.items():
                for lo in range(0, len(shard), 16):
                    hi = min(lo + 16, len(shard))
                    ss = slice(lo, hi)
                    if b == BLOCK:
                        collection.clear()
                        collection.update(arm=arm, record=record[ss], cp=cp[ss], gold=gold[ss],
                                          cos=cos[ss], sin=sin[ss], mask=mask[ss])
                        if arm in ("rope","rope_control"):
                            phase = t.arange(width,device="cuda").expand(hi-lo,-1).clone()
                            for j in range(lo,hi):
                                a = 2*(shard[j]["b"] ^ int(arm=="rope_control"))
                                p,q = shard[j]["records"][a:a+2]
                                pad = pads[j]
                                delta = q[0]-p[0]
                                phase[j-lo,[pad+i for i in p]] = t.tensor([pad+i+delta for i in p],device="cuda")
                                phase[j-lo,[pad+i for i in q]] = t.tensor([pad+i-delta for i in q],device="cuda")
                            collection["phase_positions"] = phase
                    states[ss] = layer(states[ss], attention_mask=mask[ss], position_ids=pos[ss],
                                       position_embeddings=(cos[ss],sin[ss]),use_cache=False)
            if b == BLOCK:
                for handle in handles:
                    handle.remove()
                collection.clear()
            del layer, attn
            if b in (0, 9, 17, 26, 39):
                progress(step*40 + b+1, total*40, b)
        finals = t.cat([v[:,-1] for v in hidden.values()])
        _, logits = rt.finish(finals)
        for j, arm in enumerate(arms):
            scores = rt.score(logits[j*len(shard):(j+1)*len(shard)],shard)
            att = np.concatenate(trace[arm])
            assert np.isfinite(att).all() and all(np.isfinite(s["scores"]).all() for s in scores)
            if not intervention:
                nat["source"].append(att)
                nat["scores"].extend(scores)
            else:
                for k, score in enumerate(scores):
                    score.update(route=att[k,:4].tolist(), write_norm=att[k,4:8].tolist(),
                                 sdpa_error=float(att[k,8]), dose=float(att[k,9]))
                    if arm.startswith("h_"):
                        score["dose"] = coordinate_doses[arm][k]
                outputs[arm].extend(scores)
        del hidden, logits
        gc.collect()
        t.cuda.empty_cache()

    natural_batches = [rows[i:i+256] for i in range(0,len(rows),256)]
    reps = [r for r in rows if r["representative"]]
    intervention_batches = [reps[i:i+32] for i in range(0,len(reps),32)] if seal else []
    total = len(natural_batches)+len(intervention_batches)
    with GpuLease(), t.inference_mode():
        for step, shard in enumerate(natural_batches):
            run_batch(shard, False)
        packed = {k:np.concatenate(v) for k,v in nat.items() if k!="scores"}
        assert len(packed["source"]) == len(rows)
        prefix_equal = sum(np.array_equal(packed["h"][i],packed["h"][i+1]) for i in range(0,len(rows),2))
        gates = dict(
            behavior=all(sum(nat["scores"][i]["gold_margin"]>0 and nat["scores"][i]["top1_correct"]
                             for i in range(cell,len(rows),16))/pre["materials"][split]["families"]>=.95
                         for cell in range(16)),
            technical=float(packed["source"][:,8].max())<=CAP and prefix_equal==len(rows)//2,
        )
        np.savez_compressed(OUT/(split+"-natural.npz"),**packed)
        if not seal:
            np.savez_compressed(OUT/"native-parameters.npz",**parameters)
        truth = np.array([r["target_record"] for r in rows])
        save(split+"-natural.json",dict(provenance=prov,gates=gates,prefix_equal=prefix_equal,
             max_error=float(packed["source"][:,8].max()),route_acc=float((packed["source"][:,:4].argmax(1)==truth).mean()),
             scores=nat["scores"],elapsed=time.time()-start,data_sha=D.sha(OUT/(split+"-natural.npz"))))
        if seal and all(gates.values()):
            for step, shard in enumerate(intervention_batches,len(natural_batches)):
                run_batch(shard, True)
                save(f"confirm-interventions-{step:02}.json",dict(provenance=prov,arms={a:outputs[a][-len(shard):] for a in ARMS}))
            save("confirm-interventions.json",dict(provenance=prov,arms=outputs,elapsed=time.time()-start))


def bits(array):
    return (array.astype(np.uint32)<<16).view(np.float32)


def effects(rows, arrays, weights):
    n = len(rows)//16
    r = np.array([r["r"] for r in rows]).reshape(n,16,2)
    b = np.array([r["b"] for r in rows]).reshape(n,16)
    identity = np.array([r["identity"] for r in rows]).reshape(n,16)
    selected = np.take_along_axis(r,b[...,None],axis=-1).squeeze(-1)
    kr = np.tile(np.array([0,1,0,1]),(n,64//4))
    ki = np.array([row["source_ids"] for row in rows]).reshape(n,64)

    def energy(z, label):
        contrast=(z*(2*label[...,None]-1)).mean(axis=1)
        return (contrast**2).mean(0)

    h = bits(arrays["h"]).reshape(n,16,-1)
    q = bits(arrays["q"]).reshape(n,16,-1)
    k = bits(arrays["k"]).reshape(n,64,-1)
    qgain = (bits(weights["q"])**2).sum(0)
    kgain = (bits(weights["k"])**2).sum(0)
    hrow,hid = energy(h,r[:,:,0])+energy(h,r[:,:,1]),energy(h,identity)
    qrow,qid,qb = energy(q,selected)*qgain,energy(q,identity)*qgain,energy(q,b)*qgain
    kid,krow = energy(k,ki)*kgain,energy(k,kr)*kgain
    return {
        "h_row": (hrow,hid,(h*h).mean((0,1))), "h_id": (hid,hrow,(h*h).mean((0,1))),
        "q_row": (qrow,qid+qb,(q*q).mean((0,1))*qgain),
        "q_id": (qid,qrow+qb,(q*q).mean((0,1))*qgain),
        "k_id": (kid,krow,(k*k).mean((0,1))*kgain), "k_row": (krow,kid,(k*k).mean((0,1))*kgain),
    }


def select():
    pre, _, rows = load_run("discovery")
    natural = D.read(OUT/"discovery-natural.json")
    assert all(natural["gates"].values()), natural["gates"]
    assert D.sha(OUT/"discovery-natural.npz")==natural["data_sha"]
    with np.load(OUT/"discovery-natural.npz") as z, np.load(OUT/"native-parameters.npz") as weights:
        factors = effects(rows,z,weights)
    indices, summaries = {}, {}
    rng = np.random.default_rng(SEED)
    for name,(signal,other,dose) in factors.items():
        chosen = np.argsort(other-signal,kind="stable")[:32]
        indices[name] = chosen.tolist()
    for name,(signal,other,dose) in factors.items():
        chosen = indices[name]
        excluded = set(chosen)
        partner = name[0] + ("_id" if name.endswith("_row") else "_row")
        excluded.update(indices[partner])
        control = []
        tie = rng.permutation(len(dose))
        for j in chosen:
            distances = abs(np.log(dose[tie]+1e-20)-np.log(dose[j]+1e-20))
            for x in tie[np.argsort(distances,kind="stable")]:
                if int(x) not in excluded:
                    excluded.add(int(x))
                    control.append(int(x))
                    break
        indices[name+"_control"] = control
        summaries[name] = dict(signal=float(signal[chosen].sum()),other=float(other[chosen].sum()),
                               control_signal=float(signal[control].sum()),control_other=float(other[control].sum()),
                               estimated_dose_ratio=float(np.sqrt(dose[chosen].sum()/dose[control].sum())))
    files = ["discovery-natural.npz","discovery-natural.json","native-parameters.npz"]
    save("candidate.json",dict(utc=datetime.now(timezone.utc).isoformat(), provenance=pre["provenance"],
         prereg_sha=D.sha(OUT/"prereg.json"),files={n:D.sha(OUT/n) for n in files},
         indices=indices,discovery_effects=summaries))
    print("Candidate sealed. Confirmation may now be read", flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["freeze","discovery","select","confirm"])
    stage=parser.parse_args().stage
    if stage=="freeze":
        freeze()
    elif stage=="select":
        select()
    else:
        execute(stage)
