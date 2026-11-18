"""Native pre-O output edits at the naturally reached Code: query."""

from itertools import batched
import random
import time

import numpy as np

import kv_natural_run as C


A, B, M = C.A, C.B, C.M


def select_controls(norms, groups):
    rng = random.Random(M.SEED)
    avg = {l:np.log(np.maximum(v,1e-6)).mean(0) for l,v in norms.items()}

    def draw(layer, target, excluded):
        score = avg[layer]
        pool = sorted((h for h in range(len(score)) if h not in excluded), key=lambda h:(abs(score[h]-score[target]),h))
        nearest = pool[:max(4,len(pool)//4)]
        return rng.choice(nearest),nearest

    sources, receivers, candidates = [],[],[]
    for _ in range(2):
        h,n = draw(25,18,{18,*sources})
        used = {4,34,*[t for p in receivers for t in p]}
        r0,n0 = draw(26,4,used)
        r1,n1 = draw(26,34,used | {h for h in range(len(avg[26])) if h//groups==r0//groups})
        sources.append(h)
        receivers.append([r0,r1])
        candidates.append(dict(source=n,receiver_4=n0,receiver_34=n1))
    return dict(source=sources,receivers=receivers,candidates=candidates,
                geometric_mean={str(l):np.exp(x).tolist() for l,x in avg.items()})


def arms(ctrl):
    block,restore,patch = "zero","self","native"
    command = {"sham":{},"u":{25:(block,[18])},"self-u":{25:(restore,[18])},
               "r4":{26:(block,[4])},"self-r4":{26:(restore,[4])},
               "r34":{26:(block,[34])},"self-r34":{26:(restore,[34])},
               "pair":{26:(block,[4,34])},"self-pair":{26:(restore,[4,34])},
               "u-pair":{25:(block,[18]),26:(block,[4,34])},"rescue":{25:(block,[18]),26:(patch,[4,34])}}
    for j,(src,pair) in enumerate(zip(ctrl["source"],ctrl["receivers"],strict=True),1):
        command.update({f"u-rand{j}":{25:(block,[src])},f"r4-rand{j}":{26:(block,pair[:1])},
                        f"r34-rand{j}":{26:(block,pair[1:])},f"pair-rand{j}":{26:(block,pair)},
                        f"rescue-rand{j}":{25:(block,[18]),26:(patch,pair)}})
    return command


def trace(m,layer,heads,p,q,k,v,before,after,write,with_residual,output):
    t = m.t
    a,b = after.shape[:2]
    kk = k[:,[h//m.groups for h in heads]]
    vv = v[:,[h//m.groups for h in heads]]
    qq = q[:,heads]
    visible = p["mask"][:,:,-1:].cuda().repeat(a,1,1,1)
    s = (qq.float()*kk.float()).sum(-1)*(128**-.5)
    alpha = s.masked_fill(~visible[:,0],-t.inf).softmax(-1)
    nslots = len(M.FIELDS)
    loc = p["slots"].cuda().repeat(a,1)
    picked = alpha.gather(2,loc[:,None].expand(-1,len(heads),-1))
    prompt = t.einsum("bht,bst->bhs",alpha,p["same"].cuda().repeat(a,1,1).float())
    vv_slot = vv.gather(2,loc[:,None,:,None].expand(-1,len(heads),-1,128)).float()
    slot_terms = picked[...,None]*vv_slot
    slot_norms = m.projections(layer,heads,slot_terms.transpose(1,2)).norm(dim=-1).transpose(1,2)
    queries=qq.reshape(a,b,len(heads),128).float()
    f = NativeFloat
    return dict(alpha=f(picked).reshape(a,b,len(heads),nslots),prompt_alpha=f(prompt).reshape(a,b,len(heads),nslots),
                slot_projected=f(slot_norms).reshape(a,b,len(heads),nslots),head_raw=f(before[:,:,heads]),
                head_edit=f(after[:,:,heads]),head_projected=f(m.projections(layer,heads,after[:,:,heads]).norm(dim=-1)),
                query_change=f((queries-queries[:1]).norm(dim=-1)),
                actual_projection_delta=f((write.float()-write[:1].float()).norm(dim=-1)),
                actual_residual_delta=f((with_residual.float()-with_residual[:1].float()).norm(dim=-1)),
                actual_block_delta=f((output.float()-output[:1].float()).norm(dim=-1)))


NativeFloat = C.Native.f32


def branch(m,layer,i,p,commands,x,mask,cs,q,k,v,o):
    t = m.t
    a,b,h = len(commands),len(x),m.heads
    base = o[:,-1]
    if i == 25:
        current = x[:,-1].unsqueeze(0).expand(a,-1,-1)
        before = base.unsqueeze(0).expand(a,-1,-1,-1)
        queries, keys, vals = q[:,:,-1:].repeat(a,1,1,1), k.repeat(a,1,1,1), v.repeat(a,1,1,1)
    else:
        current = p["branches"].cuda()
        s = tuple(z[:,-1:].repeat(a,1,1) for z in cs)
        queries, keys, vals = m.branch_qkv(layer,current.reshape(a*b,m.width),k.repeat(a,1,1,1),v.repeat(a,1,1,1),s)
        visible = mask[:,:,-1:].repeat(a,1,1,1)
        before = m.attention(layer,queries,keys,vals,visible).reshape(a,b,h,128)
    if i in C.TARGET:
        edited = before.clone()
        for j,edit in enumerate(commands.values()):
            if i in edit:
                typ,hs = edit[i]
                edited[j,:,hs] = 0
                if typ == "self":
                    edited[j,:,hs] = before[j,:,hs]
                elif typ == "native":
                    edited[j,:,hs] = base[:,hs]
    else:
        edited = before
    write = layer.self_attn.o_proj(edited.reshape(a,b,m.width))
    resid = current+write
    result = resid+layer.mlp(layer.post_attention_layernorm(resid))
    if i in C.TARGET:
        hs=C.TARGET[i]
        rec=trace(m,layer,hs,p,queries,keys,vals,before,edited,write,resid,result)
        rec["intact_head"]=NativeFloat(base[:,hs].unsqueeze(0).expand(a,-1,-1,-1))
        p["records"][i]=rec
    p["branches"] = result.cpu()


def discover(m,p):
    assert not (C.OUT/"confirm-behavior.json").exists()
    assert not (C.OUT/"freeze.json").exists()
    rows = A.load(p["materials"]["discover"]["name"])
    seen, norms = m.observe(rows,"discover")
    b = C.behavior(rows,seen)
    A.save("discover-behavior.json",b)
    if not b["passed"]:
        print("Discovery behavior gate failed",b["cells"],flush=True)
        return
    keep = np.array([r["family"] in b["eligible"] for r in rows])
    normals = {layer:np.stack([norms[r["id"]][layer] for r,ok in zip(rows,keep,strict=True) if ok]) for layer in C.TARGET}
    ctrl = select_controls(normals,m.groups)
    norm_id = B.write_np("discover-head-norms.npz",dict(ids=np.array([x["id"] for x,ok in zip(rows,keep,strict=True) if ok]),
                                layer25=normals[25],layer26=normals[26]))
    A.save("freeze.json",dict(time=C.Prior.stamp(), plan=B.sha("plan.json"), behavior=B.sha("discover-behavior.json"),
                              native=B.sha("stock-check.json"), norms=norm_id, control=ctrl, arm_order=list(arms(ctrl))))
    print("Frozen controls",ctrl["source"],ctrl["receivers"],flush=True)


def pack_result(m,p,commands,known):
    t=m.t
    letters=p["rows"][0]["token"]["letters"]
    branch=p["branches"].cuda()
    # Per-arm full vocabulary and actual greedy continuation token, no P/Q-only selection.
    logits=m.logits(branch)
    scores=NativeFloat(logits[...,letters])
    top=logits.argmax(-1).cpu().numpy()
    gold=[known[r["id"]] for r in p["rows"]]
    untreated=np.asarray([x["pq"] for x in gold],dtype=np.float32)
    assert np.array_equal(top[0],[x["top_token"] for x in gold]),(top[0],gold)
    assert np.max(abs(scores[0]-untreated))<=C.SHAM_TOL,(scores[0],untreated)
    for arm in ("self-u","self-r4","self-r34","self-pair"):
        ix=list(commands).index(arm)
        assert np.array_equal(top[ix],top[0]) and np.max(abs(scores[ix]-scores[0]))<=C.RESTORE_TOL
    out=dict(ids=np.array([r["id"] for r in p["rows"]]),pq=scores.swapaxes(0,1),top=top.T, untreated=untreated)
    for i,z in p["records"].items():
        out.update({str(i)+"-"+k:v.swapaxes(0,1) for k,v in z.items()})
    return out


def confirm(m,p):
    f=A.load("freeze.json")
    assert f["plan"]==B.sha("plan.json") and f["behavior"]==B.sha("discover-behavior.json")
    assert f["norms"]==B.sha("discover-head-norms.npz") and f["native"]==B.sha("stock-check.json")
    commands=arms(f["control"])
    assert list(commands)==f["arm_order"]
    rows=A.load(p["materials"]["confirm"]["name"])
    if not (C.OUT/"confirm-behavior.json").exists():
        seen,_=m.observe(rows,"confirm-behavior")
        b=C.behavior(rows,seen)
        b["freeze"]=B.sha("freeze.json")
        A.save("confirm-behavior.json",b)
    b=A.load("confirm-behavior.json")
    assert b["freeze"]==B.sha("freeze.json")
    if not b["passed"]:
        A.save("confirm.json",dict(state="behavior-gate-failed",shards={},behavior=B.sha("confirm-behavior.json")))
        print("Confirmation behavior gate failed",b["cells"],flush=True)
        return
    allowed=set(b["eligible"])
    known={x["id"]:x for x in b["rows"]}
    used=[dict(r,token=known[r["id"]].get("resolved_token",r["token"])) for r in rows if r["family"] in allowed]
    shard={}
    # Each part holds up to SLAB x BATCH rows. Part boundaries contain complete families.
    parts=list(batched(used,C.SLAB*C.BATCH))
    clock=time.monotonic()
    C.progress("intervene-shards",0,len(parts),clock,eligible_families=len(allowed))
    for part,records in enumerate(parts,1):
        name=f"confirm-{part:03d}.npz"
        if not (C.OUT/name).exists():
            results=[pack_result(m,x,commands,known) for _,ps in m.batches(list(records),"intervene",commands) for x in ps]
            B.write_np(name,{k:np.concatenate([z[k] for z in results]) for k in results[0]})
        with np.load(C.OUT/name) as z:
            assert np.array_equal(z["ids"],[x["id"] for x in records])
        shard[name]=B.sha(name)
        C.progress("intervene-shards",part,len(parts),clock,eligible_families=len(allowed))
    A.save("confirm.json",dict(state="finished", rows=len(used),families=len(allowed),shards=shard,
                           behavior=B.sha("confirm-behavior.json"),freeze=B.sha("freeze.json")))
