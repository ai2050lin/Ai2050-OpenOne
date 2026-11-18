"""Build, freeze and measure the L20H5 relay study in ordered stages."""

from collections import defaultdict
from pathlib import Path
import time

import numpy as np

import kv_relay_run as C
import kv_relay_native as N


A=C.A
R=A.R
SOURCES=tuple("tests/glm5/"+s for s in (
    "kv_relay_run.py","kv_relay_native.py","kv_relay_study.py","kv_address_run.py","kv_address_native.py",
    "kv_address_materials.py","kv_alpha_run.py","kv_route_run.py","kv_alpha_audit.py","own_state_run.py",
    "own_state_intervene.py","condition_chain_run.py","source_binding_resources.py",
))+("deploy/storage_policy.py","deploy/storage_policy.json","shared/dist_protocol/mechanism_trace.py","tools/power_analysis.py")
UNIT_BUDGET=192


def np32(x):
    return x.detach().cpu().float().numpy().copy()


def check(p):
    return R.Base.Previous.check_write(C.OUT/p)


def sha(p):
    return A.digest(C.OUT/p)


def save_np(p,**fields):
    np.savez_compressed(check(p),**fields)
    return sha(p)


def engine_pilot(task):
    rows=A.load("pilot-material.json")[:A.BATCH]
    obs=A.load("pilot-behavior.json")["rows"][:A.BATCH]
    good=np.array([o["joint"] for o in obs])
    r=N.Relay(task,rows,obs,good)
    try:
        r.natural()
        r.prepare(np.arange(32),np.arange(32,64))
        out=[(a.name,r.edited(a)) for a in N.arms(0)]
        actual=out[0][1]
        err=np.abs(actual["alpha"][good]-np32(r.forecast["full"]["alpha"])[:,0][good]).max()
        assert np.array_equal(actual["before"][good],np32(r.forecast["full"]["qk"])[:,0][good])
        assert np.array_equal(actual["before"][good],actual["after"][good])
        assert float(err)<3e-6
        A.save("engine-pilot.json",dict(arms=[n for n,_ in out],forecast_max_alpha_error=float(err),n=int(good.sum())))
    finally:
        r.close()


def plan():
    assert not (C.OUT/"plan.json").exists()
    assert A.load("pilot-behavior.json")["passed"]
    assert A.load("engine-pilot.json")["forecast_max_alpha_error"]<3e-6
    tok=A.Base.tok()
    parts={}
    all_words={}
    prior=C.OUT.with_name("kv_address_20261115_v1")
    for stage in ("discover","confirm"):
        rows=C.design_material(stage,tok)
        for row in rows:
            A.encoded(tok,row)
        by={r["id"]:r for r in rows}
        assert len(by)==len(rows)
        words={r[k] for r in rows for k in ("state","note","alt")}
        all_words[stage]=words
        name=f"material-{stage}.json"
        A.save(name,rows)
        parts[stage]=dict(name=name,sha=sha(name),families=len(rows)//8,n=len(rows))
    assert not all_words["discover"]&all_words["confirm"]
    for stage in ("discover","confirm"):
        p=R.Base.Previous.load(prior/f"material-{stage}.json")
        prior_word={r[k] for r in p for k in ("state","note","alt")}
        assert not prior_word&all_words["confirm"], sorted(prior_word&all_words["confirm"])
    model=A.Base.collect_identity()
    snap=A.Base.official_id(model)
    prev=R.Base.Previous.load(prior/"study-plan.json")
    assert snap==prev["model"]
    limits=R.Base.Previous.load(C.OUT.with_name("kv_alpha_20261115_v2")/"plan.json")
    power=A.Base.check_phase(.085,192,p_d=.10,alpha=.025)
    assert power["pass"]
    A.save("plan.json",dict(run=C.RUN,planned=C.time_mark(),materials=parts,model=model,snapshot=snap,
        versions=A.Base.versions(),provenance=dict(**A.Base.git_identity(),model_revision=snap,model_path=str(A.Base.MODEL),seed=C.SEED),
        sources={s:A.digest(C.ROOT/s) for s in SOURCES},power=power,batch=A.BATCH,cap=2.05,conditions=N.CONDITIONS,
        scope="answer query only; prior State/Note historical KV immutable, newly formed query and current Key may vary",
        behavior="8 cells >=90% plus all eight views in a family for mechanism",
        prediction=dict(inputs="Natural prefix and native anchors only. Build exact L20 O write for each source from the natural call. Freeze all heldout forecasts before target edited forward.",
                        branches=["exact local L20 residual+MLP","direct residual with natural MLP","only edited MLP with natural residual",
                                  "direct plus selected 192 unit changes","direct plus 192 matched-budget unselected units","discovery cell-mean shift"],
                        discovery="Rank by mean squared native unit change times squared projection leverage into L21 H19/H25 Q and two current K groups. No holdout read.",
                        local_max_alpha_tolerance=3e-6,primary="H19/25 live alpha log-ratio and cap balanced accuracy, per-family and both slots",
                        sparse_gate_relative_rmse_to_direct=.90),
        interventions=dict(arms=[x.__dict__ for x in N.arms(0)],
                           meaning="Use natural donor to block a node with source on; copy source-edited mediator into same-prefix source-off run for restoration.",
                           expected="Q dominates gain change when cached keys are fixed; current K may alter partition but not State/Note odds on its own.",
                           primary="Mean alpha deviation removed/restored, head readouts and P/Q kept separate. Q19-only versus Q19+Q25 and current K to test remaining failure."),
        triad=dict(target="Original frozen h* = min(max natural pair alpha, 2*min natural pair alpha) per head; common donor directions and amplitude across branches, no fresh balance.",
                   minimum_full_families=153,equiv=.25,dose_formula=limits["hypothesis"]),
        previous=dict(phase2941=prev["run"],design=A.digest(prior/"study-plan.json"),
                      failure=A.digest(prior/"relay-gain-audit.json")),
        pilot={n:sha(n) for n in ("pilot-material.json","pilot-behavior.json","engine-pilot.json")},
        data_budget="≤380 MiB, preserve scalar and selected native fields, plus sampled source-to-target native arrays."))


def verify():
    p=A.load("plan.json")
    assert p["versions"]==A.Base.versions()
    assert all(A.digest(C.ROOT/k)==s for k,s in p["sources"].items())
    assert all(sha(p["materials"][k]["name"])==p["materials"][k]["sha"] for k in p["materials"])
    assert all(sha(k)==s for k,s in p["pilot"].items())
    return p


def behavior(task,stage,p):
    rows=A.load(p["materials"][stage]["name"])
    name=f"behavior-{stage}.json"
    if (C.OUT/name).is_file():
        obs=A.load(name)
    else:
        obs=A.behavioral(task,rows,stage+"-behavior")
        A.save(name,obs)
    family=defaultdict(list)
    for r,x in zip(rows,obs["rows"],strict=True):
        family[r["family"]].append(x["joint"])
    assert all(len(v)==8 for v in family.values())
    keep=np.array([all(family[r["family"]]) for r in rows])
    if not obs["passed"]:
        A.save(f"{stage}-stopped.json",dict(reason="behavior gate failed",cells=obs["cells"],design=sha("plan.json")))
        return rows,None,None
    return rows,obs["rows"],keep


def discovery():
    import torch
    p=verify()
    assert not (C.OUT/"behavior-confirm.json").exists() and not (C.OUT/"discover.json").exists()
    assert A.Base.collect_identity()==p["model"]
    def work(task):
        rows,obs,keep=behavior(task,"discover",p)
        if obs is None:
            return
        unit_energy=np.zeros(9728,dtype=np.float64)
        native_leverage=None
        pools=defaultdict(list)
        t0=time.monotonic()
        for j,(batch,ob) in enumerate(zip(A.groups(rows),A.groups(obs),strict=True)):
            torch.cuda.empty_cache()
            good=keep[j*A.BATCH:j*A.BATCH+len(batch)]
            if not good.any():
                continue
            r=N.Relay(task,batch,ob,good)
            try:
                r.natural()
                r.prepare()
                last=r.l21.self_attn
                if native_leverage is None:
                    q=last.q_proj.weight.reshape(32,128,2560)[[19,25]].reshape(-1,2560)
                    k=last.k_proj.weight.reshape(8,128,2560)[[4,6]].reshape(-1,2560)
                    mat=torch.cat((q,k)).float() @ r.l20.mlp.down_proj.weight.float()
                    native_leverage=np32(mat.square().sum(0))
                diff=np32(r.z_edit-r.z[:,None])
                unit_energy+=(diff[good].astype(np.float64)**2).sum(axis=(0,1))
                alpha=np32(r.forecast["full"]["alpha"]-r.alpha0[:,None])
                for i,row in enumerate(batch):
                    if good[i]:
                        pools[f"{row['context']}{row['order']}"].append(alpha[i])
                if j==0:
                    a=r.edited(N.arms(0)[0])
                    assert np.array_equal(a["before"][good],np32(r.forecast["full"]["qk"])[:,0][good])
            finally:
                r.close()
            R.Base.Previous.stage_log("q4","discover-relay",j+1,len(rows)//A.BATCH,t0)
        utility=unit_energy*native_leverage
        save_np("discovery-units.npz",energy=unit_energy,leverage=native_leverage,score=utility)
        A.save("discover.json",dict(design=sha("plan.json"),fields_sha=sha("discovery-units.npz"),
                                   n=int(keep.sum()),cell_means={k:np.mean(v,axis=0).tolist() for k,v in pools.items()},
                                   behavior_sha=sha("behavior-discover.json")))
    A.with_task(work)


def freeze():
    verify()
    assert not (C.OUT/"behavior-confirm.json").exists() and not (C.OUT/"freeze.json").exists()
    prior=A.load("discover.json")
    assert prior["design"]==sha("plan.json") and prior["fields_sha"]==sha("discovery-units.npz")
    with np.load(C.OUT/"discovery-units.npz") as d:
        utility=d["score"]
    units=np.argsort(-utility,kind="stable")[:UNIT_BUDGET]
    rest=np.setdiff1d(np.arange(len(utility)),units)
    rng=np.random.default_rng(C.SEED)
    null=np.sort(rng.choice(rest,UNIT_BUDGET,replace=False))
    A.save("freeze.json",dict(time=C.time_mark(),plan=sha("plan.json"),discovery=sha("discover.json"),
                             units=np.sort(units).tolist(),random_units=null.tolist(),score_fraction=float(utility[units].sum()/utility.sum()),
                             baseline=prior["cell_means"],source="No confirmation natural responses yet"))


def eligible_freeze():
    p=verify()
    f=A.load("freeze.json")
    assert f["plan"]==sha("plan.json") and f["discovery"]==sha("discover.json")
    return p,f


def group_input(task,rows,obs,keep,stage,func):
    import torch
    t0=time.monotonic()
    jobs=A.groups(rows)
    for j,(batch,ob) in enumerate(zip(jobs,A.groups(obs),strict=True)):
        good=keep[j*A.BATCH:j*A.BATCH+len(batch)]
        if not good.any():
            continue
        torch.cuda.empty_cache()
        r=N.Relay(task,batch,ob,good)
        try:
            func(j,batch,r,good)
        finally:
            r.close()
        R.Base.Previous.stage_log("q4",stage,j+1,len(jobs),t0)


def forecast_record(r,batch,good,f,sample=False):
    t=r.t
    r.natural()
    r.prepare(f["units"],f["random_units"])
    columns=np32(r.forecast["full"]["qk"])
    qk=[]
    kq=[]
    for j in range(4):
        full=r.forecast["full"]["qk"][:,j]
        qk.append(np32(r.alpha_at_qk(t.stack((full[:,:,0],r.received[:,:,1]),dim=2))[0]))
        kq.append(np32(r.alpha_at_qk(t.stack((r.received[:,:,0],full[:,:,1]),dim=2))[0]))
    rec={k:np32(v["alpha"]) for k,v in r.forecast.items()}
    rec.update(
        natural=np32(r.alpha0),good=good,ids=np.array([x["id"] for x in batch]),native_scores=np32(r.base_score[:,r.endpoints]),
        q_only=np.stack(qk,1),k_only=np.stack(kq,1),
        na_qk=np32(r.received).astype(np.float32),full_qk=columns,
        strata=np.array([f"{x['context']}{x['order']}" for x in batch]),
        direct_norm=np32((r.anchors-r.a[:,None]).float().norm(dim=-1)),
        mlp_norm=np32((r.m_edit-r.m[:,None]).float().norm(dim=-1)))
    if sample:
        def bf(x):
            return x[:4].contiguous().view(t.uint16).cpu().numpy().copy()
        save_np("native-traces.npz",ids=rec["ids"][:4],L20_input=bf(r.x),L20_attention_residual=bf(r.a),
                L20_attention_residual_edited=bf(r.anchors),L20_units=bf(r.z),L20_units_edited=bf(r.z_edit),
                L20_mlp=bf(r.m),L20_mlp_edited=bf(r.m_edit),L21_input=bf(r.prev),
                keys=bf(r.bank),anchor_projected=bf(r.received),live=good[:4],positions=r.positions[:4].cpu().numpy())
    return rec


def forecast():
    p,f=eligible_freeze()
    assert not (C.OUT/"forecast.json").exists() and not list(C.OUT.glob("edited-*.npz"))
    assert A.Base.collect_identity()==p["model"]
    def work(task):
        rows,obs,keep=behavior(task,"confirm",p)
        if obs is None:
            return
        shards={}
        def one(j,batch,r,good):
            name=f"forecast-{j:03d}.npz"
            shards[name]=save_np(name,**forecast_record(r,batch,good,f,sample=j==0))
        group_input(task,rows,obs,keep,"confirm-upstream-forecasts",one)
        A.save("forecast.json",dict(time=C.time_mark(),freeze=sha("freeze.json"),behavior=sha("behavior-confirm.json"),
                                   n=int(keep.sum()),shards=shards,boundary="Only unedited native execution and local arithmetic were run before this record."))
    A.with_task(work)


def dose_checks(d,cfg,good):
    norm=lambda x:np.linalg.norm(x.astype(np.float64),axis=-1)
    want,actual,analytic,gain=(d[k] for k in ("want","actual","analytic","gain"))
    n=norm(want)
    u=norm(actual)
    cosine=np.sum(actual*want,axis=-1)/np.maximum(n*u,1e-12)
    labels=["small","capped","analytic","native","collapsed","direction"]
    flags=np.stack([
        n<cfg["norm_floor"],
        gain>2.05+2e-6,
        norm(analytic-want)>np.maximum(cfg["analytic_absolute_error_floor"],cfg["max_planned_relative_error"]*n),
        norm(actual-want)>np.maximum(cfg["actual_absolute_error_floor"],cfg["max_actual_relative_error"]*n),
        u<cfg["min_actual_norm_fraction"]*n,
        cosine<cfg["min_actual_cosine"]],-1)&good[:,None,None]
    return labels,flags,n,u


def edit():
    p,f=eligible_freeze()
    fore=A.load("forecast.json")
    assert fore["freeze"]==sha("freeze.json") and fore["behavior"]==sha("behavior-confirm.json")
    assert A.Base.collect_identity()==p["model"]
    cfg=p["triad"]["dose_formula"]
    mark=sha("forecast.json")
    def work(task):
        rows,obs,keep=behavior(task,"confirm",p)
        assert obs is not None
        ledger={}
        def one(j,batch,r,good):
            src=f"forecast-{j:03d}.npz"
            dest=f"edited-{j:03d}.npz"
            assert sha(src)==fore["shards"][src]
            if (C.OUT/dest).exists():
                with np.load(C.OUT/dest) as prev:
                    assert str(prev["forecast"])==mark and np.array_equal(prev["ids"],[x["id"] for x in batch])
                ledger[dest]=sha(dest)
                return
            with np.load(C.OUT/src) as ref:
                generated=forecast_record(r,batch,good,f)
                for key in ref.files:
                    assert np.array_equal(generated[key],ref[key]),(key,j)
            records=defaultdict(dict)
            for cell in range(4):
                for a in N.arms(cell):
                    value=r.edited(a)
                    records[a.name][cell]=value
                    if a.name=="source":
                        assert np.array_equal(value["before"][good],generated["full_qk"][:,cell][good])
            payload=dict(forecast=np.array(mark),ids=generated["ids"],good=good)
            for name,x in records.items():
                payload[name+"-alpha"]=np.stack([v["alpha"] for v in x.values()],1)
                payload[name+"-score"]=np.stack([v["score"] for v in x.values()],1)
                drift=np.stack([
                    np.linalg.norm((v["before"]-generated["na_qk"]).astype(np.float64),axis=-1) for v in x.values()],1)
                payload[name+"-drift"]=drift.astype(np.float32)
                if name=="source":
                    payload["actual-qk"]=np.stack([v["before"] for v in x.values()],1)
                if name.startswith("trio"):
                    d=[v["dose"] for v in x.values()]
                    checks=[dose_checks(y,cfg,good) for y in d]
                    bits=np.stack([b for _,b,_,_ in checks],1)
                    labels=checks[0][0]
                    pair=[]
                    for a,b in ((0,2),(1,3)):
                        delta=np.linalg.norm((d[a]["actual"]-d[b]["actual"]).astype(np.float64),axis=-1)
                        pair.append(delta>np.maximum(cfg["actual_absolute_error_floor"],
                            cfg["max_actual_paired_relative_error"]*np.maximum(checks[a][2],checks[b][2])))
                    payload[name+"-dose-flags"]=bits
                    payload[name+"-pair-flags"]=np.stack(pair,1)
                    payload[name+"-gain"]=np.stack([v["gain"] for v in d],1)
                    payload[name+"-want"]=np.stack([x[2] for x in checks],1)
                    payload[name+"-actual"]=np.stack([x[3] for x in checks],1)
                    payload["dose-labels"]=np.array(labels)
            ledger[dest]=save_np(dest,**payload)
        group_input(task,rows,obs,keep,"relay-native-edits",one)
        A.save("edited.json",dict(forecast=mark,time=C.time_mark(),n=int(keep.sum()),shards=ledger,
                                 frozen_target_retained=True))
    A.with_task(work)


commands={
    "engine-pilot":lambda:A.with_task(engine_pilot),
    "plan":plan,
    "discover":discovery,
    "freeze":freeze,
    "forecast":forecast,
    "edit":edit,
}
