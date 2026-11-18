"""Freeze neutral branch and matched readers, then run held-out original responses."""

from itertools import batched
import time

import numpy as np

import kv_length_natural as Natural
import kv_length_read as Read
import kv_length_run as C


A,B=C.A,C.B


def load(name):
    with np.load(C.OUT/name) as q:return {k:q[k] for k in q.files}


def store(name,rows,generator):
    if not (C.OUT/name).exists():B.write_np(name,generator())
    z=load(name)
    assert z["ids"].tolist()==[x["id"] for x in rows]
    return z


def behavior(m,rows,stage,binding):
    name=f"{stage}-natural.json";bank=A.load("neutral-donors.json")
    if not (C.OUT/name).exists():
        A.save(name,dict(binding=binding,rows=Natural.observe(m,rows,bank,f"{stage}-neutral-greedy")))
    z=A.load(name)
    assert z["binding"]==binding
    assert [x["id"] for x in z["rows"]]==[x["id"] for x in rows]
    g=Natural.gate(rows,z["rows"],bank)
    A.save(f"{stage}-gate.json",g)
    return z["rows"],g


def active(rows,y,gate):
    matched=set(gate["eligible"]);known={x["id"]:x for x in y}
    return [dict(r,token=known[r["id"]]["adapted"] or r["token"]) for r in rows if r["family"] in matched]


def op(t,mode="value",other=None):return [*t,mode,other]


def arms(f):
    h,fact,book=C.ARMS
    cfg={"sham":[]}
    for s in ("address","value","key","query","kv","restore-k","restore-v","restore-kv","address-self","value-self","query-self"):
        cfg[s]=[op(h,s)]
    P=[op(fact),op(book)]
    cfg.update(fact=[P[0]],book=[P[1]],parallel=P,joint=[op(h),*P])
    cfg.update({"restore-h18":[op(h,"head-restore"),*P],"restore-parallel":[op(h),op(fact,"head-restore"),op(book,"head-restore")],
                "restore-all":[op(t,"head-restore") for t in C.ARMS]})
    for kind,source in zip(Read.TYPES,(h,h,fact,book),strict=True):
        how="address" if kind=="address" else "value"
        l,h0,src=source
        for j,hc in enumerate(f["random"][kind],1):
            cfg[f"{kind}-t{j}"]=[op(source,how,hc)]
            cfg[f"{kind}-r{j}"]=[op((l,hc,src),how,h0)]
    return cfg


def physical(z,cfg):
    index=list(cfg).index
    for a,b in (("address-self","sham"),("value-self","sham"),("query-self","sham"),("restore-kv","sham"),
                ("restore-all","sham"),("restore-k","value"),("restore-v","key")):
        assert np.array_equal(z["pq"][:,index(a)],z["pq"][:,index(b)]),(a,b)
        assert np.array_equal(z["top"][:,index(a)],z["top"][:,index(b)]),(a,b)
    # Float32 mathematical source formula versus the actually inserted BF16 preprojection write.
    errors={}
    for key in ("address","value"):
        g=z["geom"][:,index(key)]
        alpha=g[:,0:1];v,u,out=z[key].transpose(1,0,2)
        expected=alpha*(u-v) if key=="address" else -alpha*v
        error=np.linalg.norm(out-expected,axis=-1)/(1+np.linalg.norm(expected,axis=-1))
        errors[key]=float(error.max())
        assert errors[key]<.05,(key,errors[key])
    return dict(exact_restore=True,errors=errors,drift=float(z["drift"].max()),invalid=z["ids"][~z["technical"]].tolist())


def pilot(m,_):
    Natural.pilot(m)
    rows=A.load("material-pilot.json");y=A.load("pilot-natural.json")["rows"];g=A.load("pilot-natural-gate.json")
    assert g["passed"] and len(g["eligible"])>=6,g
    use=active(rows,y,g)[:16];ref={x["id"]:x for x in y}
    cfg=arms(dict(random={"address":[9,22],"value":[9,22],"fact":[17,10],"book":[24,30]}))
    q=Read.run(m,use,cfg,"length-pilot-physics",ref)
    checks=physical(q,cfg)
    assert q["technical"].all(),checks
    B.write_np("pilot-edits.npz",q)
    A.save("pilot-check.json",dict(passed=True,families=len(g["eligible"]),exact_pairs=g["exact_pairs"],physical=checks))
    print("Pilot native path passed",checks,flush=True)


def discover(m,p):
    assert not (C.OUT/"freeze.json").exists() and not (C.OUT/"confirm-natural.json").exists()
    r=A.load(p["panels"]["discover"]["file"]);y,g=behavior(m,r,"discover",B.sha("plan.json"))
    assert g["passed"] and len(g["eligible"])>=16,g
    rows=active(r,y,g);ref={x["id"]:x for x in y}
    data=store("discover-doses.npz",rows,lambda:Read.run(m,rows,{"sham":[]},"scan"))["scan"].mean(0)
    rng=np.random.default_rng(C.SEED);controls={};amount={}
    for i,(kind,(_,head,_)) in enumerate(zip(Read.TYPES,(C.ARMS[0],*C.ARMS),strict=True)):
        n=data[i];score=np.abs(np.log1p(n)-np.log1p(n[head]))
        pool=[int(j) for j in np.argsort(score,kind="stable") if j!=head][:10]
        m0=rng.choice(pool,2,replace=False).tolist()
        controls[kind]=m0;amount[kind]=dict(target=float(n[head]),random=n[m0].tolist(),pool=pool)
    f=dict(time=C.O.stamp(),plan=B.sha("plan.json"),behavior=B.sha("discover-natural.json"),donors=B.sha("neutral-donors.json"),
           random=controls,dose=amount,discovery_families=len(rows)//16,paths=C.ARMS)
    cfg=arms(f);f["arm_names"]=list(cfg)
    z=store("discover-native.npz",rows[:32],lambda:Read.run(m,rows[:32],cfg,"length-discover-physics",ref))
    f["physics"]=physical(z,cfg)
    f["source"]={k:B.sha(k) for k in ("discover-doses.npz","discover-native.npz","discover-gate.json")}
    A.save("freeze.json",f)
    print("Frozen physical control heads",controls,"neutral tokens",[d["text"] for d in A.load("neutral-donors.json")["sources"]],flush=True)


def confirm(m,p):
    f=A.load("freeze.json")
    assert f["plan"]==B.sha("plan.json") and f["behavior"]==B.sha("discover-natural.json") and f["donors"]==B.sha("neutral-donors.json")
    assert all(B.sha(k)==h for k,h in f["source"].items())
    r=A.load(p["panels"]["confirm"]["file"]);y,g=behavior(m,r,"confirm",B.sha("freeze.json"))
    base=dict(freeze=B.sha("freeze.json"),natural=B.sha("confirm-natural.json"),gate=B.sha("confirm-gate.json"))
    if not g["passed"]:
        A.save("confirm.json",dict(state="behavior-gate-failed",**base));return
    cfg=arms(f);assert list(cfg)==f["arm_names"]
    rows=active(r,y,g);ref={x["id"]:x for x in y}
    chunks=list(batched(rows,C.SLAB));now=time.monotonic();C.N.progress("length-holdout",0,len(chunks),now)
    checks={};hashes={};accepted=[]
    for j,chunk in enumerate(chunks,1):
        name=f"confirm-native-{j:03d}.npz"
        z=store(name,chunk,lambda:Read.run(m,chunk,cfg,"length-confirm-read",ref))
        checks[name]=physical(z,cfg);hashes[name]=B.sha(name)
        take=np.repeat(z["technical"].reshape(-1,16).all(-1),16)
        accepted.extend(z["ids"][take].tolist())
        C.N.progress("length-holdout",j,len(chunks),now)
    A.save("confirm.json",dict(state="finished",files=hashes,checks=checks,accepted=accepted,**base))
