"""Blocked factorial material, native screen, freeze, and balanced confirmation."""

from itertools import batched
import time

import numpy as np

import kv_orth_native as Native
import kv_orth_run as C


A,B=C.A,C.B


def load(name):
    with np.load(C.OUT/name) as z:
        return {k:z[k] for k in z.files}


def persist(name,rows,run):
    if not (C.OUT/name).exists():
        B.write_np(name,run())
    z=load(name)
    assert z["ids"].tolist()==[x["id"] for x in rows],name
    return z


def natural(m,rows,name,binding):
    file=f"{name}-natural.json"
    if not (C.OUT/file).exists():
        z=Native.observe(m,rows,f"orth-{name}-natural")
        A.save(file,dict(binding=binding,rows=z))
    z=A.load(file)
    assert z["binding"]==binding and [x["id"] for x in z["rows"]]==[x["id"] for x in rows]
    return z["rows"]


def behavior_gate(rows,data):
    cells={};good=[]
    for r,o in zip(rows,data,strict=True):
        assert r["id"]==o["id"]
        k=r["id"].rsplit("-",1)[-1]
        s=cells.setdefault(k,dict(n=0,state=0,through_c=0,pq=0,fact=0,emitted=0))
        letters=r["token"]["letters"]
        allowed=o["matched"] and o["top"] in letters and o["requested_state_emitted"]
        s["n"]+=1
        s["state"]+=int(o["requested_state_emitted"])
        s["through_c"]+=int(o["matched"])
        s["pq"]+=int(allowed)
        factual=allowed and o["top"]==letters[r["code_f"]=="Q"]
        emitted=allowed and o["top"]==letters[r["code_s"]=="Q"]
        s["fact"]+=int(factual);s["emitted"]+=int(emitted)
        good.append(factual if r["fact"]==r["state"] else allowed)
    for k,s in cells.items():
        level=(s["fact"] if k[-2]==k[-1] else s["pq"])
        s["passed"]=level/s["n"]>=.90
    eligible=[r["family"] for r,g in zip(rows[::16],np.asarray(good).reshape(-1,16).all(1),strict=True) if g]
    return dict(passed=all(s["passed"] for s in cells.values()),cells=cells,eligible=eligible)


def resolved(rows,data,gate):
    keep=set(gate["eligible"])
    obs={x["id"]:x for x in data}
    return [dict(r,token=obs[r["id"]]["resolved"] or r["token"]) for r in rows if r["family"] in keep]


def op(things,kind="block",peer=None):
    return [*things,kind,peer]


def config(f):
    target=f["paths"]
    all_=list(map(op,target))
    names=("h18","fact","book")
    cfg={"sham":[],**{k:[b] for k,b in zip(names,all_,strict=True)},"parallel":all_[1:],"joint":all_}
    cfg["restore-h18"]=[op(target[0],"restore"),*all_[1:]]
    cfg["restore-parallel"]=[all_[0],*[op(r,"restore") for r in target[1:]]]
    cfg["restore-all"]=[op(r,"restore") for r in target]
    cfg["self"]=[op(r,"self") for r in target]
    for k,s,random in zip(names,target,f["controls"],strict=True):
        for j,h in enumerate(random,1):
            l,old,key=s
            cfg[f"{k}-t{j}"]=[op(s,"balance",h)]
            cfg[f"{k}-r{j}"]=[op([l,h,key],"balance",old)]
    return cfg


def physics(z,cfg):
    names=list(cfg)
    for key in ("self","restore-all"):
        j=names.index(key)
        assert np.array_equal(z["pq"][:,j],z["pq"][:,0]) and np.array_equal(z["top"][:,j],z["top"][:,0]),key
    assert np.isfinite(z["physical"]).all() and np.isfinite(z["alpha"]).all()
    return dict(drift=float(z["numeric_diff"].max()),invalid=z["ids"][~z["technical"]].tolist(),self_equal=True,native_restore_equal=True)


def walsh(matrix,rows,role):
    z=matrix*Native.polarity(rows,role).reshape((len(rows),)+(1,)*(matrix.ndim-1))
    return z.reshape(-1,2,8,*z.shape[1:]).mean(2)


def pilot(m,_):
    if not (C.OUT/"material-pilot.json").exists():
        A.save("material-pilot.json",C.material("pilot",m.tok,C.pools(m.tok)["pilot"]))
    rows=A.load("material-pilot.json")
    y=natural(m,rows,"pilot",B.sha("material-pilot.json"))
    g=behavior_gate(rows,y)
    A.save("pilot-gate.json",g)
    assert g["passed"] and len(g["eligible"])>=6,g
    r=resolved(rows,y,g)[:16]
    obs={x["id"]:x for x in y}
    f=dict(paths=[C.ANCHOR,[21,7,"fact"],[24,19,"book"]],controls=[[17,19],[9,17],[3,15]])
    c=config(f)
    z=Native.panel(m,r,c,"orth-pilot-edits",obs)
    check=physics(z,c)
    assert z["technical"].all(),check
    B.write_np("pilot-edits.npz",z)
    A.save("pilot-physics.json",check)
    print("Pilot",len(g["eligible"]),"families",check,flush=True)


def offers(sc,col):
    values=[]
    for m in (0,1):
        s=sc[:,col,:,m].copy()
        for l,h in values:s[C.LAYERS.index(l)]=-np.inf
        for _ in range(4):
            i,j=np.unravel_index(s.argmax(),s.shape)
            if [C.LAYERS[i],int(j),C.ROLES[col]]==C.ANCHOR or (C.LAYERS[i]==25 and j==18):
                s[i,j]=-np.inf;i,j=np.unravel_index(s.argmax(),s.shape)
            values.append((C.LAYERS[i],int(j)));s[i]=-np.inf
    return [[l,h,C.ROLES[col]] for l,h in values]


def discover(m,p):
    assert not (C.OUT/"freeze.json").exists() and not (C.OUT/"confirm-natural.json").exists()
    rows=A.load(p["materials"]["discover"]["file"])
    y=natural(m,rows,"discover",B.sha("plan.json"));gate=behavior_gate(rows,y)
    A.save("discover-gate.json",gate)
    assert gate["passed"] and len(gate["eligible"])>=16,gate
    use=resolved(rows,y,gate);known={x["id"]:x for x in y}
    sc=persist("discover-screen.npz",use,lambda:Native.panel(m,use,{"sham":[]},"screen"))["screen"].mean(0)
    options=[C.ANCHOR,*offers(sc,1),*offers(sc,2)]
    base={"sham":[],**{f"single-{i}":[op(s)] for i,s in enumerate(options)}}
    z=persist("discover-singles.npz",use,lambda:Native.panel(m,use,base,"orth-discover-edits",known))
    keep=np.repeat(z["technical"].reshape(-1,16).all(1),16)
    assert keep.sum()>=16*15
    diff=z["pq"][keep,:,1]-z["pq"][keep,:,0]
    use0=[r for r,k in zip(use,keep,strict=True) if k]
    fact,state=(walsh(diff,use0,k).mean((0,1)) for k in ("f","s"))
    fact=fact[0]-fact[1:];state=state[0]-state[1:]
    i=max(range(1,9),key=lambda j:fact[j]);j=max((j for j in range(9,17) if options[j][:2]!=options[i][:2]),key=lambda j:state[j])
    target=[options[k] for k in (0,i,j)]
    rng=np.random.default_rng(C.SEED);controls=[];doses=[]
    for l,h,key in target:
        v=sc[C.LAYERS.index(l),C.ROLES.index(key),:,1]
        pool=[int(k) for k in np.argsort(abs(np.log1p(v)-np.log1p(v[h])),kind="stable") if k!=h][:10]
        choice=rng.choice(pool,2,replace=False).tolist()
        controls.append(choice);doses.append(dict(native=float(v[h]),pool=pool,random=v[choice].tolist()))
    f=dict(plan=B.sha("plan.json"),behavior=B.sha("discover-natural.json"),time=C.O.stamp(),paths=target,controls=controls,doses=doses,
           discovery_n=int(keep.sum())//16,selection=[dict(site=s,loss_f=float(fact[k]),loss_s=float(state[k])) for k,s in enumerate(options)])
    cfg=config(f);f["arm_names"]=list(cfg)
    probe=persist("discover-physics.npz",use[:32],lambda:Native.panel(m,use[:32],cfg,"orth-frozen-physics",known))
    f["physics"]=physics(probe,cfg)
    f["discovery"]={k:B.sha(k) for k in ("discover-screen.npz","discover-singles.npz","discover-physics.npz","discover-gate.json")}
    A.save("freeze.json",f)
    print("Frozen",target,"balanced random",controls,flush=True)


def confirm(m,p):
    f=A.load("freeze.json")
    assert f["plan"]==B.sha("plan.json") and f["behavior"]==B.sha("discover-natural.json")
    assert all(B.sha(k)==v for k,v in f["discovery"].items())
    rows=A.load(p["materials"]["confirm"]["file"])
    y=natural(m,rows,"confirm",B.sha("freeze.json"));gate=behavior_gate(rows,y)
    A.save("confirm-gate.json",gate)
    base=dict(freeze=B.sha("freeze.json"),behavior=B.sha("confirm-natural.json"),gate=B.sha("confirm-gate.json"))
    if not (C.OUT/"confirm-alternate-format.json").exists():
        A.save("confirm-alternate-format.json",Native.replay_rejects(m,rows,y))
    base["alternate_format"]=B.sha("confirm-alternate-format.json")
    if not gate["passed"]:
        A.save("confirm.json",dict(state="behavior-gate-failed",**base));return
    use=resolved(rows,y,gate);known={x["id"]:x for x in y}
    cfg=config(f);assert list(cfg)==f["arm_names"]
    shards={};checks={};ids=[]
    all_=list(batched(use,C.SLAB));start=time.monotonic()
    C.N.progress("orth-heldout",0,len(all_),start)
    for idx,part in enumerate(all_,1):
        filename=f"confirm-native-{idx:03d}.npz"
        z=persist(filename,part,lambda:Native.panel(m,part,cfg,"orth-confirm-edits",known))
        checks[filename]=physics(z,cfg);shards[filename]=B.sha(filename)
        keep=np.repeat(z["technical"].reshape(-1,16).all(1),16)
        ids.extend(z["ids"][keep].tolist())
        C.N.progress("orth-heldout",idx,len(all_),start)
    A.save("confirm.json",dict(state="finished",technical=checks,accepted=ids,shards=shards,**base))
