"""Resume the frozen phase, recording and rejecting native BF16 dose violations per view."""

from pathlib import Path
import sys

import kv_length_read as R
import kv_length_run as C


def install(p):
    source="tests/glm5/kv_length_read.py"
    path=C.ROOT/source
    assert C.A.digest(path)==p["sources"][source]
    code=path.read_text(encoding="utf-8")
    substitutions=(
        ('p["trace"]={}', 'p["trace"]={}\n            p["rounding_error"]=np.zeros((len(p["rows"]),len(cfg),3),dtype=np.float32)'),
        ('assert float(err.max())<.05,(head,how,float(err.max()))',
         'p["rounding_error"][:,j,idx]=np.maximum(p["rounding_error"][:,j,idx], F(err))'),
        ('technical=(error<=C.N.SHAM_TOL)&(top[:,0]==[known[k]["top"] for k in ids])',
         'technical=(error<=C.N.SHAM_TOL)&(top[:,0]==[known[k]["top"] for k in ids])\n                technical &= (p["rounding_error"]<.05).all(axis=(1,2))'),
        ('drift=error,dose=p["dose"],geom=p["geom"],**p["trace"]',
         'drift=error,dose=p["dose"],geom=p["geom"],rounding_error=p["rounding_error"],**p["trace"]'),
    )
    for original,recorded in substitutions:
        assert code.count(original)==1,original
        code=code.replace(original,recorded)
    # The only change is whether a frozen numerical failure aborts or is retained for exclusion.
    namespace={"__name__":"kv_length_read_measured_resume"}
    exec(compile(code,str(path),"exec"),namespace)
    R.run=namespace["run"]
    adapter=Path(__file__).resolve().relative_to(C.ROOT).as_posix()
    expected=dict(source=source,original_sha=p["sources"][source],adapter=adapter,adapter_sha=C.A.digest(C.ROOT/adapter),
                  freeze=C.B.sha("freeze.json"),primary_threshold=.05,unit="normalized BF16 head-output error",
                  policy="Keep raw measured arms. If any head edit reaches the predeclared threshold, reject that 16-view family.",
                  reason="First confirm attempt encountered (21, value, .0558721684). Original log and passed shards retained.")
    if (C.OUT/"resume-policy.json").exists():
        assert C.A.load("resume-policy.json")==expected
    else:
        C.A.save("resume-policy.json",expected)


def summarize():
    from kv_length_study import load
    led=C.A.load("confirm.json");f=C.A.load("freeze.json")
    out={};max_=0
    for k, h in led.get("files",{}).items():
        assert C.B.sha(k)==h
        z=load(k)
        if "rounding_error" not in z:
            continue
        e=z["rounding_error"];max_=max(max_,float(e.max()))
        failures=[]
        for i,j,site in zip(*(e>=.05).nonzero(),strict=True):
            assert not z["technical"][i]
            failures.append(dict(id=z["ids"][i],arm=f["arm_names"][j],site=int(site),error=float(e[i,j,site]),
                                 measured_pq=z["pq"][i,j].tolist(),native_pq=z["pq"][i,0].tolist()))
        if failures:out[k]=failures
    C.A.save("rounding-exclusions.json",dict(policy=C.B.sha("resume-policy.json"),confirm=C.B.sha("confirm.json"),
                       violations=out,max_measured=max_,n_views=len({v["id"] for cases in out.values() for v in cases})))


if __name__=="__main__":
    sys.modules.setdefault("kv_length_run",C)
    C.setup();install(C.verify());C.main("confirm");summarize()
