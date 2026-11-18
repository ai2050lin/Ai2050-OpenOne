"""Replay the old H25 directional counterexamples at the original batch shape."""

from pathlib import Path
import time

import numpy as np

import kv_bf_study as S
import kv_bf_q4 as Q


A=S.A
C=S.C
PRE=C.OUT.with_name("kv_relay_20261115_v1")


def run():
    import torch
    C.setup()
    p=S.verify()
    assert S.model_identity("q4")==p["model_q4"]
    assert (C.OUT/"q14-confirm.json").exists()
    original=A.R.Base.Previous.load(PRE/"edited.json")
    rows=A.R.Base.Previous.load(PRE/"material-confirm.json")
    obs=A.R.Base.Previous.load(PRE/"behavior-confirm.json")["rows"]
    before=[]
    for file,h in sorted(original["shards"].items()):
        assert A.digest(PRE/file)==h
        with np.load(PRE/file) as doc:
            flag=doc["trio-q-dose-flags"][:,:,1,5]
            good=doc["good"]
            if np.any(flag[good]):
                before.append((int(file[7:10]),doc["ids"],good,flag,
                               doc["trio-q-dose-flags"][:,:,1],doc["trio-q-gain"][:,:,1]))
    assert before and not (C.OUT/"direction-replay.json").exists()
    files={}
    cases=[]

    def work(task):
        t0=time.monotonic()
        for i,(j,ids,valid,dir0,gates,gains) in enumerate(before,1):
            left=j*A.BATCH
            right=left+len(ids)
            batch=rows[left:right]
            assert np.array_equal(ids,[s["id"] for s in batch])
            z=S.arms_summary(*Q.run_batch(task,batch,obs[left:right],valid,Q.METHODS),p["dose"]["limits"])
            assert np.array_equal(z["native-gain"][:,:,1][valid],gains[valid])
            assert np.array_equal(z["native-flags"][:,:,1][valid],gates[valid])
            for k,r in enumerate(batch):
                if not valid[k] or not dir0[k].any():
                    continue
                cases.append(dict(id=r["id"],prompt=r["text"],direction_arms=np.flatnonzero(dir0[k]).tolist(),
                    directional={m:np.flatnonzero(z[m+"-flags"][k,:,1,5]).tolist() for m in Q.METHODS},
                    analytic={m:np.flatnonzero(z[m+"-flags"][k,:,1,2]).tolist() for m in Q.METHODS},
                    all_three_pass={m:bool(S.view_pass(z,m)[k]) for m in Q.METHODS}))
            name=f"historical-direction-{j:03d}.npz"
            files[name]=S.write_np(name,z)
            A.R.Base.Previous.stage_log("q4","historical-H25-direction",i,len(before),t0)

    torch.set_float32_matmul_precision("highest")
    A.with_task(work)
    A.save("direction-replay.json",dict(
        scope="Retrospective source of already observed Phase 2942 direction failures. No holdout metric includes these rows.",
        original=A.digest(PRE/"edited.json"),source=A.digest(Path(__file__)),shards=files,n=len(cases),cases=cases))


if __name__=="__main__":
    run()
