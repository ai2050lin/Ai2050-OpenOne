"""Resume the frozen 32-row schedule after a memory-pressure interruption."""

from pathlib import Path
import time

import torch

from kv_route_contrast import configure
from kv_route_revised import arms
import kv_route_native as Native
import kv_route_run as R


def run():
    configure()
    torch.set_num_threads(4)
    plan = R.check()
    assert R.BATCH == plan["batch"] == 32
    selected = R.read("selection.json")
    assert selected["status"] == "frozen" and selected["plan_sha"] == R.sha(R.OUT/"plan.json")
    jobs = arms()
    expected = R.read("q4-model.json")["files"]
    assert not (R.OUT/"confirmation.json").exists()
    ledger = []
    with R.Base.Previous.GpuLease(), torch.inference_mode():
        start = time.monotonic()
        R.Base.Previous.stage_log("q4","memory-restart",0,4,start)
        model = R.load_model("q4")
        assert R.read("q4-model.json")["files"] == expected
        try:
            g = R.Task(model)
            R.Base.Previous.stage_log("q4","memory-restart",1,4,start)
            checked = []
            for suite in R.M.SUITES:
                rows = R.read(f"materials-{suite}.json")
                bpath = R.OUT/f"behavior-{suite}.json"
                torch.cuda.empty_cache()
                if bpath.exists():
                    record = R.read(bpath.name)
                    behavior, passed = record["rows"], record["passed"]
                else:
                    rows, behavior, passed = R.behavior(g,suite)
                if not passed:
                    ledger.append(dict(suite=suite,status="behavior gate failed",shards=[]))
                    continue
                batches = list(zip(R.Base.groups(rows,R.BATCH), R.Base.groups(behavior,R.BATCH), strict=True))
                shards = []
                now = time.monotonic()
                verify = suite == "multi"
                for i, (group, baseline) in enumerate(batches,1):
                    name = f"native-results-{suite}-{i:02d}.json"
                    existed = (R.OUT/name).exists()
                    if not existed or (verify and i==1):
                        torch.cuda.empty_cache()
                        newer = Native.one(g,group,baseline,suite,i,jobs)
                        if existed:
                            older = R.read(name)
                            for key in ("ids","paths","free","clock","source","distractor","attention","after_state","dose"):
                                assert newer[key] == older[key],(suite,i,key)
                            old_hash = R.sha(R.OUT/name)
                            backup = "before-restart-"+name
                            R.write(backup,older)
                            assert R.sha(R.OUT/backup)==old_hash
                            key_changes = sum(a[0]!=b[0] for a,b in zip(older["prompt_hash"],newer["prompt_hash"],strict=True))
                            value_changes = sum(a[1]!=b[1] for a,b in zip(older["prompt_hash"],newer["prompt_hash"],strict=True))
                            R.write(name,newer)
                            checked.append(dict(suite=suite,batch=i,current_sha=R.sha(R.OUT/name),previous_sha=old_hash,
                                                measured_fields_exact=True,prompt_k_layers_with_hash_changes=key_changes,
                                                prompt_v_layers_with_hash_changes=value_changes))
                        else:
                            R.write(name,newer)
                        R.Base.Previous.stage_log("q4",suite+"-restart-interventions",i,len(batches),now,
                                                  current_allocated=torch.cuda.memory_allocated(),
                                                  current_reserved=torch.cuda.memory_reserved())
                    raw = R.read(name)
                    assert raw["ids"] == [r["id"] for r in group]
                    assert set(raw["paths"]) == {a.name for a in jobs}
                    shards.append(dict(name=name,sha=R.sha(R.OUT/name)))
                ledger.append(dict(suite=suite,status="complete",shards=shards))
                if suite=="discovery":
                    assert shards==selected["discovery_shards"]
        finally:
            model.close()
    R.write("restart.json",dict(source_sha=R.sha(Path(__file__)),plan_sha=R.sha(R.OUT/"plan.json"),
                              check=checked,recovery="drop inactive CUDA allocator cache between batches, reverify existing first multi batch at the frozen readouts"))
    R.write("confirmation.json",dict(ledger=ledger,arms=[a.__dict__ for a in jobs],selection_sha=R.sha(R.OUT/"selection.json")))


if __name__ == "__main__":
    run()
