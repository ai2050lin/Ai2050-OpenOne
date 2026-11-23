"""Finish the same registered panel after exact native BF16 restoration repair."""
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile
ROOT = Path(os.environ["PYTHONPATH"].split(os.pathsep)[0]) if Path(__file__).name=="test.py" else Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"tests/glm5"),str(ROOT)]
from color_reuse_material import COLORS, ENTRY, HERE, MODEL, OUT, SEED, digest, read, save, sha, sources
from color_reuse_test import Native, arms, mean, score_items, summary, val
from deploy.storage_policy import checked_dir
from deploy.test_archive_client import runtime_identity
from shared.dist_protocol.test_archive import validate_plan
import numpy as np

THIS=HERE/"color_reuse_recovery.py"
DEST=OUT/"completion"
PARENT="record-f662410e-9d4c-4e73-adfd-858e438f1229"
MATERIAL=OUT/"study-questions.json"
RECORD=HERE/"result/routine"/("run-role-"+PARENT[7:])
SOURCE=OUT/"interrupted-evidence"
FILES=("candidate.json","case_results.jsonl","native-0000.npz","native-0001.npz","research-scripts.zip","output.log")


def version():
    return {**sources(),THIS.relative_to(ROOT).as_posix():sha(THIS)}


def prepare_recovery():
    checked_dir(DEST,128*2**20,operation="冻结恢复方案")
    if (DEST/"study-plan.json").exists():raise FileExistsError("Recovery plan already exists")
    checked_dir(SOURCE,80*2**20,operation="保留中断时独立候选和原生场")
    previous=read(OUT/"study-plan.json")
    review=read(RECORD/"review.json")
    if review["server_run_id"]!=PARENT or review["returncode"]==0:raise ValueError("not the interrupted run")
    old=Path(review["scratch"])
    for f in FILES:
        if not (SOURCE/f).exists():shutil.copyfile(old/f,SOURCE/f)
        if sha(SOURCE/f)!=sha(old/f):raise ValueError(f)
    copied={f:sha(SOURCE/f) for f in FILES}
    pack=read(MATERIAL);pack["recovery"]={"source_run":PARENT,"files":copied,"restore":"same-position exact original BF16 bytes"}
    qp=DEST/"study-questions.json";save(qp,pack)
    plan=previous
    plan["title"]+="（精确恢复与续跑）"
    plan["criteria"]+=" Preserve all already scored rows and frozen candidate verbatim. Only replace subtract/add FP32 restoration by exact original BF16 writeback. Recapture missing full-field first-item naming traces independently of scoring."
    plan["dataset"].update(revision="v1.1-exact-restore",content_sha256=sha(qp),questions_file_sha256=sha(qp),
        exposure="Same frozen split. 30 discovery, 228 calibration and 512 confirmation executions already completed. These same observations are counted once; not fresh independent replication.")
    plan["entrypoint_sha256"]=sha(THIS)
    plan["acquisition"].update(entrypoint_sha256=sha(THIS),code_sha256=digest(version()),questions_file_sha256=sha(qp),
                                environment_sha256=digest(runtime_identity()))
    plan["analysis"].update(method_version="color_reuse.v1.1-exact-restore",code_sha256=digest(version()),scoring_sha256=digest(plan["criteria"]))
    plan["analysis"]["selection"].update(source_run=PARENT,frozen_candidate_sha256=copied["candidate.json"])
    validate_plan(plan);save(DEST/"study-plan.json",plan)
    print("Recovery plan",sha(DEST/"study-plan.json"),flush=True)


def execute():
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    work=Path(os.environ["AI2050_TEST_WORK_DIR"]); qp=Path(os.environ["AI2050_TEST_QUESTIONS"])
    plan,pre,review=[read(qp.parent/n) for n in ("protocol.json","preflight.json","review.json")]
    pack=read(qp)
    if (not pre["preflight_id"] or pre["plan_sha256"]!=digest(plan) or review["archive_state"]!="pending"
            or Path(review["scratch"]).resolve()!=work.resolve() or plan["entrypoint_sha256"]!=sha(Path(__file__))
            or sha(qp)!=plan["dataset"]["questions_file_sha256"] or digest(version())!=plan["acquisition"]["code_sha256"]
            or pack["recovery"]["source_run"]!=PARENT or {f:sha(SOURCE/f) for f in FILES}!=pack["recovery"]["files"]):
        raise ValueError("recovery inputs differ from registered hashes")
    rows=pack["rows"]
    if digest([r["tokens"] for r in rows])!=plan["acquisition"]["inputs_sha256"]:raise ValueError("original prompts differ")
    print(json.dumps(dict(step="verify original snapshot and frozen recovery hashes",run=review["server_run_id"])),flush=True)
    if snapshot(MODEL)!=plan["acquisition"]["model_snapshot"]:raise ValueError("model changed")
    freeze=read(SOURCE/"candidate.json");sel=freeze["selected"]
    previous=[json.loads(s) for s in (SOURCE/"case_results.jsonl").read_text(encoding="utf8").splitlines()]
    done={(r["id"],r["arm"]) for r in previous if r["stage"]=="confirm"}
    if len(previous)!=770 or len(done)!=512 or sum(r["stage"]=="calibration" for r in previous)!=228:
        raise ValueError("unverified interruption boundary")
    state=dict(start=time.monotonic(),records=list(previous),native=["native-0000.npz","native-0001.npz"])
    for f in [*state["native"],"candidate.json"]:
        shutil.copyfile(SOURCE/f,work/f)
    shutil.copyfile(SOURCE/"case_results.jsonl",work/"case_results.jsonl")
    with zipfile.ZipFile(work/"research-scripts.zip","x",compression=zipfile.ZIP_DEFLATED) as z:
        for f in version():z.write(ROOT/f,arcname=f)
        for f in ("research-scripts.zip","output.log","case_results.jsonl"):
            z.write(SOURCE/f,arcname="interrupted/"+f)
    field={}
    with GpuLease(),torch.inference_mode(),(work/"case_results.jsonl").open("a",encoding="utf8",buffering=1) as stream:
        model=Native("qwen14")
        lost=[dict(r,arm="base",capture=True) for r in rows if r["task"]=="name" and r["stage"]=="confirm" and (r["id"],"base") in done]
        capture_state=dict(start=state["start"],records=[],native=state["native"])
        score_items(model,lost,io.StringIO(),capture_state,"recover-unsaved-fields-only",work)
        for stage in ("confirm","reference","tail","corpus"):
            jobs=[]
            for r in rows:
                if r["stage"]!=stage:continue
                variants=[dict(r,arm="base",capture=r["task"]=="name",corpus=stage=="corpus")]
                if stage=="confirm":variants.extend(arms(r,sel,("cut","ctrl","restore")))
                jobs.extend(v for v in variants if (v["id"],v["arm"]) not in done)
            data=score_items(model,jobs,stream,state,stage,work)
            if stage=="corpus":
                for c in (*COLORS,"turquoise","ochre"):
                    for group in ("colorword","background"):
                        vs=[np.mean(val(data[r["id"],sel["block"]])[sel["units"]]*sel["sign"]) for r in jobs if r["color"]==c and r["category"]==group]
                        field[c+"/"+group]=dict(n=len(vs),mean_signed_activation=mean(vs))
    cs=freeze["calibration"][f"{sel['block']}/{sel['budget']}"]
    result=summary(state["records"],sel,cs,freeze["gate"],pack["corpus"],field,review["server_run_id"])
    result["recovery"]=dict(source=PARENT,inherited=len(previous),inherited_confirmation=len(done),
                            repeated_fields_only=len(lost),newly_scored=len(state["records"])-len(previous))
    save(work/"conclusion.json",result)
    entries=[dict(path=n,role="native_field",retention="important") for n in state["native"]]
    entries.extend([dict(path="research-scripts.zip",role="script",retention="core"),dict(path="candidate.json",role="evidence",retention="important")])
    save(work/"artifacts.json",dict(schema="research_test_files.v1",files=entries))
    print(json.dumps(dict(step="finished",checks=result["checks"],cases=result["forwards_scored"])),flush=True)


if __name__=="__main__":
    if sys.argv[1:]==["prepare"]:prepare_recovery()
    elif sys.argv[1:]==["launch"]:
        from clt_role_reread_desktop import launch
        raise SystemExit(launch("study",THIS,DEST))
    elif not sys.argv[1:]:execute()
    else:raise SystemExit("use prepare or launch")
