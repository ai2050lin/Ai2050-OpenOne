"""Freeze, run and resume the original-Qwen3-4B weighted-write experiment."""

import argparse
from datetime import datetime
import importlib.metadata
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

import kv_route_run as R
import kv_alpha_materials as M
from tools.power_analysis import check_phase


ROOT = R.Base.ROOT
OUT = ROOT/"tests/glm5/result/kv_alpha_20261115_v1"
TMP = ROOT/"tests/glm5_temp/kv_alpha_20261115_v1"
RUN = "KV-ALPHA-20261115-v1"
SOURCE_DIR = Path(__file__).parent
MODEL = ROOT/"models/hf"/R.Base.MODELS["q4"]
BATCH = 16
CAP_BYTES = 335 * 2**20
PACKAGES = ("torch", "transformers", "safetensors", "numpy", "scipy")
SOURCES = (
    "tests/glm5/kv_alpha_materials.py","tests/glm5/kv_alpha_run.py","tests/glm5/kv_alpha_native.py",
    "tests/glm5/kv_route_run.py","tests/glm5/kv_route_native.py","tests/glm5/own_state_run.py",
    "tests/glm5/own_state_intervene.py","tests/glm5/condition_chain_run.py","tests/glm5/b36_switch14_engine.py",
    "tests/glm5/source_binding_resources.py","shared/dist_protocol/mechanism_trace.py",
    "tests/glm5/condition_chain_materials.py","deploy/storage_policy.py","deploy/storage_policy.json",
    "tests/glm5/kv_route_materials.py","tests/glm5/own_state_explicit.py","tests/glm5/own_state_materials.py",
    "tests/glm5/b36_switch14.py","tools/power_analysis.py",
)


def configure(output=None):
    global OUT
    if output is not None:
        OUT = Path(output).resolve()
        assert OUT.is_relative_to((ROOT/"tests/glm5/result").resolve())
    R.OUT, R.TEMP, R.RUN = OUT, TMP, RUN
    check = R.Base.Previous.check_write
    check(OUT,CAP_BYTES,operation=RUN)
    check(TMP,CAP_BYTES,operation=RUN)
    R.setup()


def versions():
    return {a:importlib.metadata.version(a) for a in PACKAGES}


def git_identity():
    base = ["git","-c",f"safe.directory={ROOT.as_posix()}","-C",str(ROOT)]
    def git(*args):
        return subprocess.check_output([*base,*args],text=True,encoding="utf-8",errors="replace").strip()
    return dict(git_sha=git("rev-parse","HEAD"),git_dirty=bool(git("status","--porcelain","-uno")))


def collect_identity():
    return {p.name:R.sha(p) for p in sorted(MODEL.iterdir())
            if p.suffix in {".json",".safetensors",".txt",".model"}}


def official_id(model_files):
    # Same file scope and JSON identity as shared/dist_protocol/runner_v3.py.
    patterns=("*.safetensors","*.bin","*.pt","*.gguf","config.json","generation_config.json",
              "tokenizer*","vocab*","merges.txt","special_tokens_map.json","*.model","*.tiktoken")
    paths={p.name:p for pattern in patterns for p in MODEL.glob(pattern) if p.is_file()}
    records={name:dict(sha256=model_files.get(name) or R.sha(p),size=p.stat().st_size) for name,p in paths.items()}
    return hashlib.sha256(json.dumps(records,sort_keys=True).encode("utf-8")).hexdigest()


def validate_plan(folder, require_runtime=True):
    plan = R.Base.Previous.load(folder/"plan.json")
    assert plan["run"]==RUN
    assert all(R.sha(folder/name)==x["sha"] for name,x in plan["materials"].items())
    assert all(R.sha(ROOT/name)==h for name,h in plan["sources"].items())
    if require_runtime:
        assert plan["runtime"]==versions()
    return plan


def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL,local_files_only=True)


def vocabulary(tokenizer):
    return {p:[w for w in M.WORDS[p].split() if len(tokenizer.encode(" "+w,add_special_tokens=False))==1] for p in M.PANELS}


def batches(rows):
    return list(R.Base.groups(rows,BATCH))


def pilot():
    import torch

    start=time.monotonic()
    torch.set_num_threads(4)
    tokenizer = tok()
    candidates = {p:[w for w in M.PILOT[p].split() if len(tokenizer.encode(" "+w,add_special_tokens=False))==1] for p in M.PANELS}
    panels = {p:M.make(p,"pilot",candidates[p]) for p in M.PANELS}
    for rows in panels.values():
        for row in rows:
            R.Base.encode(tokenizer,row)
    source={p:R.sha(ROOT/p) for p in SOURCES if (ROOT/p).is_file()}
    R.write("pilot-materials.json",dict(panels=panels,source=source))
    with R.Base.Previous.GpuLease(),torch.inference_mode():
        R.Base.Previous.stage_log("q4","pilot",0,len(panels),start)
        model=R.Base.load_model("q4")
        try:
            g=R.Task(model)
            result={}
            for i,(panel,rows) in enumerate(panels.items(),1):
                obs=[o for group in batches(rows) for o in g.natural(group)]
                result[panel]=dict(rows=obs,cells=R.gate(rows,obs))
                R.write("pilot-behavior.json",result)
                R.Base.Previous.stage_log("q4","pilot",i,len(panels),start)
        finally:
            model.close()


def freeze():
    import math
    from statistics import NormalDist
    assert not (OUT/"plan.json").exists()
    pilot_doc = R.read("pilot-behavior.json")
    assert all(sum(c["correct"] for c in pilot_doc[p]["cells"])>=44 for p in M.PANELS)
    calibration=R.read("engine-pilot.json")
    assert calibration["exact_all_vocab"] and len(calibration["paths"])==32
    tokenizer = tok()
    vocab = vocabulary(tokenizer)
    tables = {}
    all_pairs=set()
    for panel in M.PANELS:
        data = M.make(panel,"confirm",vocab[panel])
        for row in data:
            R.Base.encode(tokenizer,row)
        families={tuple(sorted((r["old"],r["new"]))) for r in data}
        assert len(families)==M.N and not (families & all_pairs)
        all_pairs |= families
        name=f"materials-{panel}.json"
        R.write(name,data)
        tables[name]=dict(sha=R.sha(OUT/name),views=len(data),families=M.N)
    for dirname, files in (("kv_route_20261115_v2",("long","multi","role")),("kv_route_20261115_v3",("polarity",))):
        for suite in files:
            p=OUT.parent/dirname/f"materials-{suite}.json"
            old=R.Base.Previous.load(p)
            assert not (all_pairs & {tuple(sorted((r["old"],r["new"]))) for r in old})
    pilot_rows=R.read("pilot-materials.json")["panels"]
    assert not (all_pairs & {tuple(sorted((r["old"],r["new"]))) for rows in pilot_rows.values() for r in rows})
    binary=check_phase(.085,M.N,p_d=.10,alpha=.025)
    m=int(M.N*.80)
    z=NormalDist()
    planned_equivalence_power=2*z.cdf(.25*math.sqrt(m)/.85-z.inv_cdf(.975))-1
    assert binary["pass"] and planned_equivalence_power>=.8
    model=collect_identity()
    snapshot=official_id(model)
    old=R.Base.Previous.load(OUT.parent/"kv_route_20261115_v3/audit.json")
    assert snapshot == old["snapshot"]
    prov=dict(**git_identity(),model_path=str(MODEL),model_revision=snapshot,seed=M.SEED)
    R.write("plan.json",dict(
        run=RUN,date=datetime.now(ZoneInfo("America/Chicago")).isoformat(timespec="minutes"),
        provenance=prov,model_files=model,runtime=versions(),batch=BATCH,cap=R.CAP,heads=R.HEADS,
        materials=tables,sources={p:R.sha(ROOT/p) for p in SOURCES},
        pilots={s:R.sha(OUT/s) for s in ("pilot-materials.json","pilot-behavior.json","engine-pilot.json","writes-engine-pilot.npz")},
        templates="Author-written fictional reports, new unordered lexical pairs, all four views within a family",
        behavior="Every preregistered cell strict >= .90. Failed suites behavior only. No rescanning heads.",
        power=dict(binary_plan=binary,equivalence_SD_assumption=.85,equivalence_retained_family_n=m,
                   equivalence_power_centered_zero=planned_equivalence_power,equivalence_logit_bound=.25),
        hypothesis=dict(raw_reversal="Polarity H19 raw State minus Note <= -.25 logits with upper 97.5% bootstrap limit < 0",
                        matching="Per layer and selected head, native BF16 V edit. Match direction and alpha-weighted dose at the answer query. Use live Q/K on each arm.",
                        low="common target alpha = min(sham State alpha, sham Note alpha), scale bound 2.05",
                        high="common target alpha = min(max(sham State alpha, sham Note alpha), 2*min(sham alphas)), scale bound 2.05",
                        parallel_simulator="Predeclared algebraic simulation using each arm's live alpha and the identical donor delta at both slots. No selected-slot V read/edit or slot-dependent V rounding in this arm; BF16 cast at head output.",
                        norm_floor=.008,max_planned_relative_error=.06,max_actual_relative_error=.35,
                        max_actual_paired_relative_error=.25,actual_absolute_error_floor=.025,
                        min_actual_norm_fraction=.50,min_actual_cosine=.85,analytic_absolute_error_floor=.001,
                        common_eligible="Both slots and both directions pass dose checks in all selected heads, in all four views in a family. Report every excluded family and continuous-effect complete population separately.",
                        primary="192 family means. H19 raw reversal and matched low equivalence on same eligible families. High dose and trio separate; examine content effect and prediction.",
                        minimum_matched_families=m,retention_note="Coverage prior to sign/equivalence conclusion. Below threshold restricted descriptive subset only.",
                        family_bootstrap_draws=10000,analysis_seed=29394001,alpha=.025,equivalence=.25,
                        analytic_mechanism="Position can change raw alpha but if every selected head receives exactly the same write at the same query and all other state is equal, its downstream effect is equal by construction. Matched equivalence alone is insufficient.",
                        raw_forward_prediction="Explain raw polarity sign and magnitude with native Value-edit-independent write injection m(native head + live alpha times identical donor delta), no fitting on heldout.",
                        raw_predictive_rmse_max=.30,raw_predictive_bias_max=.15,raw_prediction_rho_min=.70),
        independence=dict(cross_node_score_abs=.35,minimum_row_fraction=.95,
                          minimum_overlapping_admission_fraction=.90,max_family_mean_difference=.25,
                          model="identical file hashes, runtime, batch and code, new device and machine identity",
                          protocol="blind package of frozen materials and sources; operator submits full results before seeing primary measurements")))


def node_record(torch):
    from hashlib import sha256
    raw=platform.node().strip().lower()
    props=torch.cuda.get_device_properties(0)
    uuid=str(getattr(props,"uuid","") or "").strip()
    return dict(host_sha256=sha256(raw.encode()).hexdigest(),gpu=props.name,gpu_uuid=uuid,
                cuda=torch.version.cuda,cuda_capability=torch.cuda.get_device_capability(0),
                memory=props.total_memory,os=platform.system(),platform_release=platform.release())


def engine_pilot():
    import torch
    import kv_alpha_native as Native

    torch.set_num_threads(4)
    start=time.monotonic()
    rows=R.read("pilot-materials.json")["panels"]["polarity"][:BATCH]
    expected=R.read("pilot-behavior.json")["polarity"]["rows"][:BATCH]
    with R.Base.Previous.GpuLease(),torch.inference_mode():
        R.Base.Previous.stage_log("q4","engine-pilot",0,1,start)
        engine=R.Base.load_model("q4")
        try:
            R.write("engine-pilot.json",Native.measure(R.Task(engine),rows,expected,"engine-pilot"))
        finally:
            engine.close()
        R.Base.Previous.stage_log("q4","engine-pilot",1,1,start)


def execute(plan_dir):
    import torch
    import kv_alpha_native as Native

    plan=validate_plan(plan_dir)
    torch.set_num_threads(4)
    result=[]
    plan_sha=R.sha(plan_dir/"plan.json")
    with R.Base.Previous.GpuLease(),torch.inference_mode():
        assert collect_identity()==plan["model_files"]
        R.write("node.json",dict(**node_record(torch),provenance=plan["provenance"],plan_sha=plan_sha))
        start=time.monotonic()
        R.Base.Previous.stage_log("q4","load",0,1,start)
        model=R.Base.load_model("q4")
        try:
            g=R.Task(model)
            R.Base.Previous.stage_log("q4","load",1,1,start)
            for panel in M.PANELS:
                name=f"materials-{panel}.json"
                path=plan_dir/name
                rows=R.Base.Previous.load(path)
                jobs=batches(rows)
                bname=f"behavior-{panel}.json"
                torch.cuda.empty_cache()
                if (OUT/bname).exists():
                    behavior=R.read(bname)
                    assert behavior["plan_sha"]==plan_sha
                else:
                    obs=[]
                    start=time.monotonic()
                    for i,group in enumerate(jobs,1):
                        obs.extend(g.natural(group))
                        R.Base.Previous.stage_log("q4",panel+"-behavior",i,len(jobs),start)
                    cells=R.gate(rows,obs)
                    behavior=dict(rows=obs,cells=cells,passed=all(a["passed"] for a in cells),plan_sha=plan_sha,provenance=plan["provenance"])
                    R.write(bname,behavior)
                stage=dict(panel=panel,behavior_sha=R.sha(OUT/bname),status="behavior-gate-failed",shards=[])
                result.append(stage)
                if not behavior["passed"]:
                    continue
                start=time.monotonic()
                for i,(group,baseline) in enumerate(zip(jobs,batches(behavior["rows"]),strict=True),1):
                    key=f"{panel}-{i:03d}"
                    json_name=f"native-{key}.json"
                    npz_name=f"writes-{key}.npz"
                    if not (OUT/json_name).exists():
                        torch.cuda.empty_cache()
                        shard = Native.measure(g,group,baseline,key,sample=i==1)
                        R.write(json_name,dict(shard,plan_sha=plan_sha,npz_sha=R.sha(OUT/npz_name),provenance=plan["provenance"]))
                    obj=R.read(json_name)
                    assert obj["plan_sha"]==plan_sha and obj["ids"]==[r["id"] for r in group]
                    assert obj["npz_sha"]==R.sha(OUT/npz_name)
                    stage["shards"].append(dict(json=json_name,npz=npz_name,sha=R.sha(OUT/json_name),npz_sha=obj["npz_sha"]))
                    R.Base.Previous.stage_log("q4",panel+"-native",i,len(jobs),start)
                stage["status"]="complete"
        finally:
            model.close()
    R.write("execution.json",dict(plan_sha=plan_sha,panels=result,provenance=plan["provenance"]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("step",choices=("pilot","engine-pilot","freeze","run"))
    parser.add_argument("--out",type=Path)
    parser.add_argument("--frozen",type=Path)
    a=parser.parse_args()
    configure(a.out)
    if a.step=="pilot":
        pilot()
    elif a.step=="engine-pilot":
        engine_pilot()
    elif a.step=="freeze":
        freeze()
    else:
        execute(a.frozen or OUT)


if __name__=="__main__":
    main()
