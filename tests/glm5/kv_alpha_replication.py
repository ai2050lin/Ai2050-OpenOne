"""Prepare an outcome-blind independent-node job and compare returned measurements."""

import argparse
import ast
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from zoneinfo import ZoneInfo

import numpy as np

import kv_alpha_run as C


ROOT=C.ROOT
V1=C.OUT
V2=V1.with_name("kv_alpha_20261115_v2")
R=C.R
SRC="tests/glm5/"
VERSIONS={"v1":V1,"v2":V2}


def modules(path):
    tree=ast.parse((ROOT/path).read_text(encoding="utf-8-sig"),str(path))
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node,ast.ImportFrom) and node.level==0 and node.module:
            yield node.module


def source_closure(initial):
    paths=set(initial)
    remain=list(paths)
    while remain:
        name=remain.pop()
        if not name.endswith(".py"):
            continue
        for module in modules(name):
            stem=module.replace(".","/")
            candidates=[SRC+stem+".py",stem+".py",stem+"/__init__.py"]
            for p in candidates:
                if (ROOT/p).is_file() and p not in paths:
                    paths.add(p)
                    remain.append(p)
    return sorted(paths)


def prepare():
    C.configure()
    plans={v:R.Base.Previous.load(folder/"plan.json") for v,folder in VERSIONS.items()}
    paths={n for p in plans.values() for n in p["sources"]}
    for p in plans.values():
        assert all(R.sha(ROOT/n)==h for n,h in p["sources"].items())
    paths.update(SRC+f for f in ("kv_alpha_confirm.py","kv_alpha_audit.py","kv_alpha_replication.py","kv_route_audit.py"))
    paths=source_closure(paths)
    records={p:(ROOT/p).read_bytes() for p in paths}
    for version,folder in VERSIONS.items():
        for name in ("plan.json",*plans[version]["materials"]):
            records[f"tests/glm5/result/kv_alpha_replay_{version}/{name}"]=(folder/name).read_bytes()
    prereg={v:dict(run=p["run"],pinned_plan_sha=R.sha(VERSIONS[v]/"plan.json"),
                   rows=sum(x["views"] for x in p["materials"].values()),
                   snapshot=p["provenance"]["model_revision"],tolerances=p["independence"])
            for v,p in plans.items()}
    notes=f"""独立 GPU 复算工单 · KV-ALPHA-20261115

使用另一台 Windows 主机上的 NVIDIA CUDA GPU，建议至少 16 GiB 专用显存和 32 GiB 主存。
包内为冻结材料、计划、经过哈希的源文件。所有本地实验读出均未放入此包。
把 ZIP 解压到一个新的项目目录，Windows 使用 D 或其他合规盘，保留 20 GiB 余量。
在 models/hf/qwen3-4b 放入核对一致的原始权重和 tokenizer，保留文件名。准备 CUDA
和 plan.json 中的 Python 包版本，以及 accelerate、psutil、tzdata。模型文件由各次运行自动重哈希。
可直接在解压后的项目根目录使用已有的兼容 Python 解释器运行完再进入下一版本。

    python -B -u tests/glm5/kv_alpha_run.py run --frozen tests/glm5/result/kv_alpha_replay_v1 --out tests/glm5/result/kv_alpha_independent_v1
    python -B -u tests/glm5/kv_alpha_audit.py --frozen tests/glm5/result/kv_alpha_replay_v1 --out tests/glm5/result/kv_alpha_independent_v1
    python -B -u tests/glm5/kv_alpha_confirm.py run --frozen tests/glm5/result/kv_alpha_replay_v2 --out tests/glm5/result/kv_alpha_independent_v2
    python -B -u tests/glm5/kv_alpha_confirm.py audit tests/glm5/result/kv_alpha_independent_v2 tests/glm5/result/kv_alpha_replay_v2

完整回传两个 independent 输出目录，包括 node.json、execution.json、behavior、全部已写分片、
progress.jsonl、audit 和 audit-rows。不要在结果冻结回传前查看首次运行的分数。回传前由操作人核对
实际物理主机和 GPU 身份。自动核验要求两者标识都不同，冻结 plan、模型哈希和 SDK 版本一致。
核对人使用 kv_alpha_replication.py compare-v1 / compare-v2 指定原样回传目录。
工单覆盖计划摘要 {json.dumps(prereg,ensure_ascii=False)}
"""
    records["REPRODUCE.txt"]=notes.encode("utf-8")
    inventory={p:sha256(b).hexdigest() for p,b in records.items()}
    records["REPLICA-MANIFEST.json"]=json.dumps(inventory,sort_keys=True).encode("utf-8")
    path=R.Base.Previous.check_write(V1/"independent-gpu-job.zip")
    with ZipFile(path,"w",ZIP_DEFLATED) as z:
        for p,b in records.items():
            z.writestr(p,b)
    with ZipFile(path) as z:
        assert set(z.namelist())==set(records) and all(sha256(z.read(p)).hexdigest()==h for p,h in inventory.items())
        assert not any("/behavior-" in p or "/native-" in p or "/audit.json" in p for p in z.namelist())
    R.write("replication-job.json",dict(
        schema="kv_alpha_replication_job.v1",status="ready-awaiting-independent-node",delivery=None,receipt=None,
        time=datetime.now(ZoneInfo("America/Chicago")).isoformat(timespec="minutes"),
        scope="Two same-protocol reruns. No second host assigned in accessible environment.",
        manifest=inventory,package=path.name,package_sha=R.sha(path),plans=prereg,
        expected_runtime=plans["v1"]["runtime"],expected_environment="Physical separate Windows/CUDA computer",
        provenance=plans["v1"]["provenance"]))


def read(dir,name):
    return R.Base.Previous.load(dir/name)


def compare(version,other):
    local=VERSIONS[version]
    C.configure(local)
    plan=read(local,"plan.json")
    limits=plan["independence"]
    a,b=read(local,"node.json"),read(other,"node.json")
    for name in ("host_sha256","gpu_uuid"):
        assert a[name] and b[name] and a[name]!=b[name],f"independent {name} required"
    assert a["plan_sha"]==b["plan_sha"]==R.sha(local/"plan.json")
    assert a["provenance"]["model_revision"]==b["provenance"]["model_revision"]==plan["provenance"]["model_revision"]
    x,y=read(local,"execution.json"),read(other,"execution.json")
    assert x["plan_sha"]==y["plan_sha"]==a["plan_sha"]
    stages=[]
    for left,right in zip(x["panels"],y["panels"],strict=True):
        group=left["panel"]
        assert group==right["panel"]
        assert R.sha(other/f"behavior-{group}.json")==right["behavior_sha"]
        assert R.sha(local/f"behavior-{group}.json")==left["behavior_sha"]
        p,q=[read(base,f"behavior-{group}.json")["rows"] for base in (local,other)]
        agree=sum(i["tokens"]==j["tokens"] for i,j in zip(p,q,strict=True))/len(p)
        stage=dict(panel=group,local_status=left["status"],independent_status=right["status"],
                   behavior_token_agreement=agree,metrics={},passed=left["status"]==right["status"] and agree>=limits["minimum_row_fraction"])
        stages.append(stage)
        if left["status"]!=right["status"] or left["status"]!="complete":
            continue
        for s in right["shards"]:
            assert R.sha(other/s["json"])==s["sha"] and R.sha(other/s["npz"])==s["npz_sha"]
        # Intake command requires the independent CPU audit to have rerun on those exact raw artifacts.
        auditors=[read(base,"audit.json") for base in (local,other)]
        assert auditors[0]["analysis_sha"]==auditors[1]["analysis_sha"]
        name=f"audit-rows-{group}.json"
        for base,doc in zip((local,other),auditors,strict=True):
            saved=next(row for row in doc["panels"] if row["panel"]==group)
            assert R.sha(base/name)==saved["row_sha"]
        aa,bb=[read(base,name)["rows"] for base in (local,other)]
        aligned=list(zip(aa,bb,strict=True))
        assert all(i["id"]==j["id"] and i["family"]==j["family"] for i,j in aligned)
        for g in ("h19","trio"):
            for regime in ("raw","low","high","write"):
                get=(lambda r:r["doses"][g][regime]["admit"]) if regime in ("low","high") else (lambda r:r["valid"])
                planned=[(i,j) for i,j in aligned if get(i) or get(j)]
                both=[(i,j) for i,j in planned if get(i) and get(j)]
                overlap=len(both)/len(planned) if planned else 1.
                delta=np.asarray([[j["margin"][g][regime][k]-i["margin"][g][regime][k]
                                   for k in ("state","note","source")] for i,j in both])
                fractions=np.mean(abs(delta)<=limits["cross_node_score_abs"],axis=0) if both else None
                families={}
                for (i,_),difference in zip(both,delta,strict=True):
                    families.setdefault(i["family"],[]).append(difference)
                whole=[np.mean(v,axis=0) for v in families.values() if len(v)==4]
                bias=np.mean(whole,axis=0) if whole else None
                passed=bool(overlap>=limits["minimum_overlapping_admission_fraction"] and
                            (fractions is None or (bias is not None and min(fractions)>=limits["minimum_row_fraction"] and
                                                   max(abs(bias))<=limits["max_family_mean_difference"])))
                stage["metrics"][f"{g}-{regime}"]=dict(
                    n=len(both),complete_families=len(whole),overlap=overlap,passed=passed,
                    family_mean_gap=bias.tolist() if bias is not None else None,
                    within_tolerance=fractions.tolist() if both else None)
                stage["passed"] &= passed
    R.write("independent-comparison.json",dict(version=version,independent_node=b,stages=stages,
            status="within-preregistered-tolerance" if all(s["passed"] for s in stages) else "out-of-tolerance",
            provenance=plan["provenance"]))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("step",choices=("prepare","compare-v1","compare-v2"))
    p.add_argument("path",type=Path,nargs="?")
    args=p.parse_args()
    if args.step=="prepare":
        prepare()
    else:
        assert args.path is not None
        compare(args.step.removeprefix("compare-"),args.path.resolve())
