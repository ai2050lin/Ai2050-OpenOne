"""Freeze, run, and score the 2x2x2 source experiment. Never generate on lockbox."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import gzip
import importlib.metadata
from itertools import batched
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from deploy.storage_policy import check_write, checked_dir, staging_file
from shared.dist_protocol.mechanism_trace import file_sha, digest, GpuLease
import content_origin_material as M
from source_binding_resources import resources
import numpy as np

OUT = ROOT/"tests/glm5/result/content_origin_20261118_v1"
TMP = ROOT/"tests/glm5_temp/content_origin_20261118_v1"
MODEL = ROOT/"models/hf/Qwen3-14B"
PRIOR = ROOT/"tests/glm5/result/paradigm_verdict_20261118_v1/design.json"
BUDGET = 48*2**20
THRESHOLDS = dict(neutral_correct=.75, neutral_python=.75, delta=.15, familywise_alpha=.01,
                  calibration_power=.80, causal_family_coverage=.98, actual_dose_tolerance=.03)


def stamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read(name):
    return json.loads((OUT/name).read_bytes())


def save(name, value):
    raw=json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False,separators=(",",":")).encode()
    target=check_write(OUT/name,2*len(raw)+16384)
    with staging_file(target) as tmp:
        tmp.write_bytes(raw)
    return file_sha(target)


def setup():
    for p in (OUT,TMP):
        checked_dir(p,BUDGET)
    for env in ("TEMP","TMP","TMPDIR","HF_HOME","HF_HUB_CACHE","TORCH_HOME","TORCHINDUCTOR_CACHE_DIR","TRITON_CACHE_DIR","CUDA_CACHE_PATH","MPLCONFIGDIR","XDG_CACHE_HOME"):
        os.environ[env]=str(TMP)
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")


def software():
    return {s:importlib.metadata.version(s) for s in ("torch","transformers","tokenizers","safetensors","numpy","scipy")}


def sources():
    paths=[Path(__file__),Path(M.__file__),Path(__file__).with_name("content_origin_decode.py"),
           Path(__file__).with_name("kv_bf_q14.py"),Path(__file__).with_name("source_binding_resources.py"),
           ROOT/"shared/dist_protocol/mechanism_trace.py",ROOT/"deploy/storage_policy.py"]
    return {p.relative_to(ROOT).as_posix():file_sha(p) for p in paths}


def material_path(split):
    return OUT/("material-"+split+".json.gz")


def freeze():
    from transformers import AutoTokenizer
    from tools.power_analysis import check_phase
    import content_origin_decode as decoder
    assert not (OUT/"design.json").exists()
    prior=json.loads(PRIOR.read_bytes())["model"]
    for name,info in prior["files"].items():
        assert (MODEL/name).stat().st_size==info["size"] and file_sha(MODEL/name)==info["sha256"],name
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    mats={}
    for split,n in M.SIZES.items():
        rows=M.build(tok,split)
        assert len({r["family"] for r in rows})==n and len({r["id"] for r in rows})==len(rows)
        assert max(len(r["ids"]) for r in rows)+1<=decoder.WIDTH
        mats[split]=dict(families=n,views=len(rows),digest=digest(rows),
                         max_tokens=max(len(r["ids"]) for r in rows))
        if split!="lockbox":
            raw=json.dumps(rows,ensure_ascii=False,separators=(",",":")).encode()
            with staging_file(check_write(material_path(split),len(raw))) as tmp:
                tmp.write_bytes(gzip.compress(raw,mtime=0))
            mats[split]["file_sha256"]=file_sha(material_path(split))
    power=check_phase(.15,256,.35,.005)
    assert power["pass"]
    d=dict(at=stamp(),run=OUT.name,source=sources(),software=software(),model=prior,
           model_check="Every local weight shard and tokenizer file hashed before freeze",
           model_fstats={n:(MODEL/n).stat().st_mtime_ns for n in prior["files"]},
           prior_sha=file_sha(PRIOR),materials=mats,thresholds=THRESHOLDS,power=power,
           generation=dict(mode="native BF16, greedy full vocabulary, no answer prefix, no sampling",cap=decoder.CAP,
                           eos="model generation_config EOS plus tokenizer EOS; retain all capped outputs"),
           materials_note="Project-authored. All three main toggles verified tokenwise for each family and frame. Archive position and bracket presence balanced by family. English nouns with frozen Chinese aliases; role-specific pools are disjoint between study splits. Both holdout frames on every confirm family.",
           primary="For requested revised source only, family delta = mean_payload(archive_both(code)-archive_both(prose)). Both untrained frames need >=.15 with lower simultaneous 99% paired family bootstrap above zero (Bonferroni percentile .0025 per tail for each of two). All outputs in denominator, including parse errors and nontermination. Pilot neutral requested content and Python per frame/style/source >=.75; report strict joint separately.",
           secondary="Retain joint object/number table per style,source,frame,payload; report old, missing, multiple and mixed answers. Report paired payload-following and requested-source switching; descriptive order, brackets and actual reply length. No gate on exploratory subsets.",
           power_check="After discovery estimate paired-family SD by frame. At fixed 256, require normal design power >=.80 at +.15 in both frames, alpha=.005 each. Failure stops before confirm.",
           native="Conditional on passing both primary behavior contrasts. B26/H36 at first message-content decision after natural common prefix, H36-only archive/current mass redistribution with V fixed, separately target-source V/O suppression with attention fixed. Native sham and restoration, equal realized norm tolerance .03, >=.98 full-family coverage, matched random and natural source controls, lower simultaneous 99% CI >0 and >=.10 logit advantage; behavioral claim additionally needs >=10pp source-choice change. Freeze technical arms before native data.",
           opened="Pilot used only for neutral calibration. No candidate selection or threshold fitting on confirm. Lockbox material hash only.")
    print("FROZEN",save("design.json",d),mats,flush=True)


def material(d,split):
    path=material_path(split)
    assert file_sha(path)==d["materials"][split]["file_sha256"]
    rows=json.loads(gzip.decompress(path.read_bytes()))
    assert digest(rows)==d["materials"][split]["digest"]
    return rows


def stat_panel(joined, frame):
    table=defaultdict(dict)
    for r in joined:
        if r["frame"]==frame and not r["requested"]:
            table[r["family"]][r["style"],r["payload"]]=int(r["score"]["archive_both"])
    return np.array([sum(v[1,i]-v[0,i] for i in (0,1))/2 for _,v in sorted(table.items())])


def summary(rows,split):
    from scipy.stats import norm
    if split=="pilot":
        cells=defaultdict(list)
        for r in rows:
            cells[r["frame"],r["style"],r["requested"]].append(r)
        rates={str(k):{metric:float(np.mean([x["score"][metric] for x in group]))
                       for metric in ("correct","python","joint")} for k,group in cells.items()}
        return dict(passed=all(x["correct"]>=.75 and x["python"]>=.75 for x in rates.values()),cells=rates)
    rng=np.random.default_rng(29812027)
    frames=M.FRAMES_BY_SPLIT[split]
    family_indices=rng.integers(0,M.SIZES[split],size=(30000,M.SIZES[split]))
    effects={}
    for f in frames:
        x=stat_panel(rows,f)
        sample=x[family_indices].mean(1)
        ci=np.quantile(sample,[.0025,.9975]).tolist()
        sd=x.std(ddof=1)
        power=1. if sd==0 else float(norm.cdf(.15*16/sd-norm.ppf(.9975)))
        effects[f]=dict(delta=x.mean(),ci=ci,power=power,sd=sd,n=len(x),
                        passed=bool(x.mean()>=.15 and ci[0]>0))
    matrix=defaultdict(Counter)
    for r in rows:
        k="/".join(str(r[name]) for name in ("frame","style","payload","requested"))
        matrix[k][r["score"]["obj"]+"/"+r["score"]["num"]]+=1
    return dict(passed=all(e["passed"] for e in effects.values()),power_pass=all(e["power"]>=.8 for e in effects.values()),
                effects=effects,matrix=matrix)


def run():
    import torch
    import content_origin_decode as decoder
    d=read("design.json")
    assert d["source"]==sources() and d["software"]==software()
    assert all((MODEL/n).stat().st_mtime_ns==mt for n,mt in d["model_fstats"].items())
    binding=file_sha(OUT/"design.json")
    torch.set_num_threads(4)
    def log(stage,done,total,t0,**more):
        elapsed=time.monotonic()-t0
        event=dict(at=stamp(),stage=stage,done=done,total=total,seconds=round(elapsed,2),
                   eta_seconds=round(elapsed/done*(total-done),1) if done else None,
                   **resources(),**more)
        if torch.cuda.is_initialized():
            event["cuda_peak_bytes"]=torch.cuda.max_memory_allocated()
        with check_write(OUT/"progress.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(event)+"\n")
        print(stage,done,"/",total,more,round(elapsed),flush=True)
    with GpuLease(),torch.inference_mode():
        started=time.monotonic()
        log("load",0,1,started)
        decoder.native.RESIDENT=int(7.15*2**30)
        model=decoder.native.Engine()
        log("load",1,1,started,resident=model.bytes)
        try:
            for split in ("pilot","discovery","confirm"):
                rows=material(d,split)
                joined=[]
                started=time.monotonic()
                for part,group in enumerate(batched(rows,decoder.SLAB)):
                    check_write(OUT,BUDGET,operation="source study")
                    key=f"{split}-{part:03}.json"
                    if not (OUT/key).exists():
                        batch_start=time.monotonic()
                        last=[0.]
                        def progress(step,layer):
                            now=time.monotonic()
                            if now-last[0]>45 or step==0 and layer==10:
                                log(split+"-decode",step+layer/40,decoder.CAP,batch_start,shard=part,
                                    completed=len(joined),total_views=len(rows))
                                last[0]=now
                        decoded=decoder.greedy(model,group,progress,shadow=split=="pilot" and part==0)
                        save(key,dict(at=stamp(),binding=binding,rows=decoded))
                    record=read(key)
                    assert record["binding"]==binding
                    for r,answer in zip(group,record["rows"],strict=True):
                        assert r["id"]==answer["id"]
                        meta={k:v for k,v in r.items() if k not in ("ids","prompt","spans")}
                        joined.append(dict(meta,**{k:v for k,v in answer.items() if k!="id"},
                                           score=M.score(answer["text"],r,answer["stopped"])))
                    log(split,len(joined),len(rows),started,shard=part)
                stats=summary(joined,split)
                save(split+"-summary.json",dict(binding=binding,at=stamp(),**stats))
                print(split,"gate",stats.get("effects",stats.get("cells")),flush=True)
                if split=="pilot" and not stats["passed"] or split=="discovery" and not stats["power_pass"]:
                    break
        finally:
            model.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("freeze","run"))
    action=parser.parse_args().action
    setup()
    (freeze if action=="freeze" else run)()
