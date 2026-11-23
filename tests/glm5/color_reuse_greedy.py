"""Format audit: continue actual greedy answers from the registered first tokens."""
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import zipfile
ROOT=Path(os.environ["PYTHONPATH"].split(os.pathsep)[0]) if Path(__file__).name=="test.py" else Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"tests/glm5"),str(ROOT)]
from color_reuse_material import HERE, MODEL, OUT, digest, read, save, sha, sources
from color_reuse_test import Native
from deploy.storage_policy import checked_dir, check_write
from deploy.test_archive_client import runtime_identity
from shared.dist_protocol.test_archive import validate_plan

ENTRY=HERE/"color_reuse_greedy.py"
DEST=OUT/"greedy"
PREVIOUS=OUT/"completion/study-package"
ARMS=("base","cut","ctrl","restore")
LIMIT=12
CRITERIA="""Post hoc measurement audit on the exposed fixed 324 comparison prompts, all four arms.
Start from each saved real greedy top-1 token. Parse the first standalone same or different in the
greedy continuation, ignoring case, up to 12 total tokens. No forced token, no label insertion or
resampling. Retain actual IDs/text and undecided, report all nine cells, same/different accuracy,
per-arm predicted-answer flips including new errors and repairs, and exact restored decoded answer.
To count a natural behavioral capability require >=80% correct in each of nine cells, treating
unresolved as incorrect. For the causal question report all actual red-first-item transitions by
reference color, without modifying the original calibration, thresholds or candidate.
This is an explanatory audit of an exposed test, not an independent confirmation."""

def version():
    return {**sources(),ENTRY.relative_to(ROOT).as_posix():sha(ENTRY)}


def prepare_audit():
    checked_dir(DEST,8*2**20,operation="冻结实际生成格式核查")
    if (DEST/"study-plan.json").exists():raise FileExistsError("existing audit")
    material=read(PREVIOUS/"questions.json")["rows"]
    cases=[json.loads(s) for s in (PREVIOUS/"case_results.jsonl").read_text(encoding="utf8").splitlines()]
    rows=[]
    for c in cases:
        if c["stage"]=="confirm" and c["task"]=="compare":
            q=material[c["id"]]
            rows.append(dict(id=q["id"],item=q["item"],ref=q["ref"],color=q["color"],refcolor=q["refcolor"],
                             gold=q["gold"],arm=c["arm"],tokens=q["tokens"],pos=q["pos"],prompt=q["prompt"],
                             first_token=c["top"],first_text=c["top_text"]))
    if len(rows)!=1296 or len({(r["id"],r["arm"]) for r in rows})!=1296:raise ValueError("coverage")
    parent=read(PREVIOUS/"conclusion.json")["run_id"]
    qfile=DEST/"study-questions.json"
    save(qfile,dict(rows=rows,candidate=read(PREVIOUS/"candidate.json")["selected"],prior=parent,
                    first_token_source_sha256=sha(PREVIOUS/"case_results.jsonl")))
    p=read(PREVIOUS/"protocol.json")
    p.update(title="Qwen3-14B 颜色比较：首位格式符后的真实答案与阻断改答",
        objective="读到模型真实生成的同异答案，核对首位小写候选评分受格式影响的程度。",
        principle="复用每臂真实首 token，只对尚未出现答案的记录按原模型原干预贪心续写。",
        hypothesis="部分首位加粗符后会生成可判定的同异词。不同参照色上的实际改答分开检验。",
        alternatives="答案格式差异、固有异答偏好、真实颜色区分和原生单元作用微弱。",criteria=CRITERIA,entrypoint_sha256=sha(ENTRY))
    p["dataset"].update(id="color-reuse/greedy-format-audit",revision="v1",case_count=len(rows),
        content_sha256=sha(qfile),questions_file_sha256=sha(qfile),split_manifest_sha256=digest([r["id"] for r in rows]),
        exposure="Same exposed prompts and frozen units. Initial greedy tokens available; later tokens unmeasured before this audit.")
    p["acquisition"].update(entrypoint_sha256=sha(ENTRY),questions_file_sha256=sha(qfile),code_sha256=digest(version()),
        inputs_sha256=digest([(r["tokens"],r["first_token"],r["arm"]) for r in rows]),environment_sha256=digest(runtime_identity()),
        generation=dict(script_args=[],sample=False,max_total_tokens=LIMIT),capture=dict(layers=[28,39],readouts=["frozen first position","each generated token"],
            boundary="Greedy full-vocabulary argmax, same native cut/control/restore, causal position fixed"))
    p["analysis"].update(method_version="color_reuse.greedy.v1",code_sha256=digest(version()),scoring_sha256=digest(CRITERIA),
                         selection=dict(stage="study",source_run=parent,frozen_candidate_sha256=sha(PREVIOUS/"candidate.json")))
    validate_plan(p);save(DEST/"study-plan.json",p)
    print("Frozen",len(rows),"arms",sha(DEST/"study-plan.json"),flush=True)


def parse(text):
    match=re.search(r"(?<![A-Za-z])(same|different)(?![A-Za-z])",text,re.I)
    return match.group(1).lower() if match else None


def execute():
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    from source_binding_resources import resources
    work=Path(os.environ["AI2050_TEST_WORK_DIR"]);qp=Path(os.environ["AI2050_TEST_QUESTIONS"])
    plan,pre,review=[read(qp.parent/n) for n in ("protocol.json","preflight.json","review.json")]
    if (not pre["preflight_id"] or pre["plan_sha256"]!=digest(plan) or review["archive_state"]!="pending"
            or Path(review["scratch"]).resolve()!=work.resolve() or sha(qp)!=plan["dataset"]["questions_file_sha256"]
            or sha(Path(__file__))!=plan["entrypoint_sha256"] or digest(version())!=plan["acquisition"]["code_sha256"]):
        raise ValueError("audit identity mismatch")
    print(json.dumps(dict(step="verify snapshot",run=review["server_run_id"])),flush=True)
    if snapshot(MODEL)!=plan["acquisition"]["model_snapshot"]:raise ValueError("snapshot changed")
    pack=read(qp);rows=pack["rows"];cand=pack["candidate"]
    if digest([(r["tokens"],r["first_token"],r["arm"]) for r in rows])!=plan["acquisition"]["inputs_sha256"]:
        raise ValueError("first token identity changed")
    table={}; start=time.monotonic()
    with GpuLease(),torch.inference_mode(),(work/"case_results.jsonl").open("x",encoding="utf8",buffering=1) as f:
        model=Native("qwen14");model.captures={}
        def record(r,text,label):
            stored={k:r[k] for k in ("id","arm","item","ref","color","refcolor","gold","first_text","generated")}
            stored.update(text=text,answer=label,correct=label==r["gold"])
            table[r["id"],r["arm"]]=stored
            f.write(json.dumps(stored,ensure_ascii=False)+"\n")
        pending=[]
        for r in rows:
            r["generated"]=[r["first_token"]]
            if model.tok.decode(r["generated"])!=r["first_text"]:raise ValueError("archived token text differs")
            label=parse(r["first_text"])
            if label:record(r,r["first_text"],label)
            else:pending.append(r)
        for step in range(2,LIMIT+1):
            if not pending:break
            next_pending=[]; step_start=time.monotonic()
            for lo in range(0,len(pending),320):
                items=pending[lo:lo+320]
                model.batch=[dict(r,tokens=r["tokens"]+r["generated"],**{k:cand[k] for k in ("units","ctrl","budget","block")}) for r in items]
                check_write(work,4*2**20,operation="自然输出核查")
                def log(b,layer,x,y,part,length,offset=0,**kwargs):
                    if offset==0 and b in (-1,19,39):
                        done=lo+len(items)*(b+1)/40
                        print(json.dumps(dict(step=step,done=done,total=len(pending),scored=len(table),
                            elapsed=time.monotonic()-start,step_eta=(time.monotonic()-step_start)*(len(pending)-done)/done if done else None,**resources())),flush=True)
                _,scores=model.forward(model.batch,log)
                top=scores.argmax(-1).tolist()
                for r,t in zip(items,top):
                    r["generated"].append(t)
                    txt=model.tok.decode(r["generated"]);label=parse(txt)
                    if label or t==model.tok.eos_token_id or step==LIMIT:record(r,txt,label)
                    else:next_pending.append(r)
                del scores
            pending=next_pending
    if len(table)!=len(rows):raise ValueError("incomplete audit")
    outcomes={}
    for arm in ARMS:
        rs=[r for (_,a),r in table.items() if a==arm]
        cells={}
        for first in ("red","yellow","blue"):
            for ref in ("red","yellow","blue"):
                group=[r for r in rs if (r["color"],r["refcolor"])==(first,ref)]
                cells[first+"/"+ref]=dict(correct=sum(r["correct"] for r in group),n=len(group),
                        new_error=sum(table[r["id"],"base"]["correct"] and not r["correct"] for r in group),
                        repaired=sum(not table[r["id"],"base"]["correct"] and r["correct"] for r in group))
        outcomes[arm]=dict(cells=cells,correct=sum(r["correct"] for r in rs),unresolved=sum(r["answer"] is None for r in rs),
            changed=sum(r["answer"]!=table[r["id"],"base"]["answer"] for r in rs))
    result=dict(stage="greedy_color",run_id=review["server_run_id"],source_run=pack["prior"],outcomes=outcomes,
        unique_prompts=len(rows)//4,scored=len(rows),max_tokens=max(len(r["generated"]) for r in table.values()),
        baseline_gate=all(v["correct"]>=.8*v["n"] for v in outcomes["base"]["cells"].values()),
        exact_restored_text=sum(table[i,"restore"]["generated"]==r["generated"] for (i,a),r in table.items() if a=="base"),
        note="Format audit on exposed material, distinct from the preregistered first-step test.")
    save(work/"conclusion.json",result)
    with zipfile.ZipFile(work/"research-scripts.zip","x",compression=zipfile.ZIP_DEFLATED) as z:
        for path in version():z.write(ROOT/path,arcname=path)
    save(work/"artifacts.json",dict(schema="research_test_files.v1",files=[dict(path="research-scripts.zip",role="script",retention="core")]))
    print(json.dumps(dict(step="finished",cases=len(table),max_tokens=result["max_tokens"])),flush=True)


if __name__=="__main__":
    if sys.argv[1:]==["prepare"]:prepare_audit()
    elif sys.argv[1:]==["launch"]:
        from clt_role_reread_desktop import launch
        raise SystemExit(launch("study",ENTRY,DEST))
    elif not sys.argv[1:]:execute()
    else:raise SystemExit("use prepare or launch")
