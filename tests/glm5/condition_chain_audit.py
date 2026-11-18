"""Recompute frozen material and output identity, score failures, and account for exact prefix reuse."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import re
import sys
import time

import numpy as np
import condition_chain_run as R
import condition_chain_materials as M
import condition_chain_chat as C
import condition_chain_candidates as S
import condition_chain_isolated as I
import condition_chain_diverse_candidates as D
from condition_chain_analysis import json_cells, native, paired_chain, stats

BASE = R.ROOT/"tests/glm5/result"
FOLDERS = (
    "condition_chain_20261115_v1",
    "condition_chain_chat_20261115_v2",
    "condition_chain_candidates_20261115_v3",
    "condition_chain_isolated_20261115_v4",
    "condition_chain_diverse_candidates_20261115_v5",
)
RAW_PROMPTS,RAW_ENCODE=R.prompts,R.encode
CHAT_RENDER=C.render


def setup_version(version):
    R.OUT=BASE/FOLDERS[version-1]
    R.TMP=R.ROOT/"tests/glm5_temp"/FOLDERS[version-1]
    R.RUN=R.load(R.OUT/"plan.json")["run_id"]
    R.setup()
    C.render={1:CHAT_RENDER,2:CHAT_RENDER,3:S.render,4:I.render,5:S.render}[version]
    R.prompts=RAW_PROMPTS if version==1 else C.render
    R.encode=RAW_ENCODE if version==1 else C.encode
    return R.protocol()


def center_identity(key, identity):
    from shared.dist_protocol.runner_v3 import SNAPSHOT_PATTERNS
    folder=R.ROOT/"models/hf"/M.MODELS[key]
    files={p.name:p for pattern in SNAPSHOT_PATTERNS for p in folder.glob(pattern) if p.is_file()}
    details={name:dict(sha256=identity["files"].get(name) or R.file_sha(path),size=path.stat().st_size)
             for name,path in files.items()}
    return hashlib.sha256(json.dumps(details,sort_keys=True).encode("utf-8")).hexdigest()


def score(material, outputs, tokenizer, version):
    assert len(outputs)==len(material)
    records=[]
    groups=defaultdict(list)
    for r,o in zip(material,outputs,strict=True):
        token,position=R.encode(tokenizer,r)
        assert (r["id"],token,position)==(o["id"],o["tokens"],o["sites"])
        assert min(position)>=0 and position==sorted(set(position))
        opts=np.array(o["option_logits"],dtype=np.float64)
        idx=r["answer"]
        if r["panel"]=="chain":
            source,replace,distract,route=r["factors"]
            state=[0,2][source]
            end=state+replace
            assert r["truth"]==[state,end,end]
            assert r["option_order"][idx]==end
        elif version==4:
            assert r["options"][idx]==r["correct_option"]
        elif r["task"]=="boundary" and "evidence" in r:
            a,b=(r["evidence"][q] for q in ("a","b"))
            assert (a>b if version==1 else b>a)==bool(r["factors"][0])
        assert np.isfinite(opts).all() and np.isfinite(o["total_logsumexp"])
        assert float(opts[idx])<=o["total_logsumexp"]+1e-6
        rank=int(opts.argmax())
        strictly=bool(opts[idx]>max(sc for j,sc in enumerate(opts) if j!=idx))
        assert strictly==o["correct"]
        g=o.get("generated")
        initial=None
        if g is not None:
            actual=tokenizer.decode(g,skip_special_tokens=True).strip()
            assert o["natural_answer"]==actual
            initial=re.match(r"^\s*[*_]*(?:option\s+)?([ABCD])(?=$|\W)",actual)
            initial=bool(initial and initial[1]=="ABCD"[idx])
            if version==4:
                assert initial==o["natural_initial_correct"]
            assert o["natural_correct"] == (actual=="ABCD"[idx])
        a=np.logaddexp.reduce(opts)
        records.append(dict(id=r["id"],family=r["family"],task=r["task"],stage=r.get("stage",""),
                            choices=len(opts),correct=strictly,first_letter=initial,exact=o["natural_correct"],
                            margin=float(opts[idx]-max(sc for j,sc in enumerate(opts) if j!=idx)),
                            restricted_nll=float(a-opts[idx]), full_nll=float(o["total_logsumexp"]-opts[idx]),
                            restricted_mass=float(min(1.,np.exp(a-o["total_logsumexp"]))),
                            initial_example=o["natural_answer"],tie=bool(rank!=idx and opts[rank]==opts[idx])))
        groups[r["family"]].append((tuple(r["factors"]),tuple(token[:position[0]+1])))
    assert len(groups)>0
    return records,groups


def section(rows,outputs,tokenizer,version):
    records,groups=score(rows,outputs,tokenizer,version)
    subsets=defaultdict(list)
    for row,o in zip(rows,records,strict=True):
        subsets[(row["task"],row.get("stage",""))].append((row,o))
    summary=[]
    for (task,stage),prs in subsets.items():
        # Whole prompt duplicates can change with option layout. Exact prefix clusters
        # preserve matched family-pair similarity for this check.
        rt=[o for _,o in prs]
        unique_fams={o["family"] for o in rt}
        exact=defaultdict(list)
        subset_arms=defaultdict(list)
        for row,item in prs:
            source=next(pair[1] for pair in groups[row["family"]] if pair[0]==tuple(row["factors"]))
            subset_arms[row["family"]].append((tuple(row["factors"]),source))
            exact[tuple(row["factors"])].append(item["correct"])
        accuracy_by_arm={",".join(map(str,arm)):sum(flags)/len(flags) for arm,flags in exact.items()}
        summary.append(dict(task=task,stage=stage,n=len(rt),families=len(unique_fams),
                            exact_distinct_first_prefix_families=len({tuple(sorted(v)) for v in subset_arms.values()}),
                            min_cell=min(accuracy_by_arm.values()),
                            correct=sum(x["correct"] for x in rt),
                            first_letter=sum(x["first_letter"] is True for x in rt),
                            natural_total=sum(x["first_letter"] is not None for x in rt),
                            bare_letter=sum(x["exact"] is True for x in rt),
                            mean_margin=float(np.mean([x["margin"] for x in rt])),
                            mean_restricted_nll=float(np.mean([x["restricted_nll"] for x in rt])),
                            mean_restricted_mass=float(np.mean([x["restricted_mass"] for x in rt])),
                            per_arm=accuracy_by_arm,
                            example_failures=[x for x in rt if not x["correct"]][:3]))
    return summary


def native_diagnostics(version):
    if version not in (1,3,5):
        return []
    results=[]
    for split in ("discover","words","frames"):
        raw=native(split,"candidate")
        for task in M.CANDIDATES:
            rows=[r for r in R.load(R.OUT/f"materials-{split}-candidate.json") if r["task"]==task and r["id"] in raw]
            if not rows:
                continue
            groups=defaultdict(dict)
            for row in rows:
                groups[row["family"]][tuple(row["factors"])]=row
            joint=[]
            for r in rows:
                data=raw[r["id"]]
                g,v,a=(data[k][0].item() for k in ("G","V","A"))
                assert np.isfinite([g,v,a]).all()
                ideal=g/(1+np.exp(-g))*v
                joint.append(abs(ideal-a))
            for cue in (0,1):
                values={}
                clustered=defaultdict(list)
                for field in ("G","V","A"):
                    values[field]=[]
                    for record in groups.values():
                        hi,lo=record[(1,cue)],record[(0,cue)]
                        diff=float(raw[hi["id"]][field][0]-raw[lo["id"]][field][0])
                        values[field].append(diff)
                        if field=="G" and "domain_cluster" in hi:
                            clustered[hi["domain_cluster"]].append(diff)
                results.append(dict(task=task,split=split,cue=cue,n=len(groups),
                                    mean_delta={field:float(np.mean(v)) for field,v in values.items()},
                                    domain_G={k:float(np.mean(v)) for k,v in clustered.items()},
                                    G_cluster_sensitivity=stats([np.mean(x) for x in clustered.values()]) if len(clustered)>1 else None,
                                    max_abs_real_product_residual=max(joint)))
    return results


def audit(version):
    from transformers import AutoTokenizer
    import re
    p=setup_version(version)
    data=[]
    idents={}
    start=time.monotonic()
    files=sorted(R.OUT.glob("*.json"))
    for key,dirname in M.MODELS.items():
        where=R.OUT/f"{key}-model.json"
        if not where.exists():
            continue
        ident=R.load(where)
        assert ident["plan_sha"]==R.file_sha(R.OUT/"plan.json")
        assert R.digest(ident["files"])==ident["snapshot"]
        idents[key]=dict(local_file_manifest_sha256=ident["snapshot"],canonical_snapshot=center_identity(key,ident),
                         actual_placement=ident["placement"])
        tok=AutoTokenizer.from_pretrained(R.ROOT/"models/hf"/dirname,local_files_only=True)
        assert ident["letters"]==[tok.encode(" "+letter,add_special_tokens=False)[0] for letter in "ABCD"]
        for f in files:
            match=re.fullmatch(f"{key}-(pilot|discover|words|frames)(?:-(chain|candidate))?\\.json",f.name)
            if not match:
                continue
            split,panel=match.groups()
            chosen_panel=f"-{panel}" if panel else ""
            inputs=R.load(R.OUT/f"materials-{split}{chosen_panel}.json")
            if key!="q4" and split in ("words","frames"):
                inputs=[r for r in inputs if r["index"]<32]
            result=R.load(f)
            outputs=result["results"]
            summaries=section(inputs,outputs,tok,version)
            if version!=4:
                assert json_cells(R.cell_pass(inputs,outputs,p["thresholds"]["behavior"]))==result["gate"]
            else:
                for g in result["gate"]:
                    task,stage,flip=g["cell"]
                    group=[o for row,o in zip(inputs,outputs,strict=True)
                           if (row["task"],row["stage"],row["factors"][0])==(task,stage,flip)]
                    assert len(group)==g["n"] and sum(o["correct"] for o in group)==g["correct"]
                    assert g["passed"]==(g["correct"]/g["n"]>=p["thresholds"]["cell_accuracy"])
            data.append(dict(model=key,split=split,panel=panel or "isolated",input_rows=len(inputs),summary=summaries,
                             output_sha256=R.file_sha(f),gates=result["gate"],
                             paired=paired_chain(inputs,outputs) if panel=="chain" else []))
            print(f"v{version} {key} {split} {panel} audited={len(inputs)} elapsed_s={time.monotonic()-start:.1f}",flush=True)
    native_list=native_diagnostics(version)
    diagnostics=[R.load(file) for file in [R.OUT/"analysis.json",R.OUT/"interventions.json"] if file.exists()]
    R.write("audit.json",dict(version=version,run_id=R.RUN,plan_sha256=R.file_sha(R.OUT/"plan.json"),models=idents,
                              verified_rows=sum(r["input_rows"] for r in data),sections=data,native_descriptive=native_list,
                              analysis_summaries=[d.get("candidates",d.get("status")) for d in diagnostics],
                              limitation="CPU audit of identity, tokenizer, metrics and gates; no independent model rerun"))


if __name__=="__main__":
    for v in map(int,sys.argv[1:]):
        audit(v)
