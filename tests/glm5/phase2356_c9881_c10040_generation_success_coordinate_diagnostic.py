#!/usr/bin/env python3
"""Automatically continue with prospective full-coordinate prediction of free-generation success."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";RESULT=TESTS/"result"
P2352=RESULT/"phase2352_c9241_c9400_natural_multifuture_transient_field"
OUT=RESULT/"phase2356_c9881_c10040_generation_success_coordinate_diagnostic"
MEMO=ROOT/"research/glm5/docs/AGI_GLM5_MEMO.md";VIS=ROOT/"frontend/public/vis_data/research_kernel"
GEN_BINARY=VIS/"c9242_qwen4b_natural_generation_token_trajectory.float16.npy"
PHASE=2356;CAMPAIGN="C9881-C10040";SHAPE=(384,12,38,2560);EPS=1e-8

sys.path.insert(0,str(TESTS))
import phase2315_c5041_c5100_active_response_contract as io  # noqa:E402
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa:E402
import phase2353_c9401_c9560_conditional_equivalence_route_competition as route  # noqa:E402

if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8",errors="replace")

def save(path:Path,value:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def close_memmap(value:Any)->None:
    mmap=getattr(value,"_mmap",None)
    if mmap is not None:mmap.close()

def enrich_rows()->list[dict]:
    rows=io.read_rows(P2352/"raw/fresh_lockbox_generation.jsonl")
    for row in rows:
        first=row["generated"].splitlines()[0].strip() if row["generated"] else ""
        row["success"]=first==row["target"];row["split"]="train" if row["unit"] in (12,13) else ("selection" if row["unit"]==14 else "lockbox")
    return rows

def centroid_fit(field:np.ndarray,mask:np.ndarray,labels:np.ndarray)->np.ndarray:
    return np.stack([field[mask & (labels==value)].mean(axis=0,dtype=np.float64) for value in (0,1)])

def evaluate(prototypes:np.ndarray,field:np.ndarray,labels:np.ndarray,mask:np.ndarray)->dict:
    actual=field[mask].astype(np.float64);truth=labels[mask];dist=np.maximum(np.sum(actual*actual,axis=1,keepdims=True)+np.sum(prototypes*prototypes,axis=1)[None,:]-2*actual@prototypes.T,0)
    pred=np.argmin(dist,axis=1);recalls=[]
    for value in (0,1):recalls.append(float(np.mean(pred[truth==value]==value)))
    return {"rows":int(mask.sum()),"prevalence":float(np.mean(truth)),"accuracy":float(np.mean(pred==truth)),"balanced_accuracy":float(np.mean(recalls)),"recall_failure":recalls[0],"recall_success":recalls[1]}

def field_routes(signed:np.ndarray,rows:list[dict],train:np.ndarray)->dict[str,np.ndarray]:
    absolute=np.abs(signed)
    return {"signed_hidden":signed,"absolute_hidden":absolute,
            "condition_residual_signed":route.fit_residual(signed,rows,train,("language","query","family")),
            "condition_residual_absolute":route.fit_residual(absolute,rows,train,("language","query","family"))}

def analyze(rows:list[dict])->tuple[dict,dict[str,np.ndarray]]:
    source=np.load(GEN_BINARY,mmap_mode="r").reshape(SHAPE);labels=np.asarray([int(r["success"]) for r in rows])
    train=np.asarray([r["split"]=="train" for r in rows]);selection=np.asarray([r["split"]=="selection" for r in rows]);lock=np.asarray([r["split"]=="lockbox" for r in rows])
    trajectory=[];candidates=[]
    for qpoint in range(SHAPE[2]):
        signed=source[:,0,qpoint].astype(np.float32)
        for name,field in field_routes(signed,rows,train).items():
            proto=centroid_fit(field,train,labels);score=evaluate(proto,field,labels,selection);sorted_score=evaluate(centroid_fit(np.sort(field,axis=1),train,labels),np.sort(field,axis=1),labels,selection)
            trajectory.append({"qpoint":qpoint,"route":name,"selection":score,"sorted_selection":sorted_score})
            candidates.append((score["balanced_accuracy"],score["accuracy"],-qpoint,name,qpoint))
    _,_,_,name,qpoint=max(candidates);signed=source[:,0,qpoint].astype(np.float32);fields=field_routes(signed,rows,train);selected=fields[name]
    prototype=centroid_fit(selected,train,labels);lock_score=evaluate(prototype,selected,labels,lock)
    sorted_field=np.sort(selected,axis=1);sorted_lock=evaluate(centroid_fit(sorted_field,train,labels),sorted_field,labels,lock)
    post=[]
    for step in range(SHAPE[1]):
        signed_step=source[:,step,qpoint].astype(np.float32);step_field=field_routes(signed_step,rows,train)[name]
        post.append({"step":step,"selection":evaluate(centroid_fit(step_field,train,labels),step_field,labels,selection),
                     "lockbox":evaluate(centroid_fit(step_field,train,labels),step_field,labels,lock)})
    families=sorted({r["family"] for r in rows});train_f=set(families[:6]);selection_f=set(families[6:9]);lock_f=set(families[9:])
    family_train=np.asarray([r["family"] in train_f for r in rows]);family_selection=np.asarray([r["family"] in selection_f for r in rows]);family_lock=np.asarray([r["family"] in lock_f for r in rows])
    family_field=field_routes(signed,rows,family_train)[name];family_proto=centroid_fit(family_field,family_train,labels)
    family_holdout={"train_families":sorted(train_f),"selection_families":sorted(selection_f),"lockbox_families":sorted(lock_f),
                    "selection":evaluate(family_proto,family_field,labels,family_selection),"lockbox":evaluate(family_proto,family_field,labels,family_lock)}
    cells={}
    for factor in ("language","query"):
        for value in sorted({r[factor] for r in rows}):
            mask=lock & np.asarray([r[factor]==value for r in rows]);cells[f"{factor}:{value}"]=evaluate(prototype,selected,labels,mask)
    result={"success_definition":"first generated line exactly equals target two-word name","overall_success":float(np.mean(labels)),
            "split_counts":{s:sum(r["split"]==s for r in rows) for s in ("train","selection","lockbox")},
            "selection_trajectory":trajectory,"selected_route":name,"selected_qpoint":qpoint,"lockbox":lock_score,"sorted_lockbox":sorted_lock,
            "coordinate_advantage":lock_score["balanced_accuracy"]-sorted_lock["balanced_accuracy"],"post_token_diagnostic":post,
            "family_holdout":family_holdout,"lockbox_cells":cells,
            "nuisance_baselines":{"query_only_balanced_accuracy":0.9405594405594406,
                                  "language_query_balanced_accuracy":0.8251748251748252,
                                  "family_query_balanced_accuracy":0.9038461538461539},
            "gate":{"prospective_balanced_accuracy_pass":lock_score["balanced_accuracy"]>=0.60,
                    "coordinate_identity_pass":lock_score["balanced_accuracy"]>=sorted_lock["balanced_accuracy"]+0.05,
                    "unseen_family_pass":family_holdout["lockbox"]["balanced_accuracy"]>=0.60,
                    "beats_query_only_baseline":lock_score["balanced_accuracy"]>=0.9405594405594406+0.05}}
    result["gate"]["universal_generation_success_marker_passed"]=all(result["gate"].values())
    close_memmap(source);return result,{"selected_step0_field":selected,"row_sorted_control":sorted_field,"signed_hidden_control":signed}

def publish(rows:list[dict],analysis:dict,fields:dict[str,np.ndarray])->dict:
    matrix=np.concatenate(list(fields.values()));dataset_id="c9881_qwen4b_generation_success_coordinate_passport";binary=VIS/f"{dataset_id}.float32.npy"
    out=atlas.create_binary(binary.name,*matrix.shape,np.float32);out[:]=matrix;out.flush();close_memmap(out);metadata=[]
    for view in fields:
        metadata.extend({"case_id":r["case_id"],"family":r["family"],"language":r["language"],"query":r["query"],"unit":r["unit"],
                         "success":r["success"],"split":r["split"],"step":0,"qpoint":analysis["selected_qpoint"],"view":view} for r in rows)
    return atlas.write_metadata(dataset_id,"Qwen3-4B prospective generation-success coordinate passport",binary,metadata,"Qwen3-4B-FP16",
        "generation_success_coordinate_passport_v1","prospective diagnostic; not a semantic or causal gear",
        "384 autonomous fresh-lockbox generations; units12-13 train, unit14 select, unit15 lockbox",
        "pre-first-token step0, all 2560 coordinates with coordinate-sorted control",
        {"phase":PHASE,"campaign":CAMPAIGN,"coordinate_count":2560,"qpoint":analysis["selected_qpoint"],"no_topk":True})

def append_memo(result:dict)->None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):return
    stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text=rf"""

## Phase {PHASE}: 自主生成成败的输出前全坐标前瞻诊断（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 因上一大阶段结束后总目标仍是破解语言编码，自动继续研究Phase2352的384条真实贪心生成。把“第一行严格等于目标双词名称”作为成功，不使用teacher forcing标签；units12–13训练、unit14选路线/层、unit15锁箱。只用生成第一个token之前的step0场比较signed、absolute及去除语言/查询/族主效应的两种残差，全2560坐标；以失败/成功双质心预测，报告balanced accuracy和坐标排序控制。另做三族完全未见family锁箱，step>0只作事后诊断。

$$
BA=\frac12\left(\frac{{TP}}{{P}}+\frac{{TN}}{{N}}\right),\qquad
\hat z=\arg\min_{{z\in\{{0,1\}}}}\|X-\mu_z\|_2^2.
$$

**结果汇总。** 预测与锁箱 `{json.dumps(result['analysis'],ensure_ascii=False)}`；客户端热力图 `{json.dumps(result['dataset'],ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2356_c9881_c10040_generation_success_coordinate_diagnostic.py`；结果 `tests/glm5/result/phase2356_c9881_c10040_generation_success_coordinate_diagnostic`；客户端`c9881`。

**理论进展、问题硬伤与结论。** 若step0能跨未见unit和未见族预测成败，它只说明输出前状态含有通用失败风险标志，不等于编码了正确答案、更不等于因果齿轮；若仅step>0变强，则很可能是已生成token泄漏。必须同时看排序控制和未见族门。该Phase把下一阶段从“继续族分类”推进到“什么内部状态预示自主生成会兑现已知答案”。
"""
    with MEMO.open("a",encoding="utf-8",newline="\n") as h:h.write(text)

def main()->None:
    final=OUT/"analysis/final.json"
    if final.exists():
        result=json.loads(final.read_text(encoding="utf-8"));append_memo(result);print(json.dumps(result,ensure_ascii=False,indent=2));return
    rows=enrich_rows();analysis,fields=analyze(rows);dataset=publish(rows,analysis,fields);verification=atlas.verify(dataset);verified=all(v for k,v in verification.items() if k!="id")
    catalog=atlas.update_catalog([dataset]);build=atlas.frontend_build();checks={"rows":len(rows)==384,"asset":verified,"frontend_build":build["passed"]}
    result={"phase":PHASE,"campaign":CAMPAIGN,"analysis":analysis,"dataset":json.loads(json.dumps(dataset,default=str)),"verification":verification,
            "catalog":json.loads(json.dumps(catalog,default=str)),"frontend_build":build,"checks":checks,"all_checks_passed":all(checks.values())}
    save(final,result)
    if not result["all_checks_passed"]:raise RuntimeError(("phase2356_failed",checks))
    append_memo(result);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
