#!/usr/bin/env python3
"""Phase1336: standard-executor replay and multi-interface behavior for C045."""
from __future__ import annotations
import argparse, hashlib, json, math, re, string, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any
import torch

ROOT=Path(__file__).resolve().parents[2]; T=ROOT/"tests/glm5"; sys.path.insert(0,str(T))
import phase1331_relational_measurement_core as core  # noqa: E402
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16  # noqa: E402

PHASE,CAMPAIGN=1336,"C045"; SCRIPT=Path(__file__).resolve(); AUDITOR=T/"phase1336_c045_standard_behavior_audit.py"
UTIL=T/"phase1332_bf16_utils.py"; PARENT=T/"result/phase1335_c045_standard_executor_contract"
OUT=T/"result/phase1336_c045_standard_behavior"; MODELS=("qwen3","glm4","deepseek7b")


def parent_ok():
    p=core.load(PARENT/"protocol/preregistration.json"); a=core.load(PARENT/"audit/independent_final_audit.json")
    if p["authorization"]!="run_phase1336_c045_standard_behavior" or not a["all_checks_passed"]:raise RuntimeError("not authorized")
    return p


def prepare(force):
    p=parent_ok(); path=OUT/"protocol/execution_manifest.json"
    if path.exists() and not force:raise RuntimeError(f"{path} exists")
    if any((OUT/f"raw/{m}_behavior.jsonl").exists() for m in MODELS):raise RuntimeError("results exist")
    case_ids=set(p["executor_gate"]["case_ids"]); widths={}; generation_widths={}; groups={}
    for m in MODELS:
        compiled=core.rows(PARENT/f"compiled/{m}_behavior.jsonl")
        scoring=[row for row in compiled if row["interface"] in {"binary","choice"}]
        widths[m]=max(len(row["prompt_ids"])+len(candidate) for row in scoring for candidate in row["candidate_ids"])
        generation=[row for row in compiled if row["interface"]=="generation"]
        generation_widths[m]=max(len(row["prompt_ids"]) for row in generation)+6
        selected=[row for row in compiled if row["case_id"] in case_ids]
        order=[row["case_id"] for row in selected]; permuted=order[::2]+order[1::2]
        groups[m]={"cohort_a":[order[i:i+8] for i in range(0,48,8)],
                   "cohort_permuted":[permuted[i:i+8] for i in range(0,48,8)]}
    frozen={"phase":PHASE,"campaign":CAMPAIGN,"schema":"phase1336.c045.standard_behavior.v1",
            "parent_protocol_sha256":core.sha(PARENT/"protocol/preregistration.json"),"parent_contract_sha256":p["contract_sha256"],
            "model_order":list(MODELS),"batch_size":8,"precision":"bfloat16-no-quantization","padding_side":"right",
            "explicit_position_ids":True,"score_dtype":"float32_log_softmax","score_width_by_model":widths,
            "generation_width_by_model":generation_widths,"generation_max_new_tokens":6,"executor_groups":groups,
            "executor_gate":p["executor_gate"],"behavior_gate":p["behavior"],"overwrite_after_run":False,
            "script_sha256":core.sha(SCRIPT),"auditor_sha256":core.sha(AUDITOR),"util_sha256":core.sha(UTIL)}
    frozen["manifest_sha256"]=core.digest(frozen);frozen["created_at_utc"]=datetime.now(timezone.utc).isoformat()
    core.save(path,frozen);print(json.dumps(frozen,indent=2))


def tensors(batch,width,pad_id,device,key="prompt_ids"):
    ids=torch.full((len(batch),width),int(pad_id),dtype=torch.long,device=device);mask=torch.zeros_like(ids);lengths=[]
    for i,row in enumerate(batch):
        seq=torch.tensor(row[key],dtype=torch.long,device=device);ids[i,:len(seq)]=seq;mask[i,:len(seq)]=1;lengths.append(len(seq))
    pos=mask.cumsum(-1)-1;pos.masked_fill_(mask==0,0);return ids,mask,pos,lengths


def prompt_scores(model,device,batch,width,pad_id):
    ids,mask,pos,lengths=tensors(batch,width,pad_id,device)
    with torch.inference_mode(): logits=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False).logits
    out=[]
    for i,row in enumerate(batch):
        lp=torch.log_softmax(logits[i,lengths[i]-1].float(),-1);out.append([float(lp[c[0]].item()) for c in row["candidate_ids"]])
    del ids,mask,pos,logits;return out


def sequence_scores(model,device,jobs,width,pad_id):
    output=[]
    for start in range(0,len(jobs),8):
        batch=jobs[start:start+8];ids,mask,pos,_=tensors(batch,width,pad_id,device,key="sequence")
        with torch.inference_mode(): logits=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False).logits
        log_probs=torch.log_softmax(logits.float(),-1)
        for i,row in enumerate(batch):
            values=[float(log_probs[i,row["prompt_len"]+j-1,token].item()) for j,token in enumerate(row["candidate"])]
            output.append(sum(values)/len(values))
        del ids,mask,pos,logits,log_probs
    return output


def manual_generate(model,device,batch,width,pad_id,eos_ids,max_new):
    ids,mask,pos,lengths=tensors(batch,width,pad_id,device);generated=[[] for _ in batch];active=[True]*len(batch)
    for _ in range(max_new):
        with torch.inference_mode(): logits=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False).logits
        for i in range(len(batch)):
            if not active[i]:continue
            token=int(torch.argmax(logits[i,lengths[i]-1].float()).item());generated[i].append(token)
            if token in eos_ids or lengths[i]>=width:active[i]=False;continue
            ids[i,lengths[i]]=token;mask[i,lengths[i]]=1;pos[i,lengths[i]]=lengths[i];lengths[i]+=1
        del logits
        if not any(active):break
    del ids,mask,pos;return generated


def normalize(text):
    value=text.lower().strip();value=re.sub(r"[\s]+"," ",value);return value.strip(string.whitespace+string.punctuation)


def partition_metric(records,key,value):
    chosen=[row for row in records if row[key]==value];return sum(row["correct"] for row in chosen)/len(chosen)


def run_model(model_name):
    p=parent_ok();manifest=core.load(OUT/"protocol/execution_manifest.json")
    frozen={k:v for k,v in manifest.items() if k not in {"manifest_sha256","created_at_utc"}}
    if core.digest(frozen)!=manifest["manifest_sha256"]:raise RuntimeError("manifest hash")
    result_path=OUT/f"analysis/{model_name}_summary.json"
    if result_path.exists():raise RuntimeError("formal result exists")
    source=core.rows(PARENT/"material/frozen_behavior_cases.jsonl");compiled=core.rows(PARENT/f"compiled/{model_name}_behavior.jsonl")
    if len(source)!=len(compiled) or any(a["case_id"]!=b["case_id"] for a,b in zip(source,compiled)):raise RuntimeError("compiled mismatch")
    by_id={row["case_id"]:row for row in compiled}; model=None
    print(f"[Phase1336] loading {model_name}",flush=True)
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    try:
        model,tokenizer,device,placement=load_bf16(model_name);pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        width=int(manifest["score_width_by_model"][model_name]); groups=manifest["executor_groups"][model_name]
        def run_groups(group_values):
            out={}
            for group in group_values:
                batch=[by_id[x] for x in group];scores=prompt_scores(model,device,batch,width,int(pad));out.update(zip(group,scores))
            return out
        cohort=run_groups(groups["cohort_a"]);permuted=run_groups(groups["cohort_permuted"]);repeat=run_groups(groups["cohort_a"])
        executor_rows=[]
        for case_id in manifest["executor_gate"]["case_ids"]:
            executor_rows.append({"case_id":case_id,"cohort_a":cohort[case_id],"cohort_permuted":permuted[case_id],"cohort_a_repeat":repeat[case_id]})
        executor_scores=[v for r in executor_rows for key in ("cohort_a","cohort_permuted","cohort_a_repeat") for v in r[key]]
        perm_diff=max(abs(a-b) for r in executor_rows for a,b in zip(r["cohort_a"],r["cohort_permuted"]))
        repeat_diff=max(abs(a-b) for r in executor_rows for a,b in zip(r["cohort_a"],r["cohort_a_repeat"]))
        rank=sum((r["cohort_a"][0]>r["cohort_a"][1])==(r["cohort_permuted"][0]>r["cohort_permuted"][1]) for r in executor_rows)/len(executor_rows)
        eg=p["executor_gate"];executor_metrics={"finite_fraction":sum(math.isfinite(v) for v in executor_scores)/len(executor_scores),
            "permuted_rank_agreement":rank,"permuted_max_abs_score_diff":perm_diff,"repeat_max_abs_score_diff":repeat_diff,"case_count":len(executor_rows)}
        executor_gates={"finite":executor_metrics["finite_fraction"]>=eg["finite_fraction_min"],"rank":rank>=eg["permuted_rank_agreement_min"],
                        "permuted":perm_diff<=eg["permuted_max_abs_score_diff_max"],"repeat":repeat_diff<=eg["repeat_max_abs_score_diff_max"],"count":len(executor_rows)==48}
        executor_qualified=all(executor_gates.values())
        behavior_records=[]; behavior_metrics={};behavior_gates={};behavior_qualified=False
        if executor_qualified:
            score_rows=[(a,b) for a,b in zip(source,compiled) if a["interface"] in {"binary","choice"}]
            jobs=[]
            for src,row in score_rows:
                for index,candidate in enumerate(row["candidate_ids"]):jobs.append({"case_id":src["case_id"],"candidate_index":index,
                    "sequence":row["prompt_ids"]+candidate,"prompt_len":len(row["prompt_ids"]),"candidate":candidate})
            values=sequence_scores(model,device,jobs,width,int(pad));score_map=defaultdict(dict)
            for job,value in zip(jobs,values):score_map[job["case_id"]][job["candidate_index"]]=value
            for src,_ in score_rows:
                scores=[score_map[src["case_id"]][i] for i in range(len(src["candidates"]))];gold=src["gold_position"]
                wrong=max(v for i,v in enumerate(scores) if i!=gold);correct=scores[gold]>wrong
                behavior_records.append({"case_id":src["case_id"],"interface":src["interface"],"partition":src["partition"],
                    "surface":src["surface"],"truth":src.get("truth"),"pair_key":src.get("pair_key"),"scores":scores,
                    "gold_position":gold,"margin":scores[gold]-wrong,"correct":correct})
            generation_pairs=[(a,b) for a,b in zip(source,compiled) if a["interface"]=="generation"]
            eos=set(getattr(model.config,"eos_token_id",[]) if isinstance(getattr(model.config,"eos_token_id",None),list) else [getattr(model.config,"eos_token_id",tokenizer.eos_token_id)])
            eos.add(tokenizer.eos_token_id)
            for start in range(0,len(generation_pairs),8):
                pairs=generation_pairs[start:start+8];tokens=manual_generate(model,device,[b for _,b in pairs],int(manifest["generation_width_by_model"][model_name]),int(pad),eos,manifest["generation_max_new_tokens"])
                for (src,_),ids in zip(pairs,tokens):
                    text=tokenizer.decode(ids,skip_special_tokens=True);norm=normalize(text);correct=norm in src["accepted_normalized_outputs"]
                    behavior_records.append({"case_id":src["case_id"],"interface":"generation","partition":src["partition"],
                        "surface":src["surface"],"target_family":src["target_family"],"generated_token_ids":ids,
                        "generated_text":text,"normalized":norm,"gold":src["gold_value"],"correct":correct})
            binary=[r for r in behavior_records if r["interface"]=="binary"];choice=[r for r in behavior_records if r["interface"]=="choice"];generation=[r for r in behavior_records if r["interface"]=="generation"]
            pair_groups=defaultdict(list)
            for r in binary:pair_groups[r["pair_key"]].append(r["correct"])
            bm={"accuracy":sum(r["correct"] for r in binary)/len(binary),"partition":{x:partition_metric(binary,"partition",x) for x in ("discovery","confirmation","holdout")},
                "surface":{x:partition_metric(binary,"surface",x) for x in sorted({r["surface"] for r in binary})},
                "polarity":{str(x):sum(r["correct"] for r in binary if r["truth"]==x)/sum(r["truth"]==x for r in binary) for x in (True,False)},
                "paired_success":sum(len(v)==2 and all(v) for v in pair_groups.values())/len(pair_groups),"median_margin":median(r["margin"] for r in binary)}
            cm={"accuracy":sum(r["correct"] for r in choice)/len(choice),"partition":{x:partition_metric(choice,"partition",x) for x in ("discovery","confirmation","holdout")},
                "surface":{x:partition_metric(choice,"surface",x) for x in sorted({r["surface"] for r in choice})},"median_margin":median(r["margin"] for r in choice)}
            gm={"accuracy":sum(r["correct"] for r in generation)/len(generation),"partition":{x:partition_metric(generation,"partition",x) for x in ("discovery","confirmation","holdout")},
                "surface":{x:partition_metric(generation,"surface",x) for x in sorted({r["surface"] for r in generation})}}
            bg=p["behavior"]["binary_gate"];cg=p["behavior"]["choice_gate"];gg=p["behavior"]["generation_gate"]
            behavior_gates={"binary_accuracy":bm["accuracy"]>=bg["accuracy_min"],"binary_partition":min(bm["partition"].values())>=bg["partition_min"],
                "binary_surface":min(bm["surface"].values())>=bg["surface_min"],"binary_polarity":min(bm["polarity"].values())>=bg["polarity_min"],
                "binary_pairs":bm["paired_success"]>=bg["paired_success_min"],"binary_margin":bm["median_margin"]>=bg["median_margin_min"],
                "choice_accuracy":cm["accuracy"]>=cg["accuracy_min"],"choice_partition":min(cm["partition"].values())>=cg["partition_min"],
                "choice_surface":min(cm["surface"].values())>=cg["surface_min"],"choice_margin":cm["median_margin"]>=cg["median_margin_min"],
                "generation_accuracy":gm["accuracy"]>=gg["exact_normalized_accuracy_min"],"generation_partition":min(gm["partition"].values())>=gg["partition_min"],
                "generation_surface":min(gm["surface"].values())>=gg["surface_min"]}
            behavior_metrics={"binary":bm,"choice":cm,"generation":gm};behavior_qualified=all(behavior_gates.values())
        core.write_rows(OUT/f"raw/{model_name}_executor.jsonl",executor_rows);core.write_rows(OUT/f"raw/{model_name}_behavior.jsonl",behavior_records)
        runtime={"model":model_name,"device":str(device),"placement":placement,"quantization_audit":quantization_audit(model),
                 "peak_cuda_bytes":int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,"completed_at_utc":datetime.now(timezone.utc).isoformat()}
        core.save(OUT/f"runtime/{model_name}.json",runtime)
        summary={"phase":PHASE,"campaign":CAMPAIGN,"model":model_name,"executor_metrics":executor_metrics,"executor_gates":executor_gates,
                 "executor_qualified":executor_qualified,"behavior_metrics":behavior_metrics,"behavior_gates":behavior_gates,
                 "behavior_qualified":behavior_qualified,"qualified":executor_qualified and behavior_qualified,
                 "executor_raw_sha256":core.sha(OUT/f"raw/{model_name}_executor.jsonl"),"behavior_raw_sha256":core.sha(OUT/f"raw/{model_name}_behavior.jsonl"),
                 "runtime_sha256":core.sha(OUT/f"runtime/{model_name}.json"),"finished_at_utc":datetime.now(timezone.utc).isoformat()}
        core.save(result_path,summary);print(json.dumps(summary,indent=2))
    finally:
        if model is not None:release_bf16(model)
        print(f"[Phase1336] released {model_name}",flush=True)


def finalize():
    p=parent_ok();summaries={m:core.load(OUT/f"analysis/{m}_summary.json") for m in MODELS};qualified=[m for m in MODELS if summaries[m]["qualified"]]
    passed=len(qualified)>=p["behavior"]["minimum_authorized_models"]
    final={"phase":PHASE,"campaign":CAMPAIGN,"qualified_models":qualified,"qualified_model_count":len(qualified),"all_gates_passed":passed,
           "authorization":"run_phase1337_c045_hidden_relation_field" if passed else "close_c045_standard_behavior",
           "model_summary_sha256":{m:core.sha(OUT/f"analysis/{m}_summary.json") for m in MODELS},"manifest_sha256":core.sha(OUT/"protocol/execution_manifest.json"),
           "finished_at_utc":datetime.now(timezone.utc).isoformat()}
    core.save(OUT/"analysis/final.json",final);print(json.dumps(final,indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument("--prepare",action="store_true");g.add_argument("--model",choices=MODELS);g.add_argument("--finalize",action="store_true");ap.add_argument("--force",action="store_true");a=ap.parse_args()
    prepare(a.force) if a.prepare else run_model(a.model) if a.model else finalize()
