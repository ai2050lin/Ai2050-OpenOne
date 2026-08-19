#!/usr/bin/env python3
"""Independent pre/post audit for Phase1336."""
from __future__ import annotations
import argparse,json,math,sys
from collections import defaultdict
from pathlib import Path
from statistics import median
ROOT=Path(__file__).resolve().parents[2];T=ROOT/"tests/glm5";sys.path.insert(0,str(T))
import phase1331_relational_measurement_core as core  # noqa: E402
PARENT=T/"result/phase1335_c045_standard_executor_contract";OUT=T/"result/phase1336_c045_standard_behavior";MODELS=("qwen3","glm4","deepseek7b")


def pre():
 m=core.load(OUT/"protocol/execution_manifest.json");p=core.load(PARENT/"protocol/preregistration.json");a=core.load(PARENT/"audit/independent_final_audit.json")
 frozen={k:v for k,v in m.items() if k not in {"manifest_sha256","created_at_utc"}}
 checks={"parent":p["authorization"]=="run_phase1336_c045_standard_behavior" and a["all_checks_passed"],"hash":core.digest(frozen)==m["manifest_sha256"],
 "sources":core.sha(T/"phase1336_c045_standard_behavior.py")==m["script_sha256"] and core.sha(Path(__file__).resolve())==m["auditor_sha256"] and core.sha(T/"phase1332_bf16_utils.py")==m["util_sha256"],
 "parent_hash":core.sha(PARENT/"protocol/preregistration.json")==m["parent_protocol_sha256"],"order":m["model_order"]==list(MODELS),"batch":m["batch_size"]==8,
 "groups":all(len(v["cohort_a"])==len(v["cohort_permuted"])==6 for v in m["executor_groups"].values()),
 "no_results":not any((OUT/f"raw/{x}_executor.jsonl").exists() for x in MODELS)}
 result={"phase":1336,"stage":"pre_model","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"authorization":"run_models_in_frozen_order" if all(checks.values()) else "none"}
 core.save(OUT/"audit/independent_preaudit.json",result);print(json.dumps(result,indent=2));
 if not result["all_checks_passed"]:raise SystemExit(1)


def post():
 p=core.load(PARENT/"protocol/preregistration.json");m=core.load(OUT/"protocol/execution_manifest.json");f=core.load(OUT/"analysis/final.json");checks={};qualified=[]
 for model in MODELS:
  er=core.rows(OUT/f"raw/{model}_executor.jsonl");br=core.rows(OUT/f"raw/{model}_behavior.jsonl");s=core.load(OUT/f"analysis/{model}_summary.json");rt=core.load(OUT/f"runtime/{model}.json")
  vals=[v for r in er for k in ("cohort_a","cohort_permuted","cohort_a_repeat") for v in r[k]];rank=sum((r["cohort_a"][0]>r["cohort_a"][1])==(r["cohort_permuted"][0]>r["cohort_permuted"][1]) for r in er)/len(er)
  pd=max(abs(a-b) for r in er for a,b in zip(r["cohort_a"],r["cohort_permuted"]));rd=max(abs(a-b) for r in er for a,b in zip(r["cohort_a"],r["cohort_a_repeat"]));eg=p["executor_gate"]
  em={"finite_fraction":sum(math.isfinite(v) for v in vals)/len(vals),"permuted_rank_agreement":rank,"permuted_max_abs_score_diff":pd,"repeat_max_abs_score_diff":rd,"case_count":len(er)}
  egs={"finite":em["finite_fraction"]>=eg["finite_fraction_min"],"rank":rank>=eg["permuted_rank_agreement_min"],"permuted":pd<=eg["permuted_max_abs_score_diff_max"],"repeat":rd<=eg["repeat_max_abs_score_diff_max"],"count":len(er)==48};eq=all(egs.values())
  checks[f"{model}_executor"]=s["executor_metrics"]==em and s["executor_gates"]==egs and s["executor_qualified"]==eq and len(er)==48
  checks[f"{model}_behavior_count"]=(len(br)==480 if eq else len(br)==0)
  if eq:
   binary=[r for r in br if r["interface"]=="binary"];choice=[r for r in br if r["interface"]=="choice"];generation=[r for r in br if r["interface"]=="generation"]
   def acc(rows,key,value):
    selected=[r for r in rows if r[key]==value];return sum(r["correct"] for r in selected)/len(selected)
   pairs=defaultdict(list)
   for r in binary:pairs[r["pair_key"]].append(r["correct"])
   bm={"accuracy":sum(r["correct"] for r in binary)/len(binary),"partition":{x:acc(binary,"partition",x) for x in ("discovery","confirmation","holdout")},
       "surface":{x:acc(binary,"surface",x) for x in sorted({r["surface"] for r in binary})},
       "polarity":{str(x):sum(r["correct"] for r in binary if r["truth"]==x)/sum(r["truth"]==x for r in binary) for x in (True,False)},
       "paired_success":sum(len(v)==2 and all(v) for v in pairs.values())/len(pairs),"median_margin":median(r["margin"] for r in binary)}
   cm={"accuracy":sum(r["correct"] for r in choice)/len(choice),"partition":{x:acc(choice,"partition",x) for x in ("discovery","confirmation","holdout")},
       "surface":{x:acc(choice,"surface",x) for x in sorted({r["surface"] for r in choice})},"median_margin":median(r["margin"] for r in choice)}
   gm={"accuracy":sum(r["normalized"]==r["gold"] for r in generation)/len(generation),
       "partition":{x:sum(r["normalized"]==r["gold"] for r in generation if r["partition"]==x)/sum(r["partition"]==x for r in generation) for x in ("discovery","confirmation","holdout")},
       "surface":{x:sum(r["normalized"]==r["gold"] for r in generation if r["surface"]==x)/sum(r["surface"]==x for r in generation) for x in sorted({r["surface"] for r in generation})}}
   bg=p["behavior"]["binary_gate"];cg=p["behavior"]["choice_gate"];gg=p["behavior"]["generation_gate"]
   bgs={"binary_accuracy":bm["accuracy"]>=bg["accuracy_min"],"binary_partition":min(bm["partition"].values())>=bg["partition_min"],
        "binary_surface":min(bm["surface"].values())>=bg["surface_min"],"binary_polarity":min(bm["polarity"].values())>=bg["polarity_min"],
        "binary_pairs":bm["paired_success"]>=bg["paired_success_min"],"binary_margin":bm["median_margin"]>=bg["median_margin_min"],
        "choice_accuracy":cm["accuracy"]>=cg["accuracy_min"],"choice_partition":min(cm["partition"].values())>=cg["partition_min"],
        "choice_surface":min(cm["surface"].values())>=cg["surface_min"],"choice_margin":cm["median_margin"]>=cg["median_margin_min"],
        "generation_accuracy":gm["accuracy"]>=gg["exact_normalized_accuracy_min"],"generation_partition":min(gm["partition"].values())>=gg["partition_min"],
        "generation_surface":min(gm["surface"].values())>=gg["surface_min"]}
   bq=all(bgs.values())
   checks[f"{model}_behavior_metrics"]=s["behavior_metrics"]=={"binary":bm,"choice":cm,"generation":gm} and s["behavior_gates"]==bgs and s["behavior_qualified"]==bq
   checks[f"{model}_qualified"]=s["qualified"]==(eq and bq)
  else:
   checks[f"{model}_behavior_metrics"]=s["behavior_metrics"]=={} and s["behavior_gates"]=={} and not s["behavior_qualified"]
   checks[f"{model}_qualified"]=not s["qualified"]
  checks[f"{model}_summary_hash"]=core.sha(OUT/f"analysis/{model}_summary.json")==f["model_summary_sha256"][model]
  qa=rt["quantization_audit"];checks[f"{model}_runtime"]=qa["has_bf16_parameters"] and not qa["has_quantized_modules"]
  if s["qualified"]:qualified.append(model)
 passed=len(qualified)>=p["behavior"]["minimum_authorized_models"];checks["final_models"]=f["qualified_models"]==qualified
 checks["final_branch"]=f["all_gates_passed"]==passed and f["authorization"]==("run_phase1337_c045_hidden_relation_field" if passed else "close_c045_standard_behavior")
 result={"phase":1336,"campaign":"C045","checks":checks,"independently_qualified_models":qualified,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"authorization":f["authorization"] if all(checks.values()) else "none"}
 core.save(OUT/"audit/independent_final_audit.json",result);print(json.dumps(result,indent=2));
 if not result["all_checks_passed"]:raise SystemExit(1)


if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=("pre","post"),required=True);a=ap.parse_args();pre() if a.stage=="pre" else post()
