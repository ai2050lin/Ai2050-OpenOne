#!/usr/bin/env python3
"""Phase1401: Qwen3 fixed yes/no behavior qualification for C064."""
from __future__ import annotations
import inspect,json,math,sys
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
import phase1391_c062_family_factorized_behavior as runner
from phase1332_bf16_utils import load_bf16,quantization_audit,release_bf16
PHASE,CAMPAIGN=1401,"C064";CONTRACT=TESTS/"result/phase1400_c064_fixed_answer_factorial_contract";OUT=TESTS/"result/phase1401_c064_behavior";BATCH=12

def acc(rows):return sum(r["correct"] for r in rows)/len(rows)
def main():
 if (OUT/"analysis/final.json").exists():raise RuntimeError("Phase1401 already exists")
 cf=core.load(CONTRACT/"analysis/final.json");ca=core.load(CONTRACT/"audit/independent_final_audit.json");p=core.load(CONTRACT/"protocol/preregistration.json")
 if cf["authorization"]!="run_phase1401_c064_behavior" or not ca["all_checks_passed"]:raise RuntimeError("Phase1400 did not authorize")
 a0=core.rows(CONTRACT/"material/active_cases.jsonl");s0=core.rows(CONTRACT/"material/status_cases.jsonl");ac=core.rows(CONTRACT/"compiled/qwen3_active.jsonl");sc=core.rows(CONTRACT/"compiled/qwen3_status.jsonl");factors=core.rows(CONTRACT/"material/factor_sets.jsonl")
 model=None
 try:
  model,tok,device,placement=load_bf16("qwen3");quant=quantization_audit(model);pad=int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id);supports="logits_to_keep" in inspect.signature(model.forward).parameters
  ap=[];sp=[];first=None
  for start in range(0,len(ac),BATCH):
   vals=runner.forward(model,ac[start:start+BATCH],pad,device,supports)
   if start==0:first=vals
   ap.extend(vals)
  repeat=runner.forward(model,ac[:BATCH],pad,device,supports);numeric=max(abs(x["scores"][j]-y["scores"][j]) for x,y in zip(first,repeat) for j in range(2))
  for start in range(0,len(sc),BATCH):sp.extend(runner.forward(model,sc[start:start+BATCH],pad,device,supports))
  active=[{**x,**y,"correct":y["prediction"]==x["gold_position"]} for x,y in zip(a0,ap)];status=[{**x,**y,"correct":y["prediction"]==x["gold_position"]} for x,y in zip(s0,sp)]
  ab={r["case_id"]:r for r in active};sb={r["case_id"]:r for r in status};keys=("recipient","surface_same","member_same","family_same_polarity","polarity_same_family","family_and_polarity")
  eligible=[r for r in factors if all(ab[r[k]]["correct"] for k in keys) and sb[r["status_null"]]["correct"]]
  gate=p["behavior"];per=p["material"]["eligible_per_family_partition_surface"];family_results={};selected=[]
  for family in p["material"]["families"]:
   rows=[r for r in active if r["record_family"]==family];sr=[r for r in status if r["record_family"]==family];pairs=defaultdict(list)
   for r in rows:pairs[(r["pair"],r["index"],r["surface"],r["record_family"])].append(r["correct"])
   fe=[r for r in eligible if r["family"]==family];cells=defaultdict(list)
   for r in fe:cells[(r["partition"],r["surface"])].append(r)
   chosen=[]
   if len(cells)==9 and min(len(v) for v in cells.values())>=per:
    for key in sorted(cells):chosen.extend(sorted(cells[key],key=lambda r:r["set_id"])[:per])
   metrics={"active_count":len(rows),"active_accuracy":acc(rows),"partition":{q:acc([r for r in rows if r["partition"]==q]) for q in p["material"]["partitions"]},"surface":{q:acc([r for r in rows if r["surface"]==q]) for q in p["material"]["surfaces"]},"truth":{str(v).lower():acc([r for r in rows if r["truth"]==v]) for v in (True,False)},"pair_all_fraction":sum(all(v) for v in pairs.values())/len(pairs),"status_accuracy":acc(sr),"eligible_count":len(fe),"eligible_cell_min":min((len(v) for v in cells.values()),default=0),"selected_count":len(chosen)}
   checks={"active":metrics["active_accuracy"]>=gate["family_active_accuracy_min"],"partition":min(metrics["partition"].values())>=gate["family_partition_min"],"surface":min(metrics["surface"].values())>=gate["family_surface_min"],"truth":min(metrics["truth"].values())>=gate["family_truth_min"],"pair_all":metrics["pair_all_fraction"]>=gate["family_pair_all_min"],"status":metrics["status_accuracy"]>=gate["status_accuracy_min"],"eligible_cells":len(cells)==9 and metrics["eligible_cell_min"]>=per,"selected":len(chosen)==p["material"]["selected_per_family"]}
   family_results[family]={"metrics":metrics,"checks":checks,"qualified":all(checks.values())}
   if all(checks.values()):selected.extend(chosen)
  qualified=[f for f,v in family_results.items() if v["qualified"]];breadth={"family_count":len(qualified)>=p["material"]["minimum_qualified_families"],"status_global":acc(status)>=gate["status_accuracy_min"],"numeric":numeric<=gate["same_shape_repeat_max_abs_diff"],"finite":all(math.isfinite(z) for r in active+status for z in r["scores"])};passed=all(breadth.values())
  core.write_rows(OUT/"raw/active_behavior.jsonl",active);core.write_rows(OUT/"raw/status_behavior.jsonl",status);core.write_rows(OUT/"material/eligible_factor_sets.jsonl",selected)
  summary={"phase":PHASE,"campaign":CAMPAIGN,"family_results":family_results,"qualified_families":qualified,"breadth_checks":breadth,"behavior_qualified":passed,"selected_count":len(selected),"selected_partition_counts":{q:sum(r["partition"]==q for r in selected) for q in p["material"]["partitions"]},"global":{"active_accuracy":acc(active),"status_accuracy":acc(status),"numeric_same_shape_max_abs_diff":numeric},"runtime":{"placement":placement,"quantization":quant,"finished_at_utc":datetime.now(timezone.utc).isoformat()}}
  core.save(OUT/"analysis/qwen3_behavior_summary.json",summary);auth="run_phase1402_c064_state_swap_camera" if passed else "close_c064_at_behavior_gate";core.save(OUT/"analysis/final.json",{"phase":PHASE,"campaign":CAMPAIGN,"behavior_qualified":passed,"qualified_families":qualified,"authorization":auth});print(json.dumps(summary,ensure_ascii=False,indent=2))
 finally:
  if model is not None:release_bf16(model)
if __name__=="__main__":main()
