#!/usr/bin/env python3
"""Phase1497: C086 behavior and 32-case composition-set stratification."""
from __future__ import annotations
import inspect,json,math,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; C=TESTS/"result/phase1496_c086_unlabeled_counterbalanced_contract"; OUT=TESTS/"result/phase1497_c086_behavior_stratification"; sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
import phase1391_c062_family_factorized_behavior as runner
import phase1457_c077_behavior as metric
from phase1332_bf16_utils import load_bf16,quantization_audit,release_bf16
BATCH=24
def balanced_output_accuracy(rows):
 positive=[r for r in rows if r["output_yes"]]; negative=[r for r in rows if not r["output_yes"]]
 return (metric.accuracy(positive)+metric.accuracy(negative))/2.0
def summarize(rows,sets,protocol,repeat,quant):
 keys=tuple(f"{s}_{c}_{cell}" for s in protocol["surfaces"] for c in protocol["codebooks"] for cell in protocol["cells"]); by={r["case_id"]:r for r in rows}; strat=[]
 for g in sets:
  n=sum(by[g[k]]["correct"] for k in keys); st="success" if n==32 else ("failed" if n==0 else "mixed"); strat.append({**g,"correct_count":n,"case_count":32,"stratum":st})
 interface={}
 for s in protocol["surfaces"]:
  interface[s]={}
  for c in protocol["codebooks"]:
   values=[r for r in rows if r["surface"]==s and r["codebook"]==c]; interface[s][c]={"count":len(values),"accuracy":metric.accuracy(values),"balanced_accuracy":balanced_output_accuracy(values),"semantic_truth_accuracy":{str(v).lower():metric.accuracy([r for r in values if r["semantic_truth"]==v]) for v in (True,False)}}
 relation={rel:{c:balanced_output_accuracy([r for r in rows if r["record_relation_id"]==rel and r["codebook"]==c]) for c in protocol["codebooks"]} for rel in protocol["relations"]}; counts=Counter(r["stratum"] for r in strat)
 checks={"count":len(rows)==6912,"sets":len(strat)==216 and sum(counts.values())==216,"repeat":repeat<=1e-6,"finite":all(math.isfinite(v) for r in rows for v in r["scores"]),"bf16":quant["has_bf16_parameters"],"not_quantized":not quant["has_quantized_modules"],"hidden_not_accessed":True}
 return {"phase":1497,"campaign":"C086","global_accuracy":metric.accuracy(rows),"global_balanced_accuracy":balanced_output_accuracy(rows),"interface":interface,"relation_codebook_balanced_accuracy":relation,"stratum_counts":dict(counts),"stratum_partition_counts":{st:{p:sum(r["stratum"]==st and r["partition"]==p for r in strat) for p in protocol["partitions"]} for st in ("success","mixed","failed")},"error_count":sum(not r["correct"] for r in rows),"error_codebook":dict(Counter(r["codebook"] for r in rows if not r["correct"])),"error_semantic_truth":dict(Counter(str(r["semantic_truth"]).lower() for r in rows if not r["correct"])),"numeric_repeat_max_abs_diff":repeat,"checks":checks,"all_integrity_checks_passed":all(checks.values()),"hidden_state_accessed":False},strat
def main():
 if (OUT/"analysis/final.json").exists(): raise RuntimeError("Phase1497 exists")
 cf=core.load(C/"analysis/final.json"); ca=core.load(C/"audit/independent_final_audit.json"); protocol=core.load(C/"protocol/preregistration.json")
 if cf["authorization"]!="run_phase1497_c086_behavior_stratification" or not ca["all_checks_passed"]: raise RuntimeError("Phase1496 authorization missing")
 source=core.rows(C/"material/active_cases.jsonl"); compiled=core.rows(C/"compiled/qwen3_active.jsonl"); sets=core.rows(C/"material/composition_sets.jsonl"); model=None
 try:
  model,tok,device,placement=load_bf16("qwen3"); quant=quantization_audit(model); pad=int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id); supports="logits_to_keep" in inspect.signature(model.forward).parameters; preds=[]; first=None
  for start in range(0,len(compiled),BATCH):
   block=runner.forward(model,compiled[start:start+BATCH],pad,device,supports)
   if first is None:first=block
   preds.extend(block)
  second=runner.forward(model,compiled[:BATCH],pad,device,supports); repeat=max(abs(a["scores"][i]-b["scores"][i]) for a,b in zip(first,second) for i in range(2)); rows=[{**r,**p,"correct":p["prediction"]==r["gold_position"]} for r,p in zip(source,preds)]; summary,strat=summarize(rows,sets,protocol,repeat,quant); summary["runtime"]={"placement":placement,"quantization":quant,"finished_at_utc":datetime.now(timezone.utc).isoformat()}
  if not summary["all_integrity_checks_passed"]:raise RuntimeError(summary["checks"])
  core.write_rows(OUT/"raw/behavior.jsonl",rows); core.write_rows(OUT/"material/stratified_composition_sets.jsonl",strat); core.save(OUT/"analysis/behavior_stratification_summary.json",summary); core.save(OUT/"analysis/final.json",{"phase":1497,"campaign":"C086","status":"behavior_stratification_complete","stratum_counts":summary["stratum_counts"],"authorization":"run_phase1498_c086_all_case_field_capture"}); print(json.dumps({k:v for k,v in summary.items() if k!="runtime"},indent=2))
 finally:
  if model is not None:release_bf16(model)
if __name__=="__main__":main()
