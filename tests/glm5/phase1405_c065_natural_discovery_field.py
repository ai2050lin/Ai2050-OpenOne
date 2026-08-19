#!/usr/bin/env python3
"""Phase1405: observe C065 natural full hidden-state factorial field."""
from __future__ import annotations
import json,math,statistics,sys
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";sys.path.insert(0,str(TESTS));import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16,quantization_audit,release_bf16
import phase1392_c062_full_field_camera as batcher
PHASE,CAMPAIGN=1405,"C065";CONTRACT=TESTS/"result/phase1403_c065_active_only_natural_state_contract";CAMERA=TESTS/"result/phase1404_c065_state_swap_camera";MATERIAL=TESTS/"result/phase1400_c064_fixed_answer_factorial_contract";OUT=TESTS/"result/phase1405_c065_natural_discovery_field"

def cosine_distance(a,b):
 a=a.float();b=b.float();return float(1-torch.dot(a,b)/(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)+1e-12))
def role_at(row,p):
 names=[k for k,v in row["role_positions"].items() if p in v];return names[0] if names else "physical"
def main():
 if (OUT/"analysis/final.json").exists():raise RuntimeError("Phase1405 exists")
 cf=core.load(CAMERA/"analysis/final.json");ca=core.load(CAMERA/"audit/independent_final_audit.json");p=core.load(CONTRACT/"protocol/preregistration.json")
 if cf["authorization"]!="run_phase1405_c065_natural_discovery_field" or not ca["all_checks_passed"]:raise RuntimeError("camera missing")
 selected=[r for r in core.rows(CONTRACT/"material/eligible_factor_sets.jsonl") if r["partition"]=="response_discovery"];compiled={r["case_id"]:r for r in core.rows(MATERIAL/"compiled/qwen3_active.jsonl")};model=None
 try:
  model,tok,device,placement=load_bf16("qwen3");quant=quantization_audit(model);pad=int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id);records=[]
  for ci,case in enumerate(selected):
   names=("recipient","member_same","family_same_polarity","polarity_same_family","family_and_polarity","surface_same");rows=[compiled[case[k]] for k in names];ids,mask,pos,offs=batcher.make_batch(rows,pad,device)
   with torch.inference_mode():out=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,output_hidden_states=True,return_dict=True)
   rec=rows[0];length=len(rec["prompt_ids"])
   same_layout=all(len(rows[j]["prompt_ids"])==length for j in range(5))
   if not same_layout:raise RuntimeError("same-surface donor layout mismatch")
   for state,hs in enumerate(out.hidden_states):
    for physical in range(length):
     p0=offs[0]+physical;role=role_at(rec,physical);base=hs[0,p0]
     distances={names[j]:cosine_distance(base,hs[j,offs[j]+physical]) for j in range(1,5)}
     surface_distance=None
     if role!="physical":
      surface_points=rows[5]["role_positions"][role]
      if len(surface_points)==1:surface_distance=cosine_distance(base,hs[5,offs[5]+surface_points[0]])
     records.append({"set_id":case["set_id"],"family":case["family"],"surface":case["surface"],"state_index":state,"position":physical,"role":role,"member_distance":distances["member_same"],"family_distance":distances["family_same_polarity"],"polarity_distance":distances["polarity_same_family"],"family_polarity_distance":distances["family_and_polarity"],"surface_role_distance":surface_distance,"family_identity_score":distances["family_same_polarity"]-distances["member_same"],"joint_polarity_score":distances["polarity_same_family"]-distances["member_same"]})
   del out,ids,mask,pos
  core.write_rows(OUT/"raw/natural_full_field.jsonl",records)
  groups=defaultdict(list)
  for r in records:groups[(r["surface"],r["state_index"],r["position"],r["role"])].append(r)
  aggregates=[]
  for (surface,state,position,role),values in groups.items():
   aggregates.append({"surface":surface,"state_index":state,"position":position,"role":role,"count":len(values),"family_identity_score_median":statistics.median(v["family_identity_score"] for v in values),"joint_polarity_score_median":statistics.median(v["joint_polarity_score"] for v in values),"family_distance_median":statistics.median(v["family_distance"] for v in values),"polarity_distance_median":statistics.median(v["polarity_distance"] for v in values),"member_distance_median":statistics.median(v["member_distance"] for v in values)})
  core.write_rows(OUT/"analysis/natural_field_aggregates.jsonl",aggregates)
  candidates=[]
  for surface in p["material"].get("surfaces",["ordinary","catalog","statement"]):
   for object_name,score_key in (("family_identity","family_identity_score_median"),("joint_polarity","joint_polarity_score_median")):
    for window_index,(lo,hi) in enumerate(p["observation"]["windows"]):
     pool=[r for r in aggregates if r["surface"]==surface and lo<=r["state_index"]<=hi and r["role"]!="physical"]
     chosen=sorted(pool,key=lambda r:(-r[score_key],-r["count"],r["state_index"],r["position"],r["role"]))[0]
     candidates.append({"candidate_id":f"{surface}:{object_name}:w{window_index}","surface":surface,"object":object_name,"window_index":window_index,"state_index":chosen["state_index"],"position":chosen["position"],"role":chosen["role"],"score":chosen[score_key],"count":chosen["count"]})
  core.save(OUT/"protocol/frozen_natural_event_candidates.json",{"schema":"c065.natural_candidates.v1","source_partition":"response_discovery","candidates":candidates,"source_field_sha256":core.sha(OUT/"raw/natural_full_field.jsonl")})
  checks={"case_count":len(selected)==18,"record_count":len(records)==sum(37*len(compiled[r["recipient"]]["prompt_ids"]) for r in selected),"state_coverage":set(r["state_index"] for r in records)==set(range(37)),"all_positions":all(len({r["position"] for r in records if r["set_id"]==case["set_id"] and r["state_index"]==0})==len(compiled[case["recipient"]]["prompt_ids"]) for case in selected),"finite":all(math.isfinite(r[k]) for r in records for k in ("member_distance","family_distance","polarity_distance","family_identity_score","joint_polarity_score")),"candidate_count":len(candidates)==18,"candidate_roles":all(r["role"]!="physical" for r in candidates),"discovery_only":set(r["set_id"] for r in records)=={r["set_id"] for r in selected},"bf16":quant["has_bf16_parameters"],"not_quantized":not quant["has_quantized_modules"]}
  summary={"phase":PHASE,"campaign":CAMPAIGN,"case_count":len(selected),"record_count":len(records),"aggregate_count":len(aggregates),"candidate_count":len(candidates),"candidate_preview":candidates,"checks":checks,"all_checks_passed":all(checks.values()),"runtime":{"placement":placement,"quantization":quant,"finished_at_utc":datetime.now(timezone.utc).isoformat()}}
  core.save(OUT/"analysis/field_summary.json",summary);core.save(OUT/"analysis/final.json",{"phase":PHASE,"campaign":CAMPAIGN,"all_checks_passed":summary["all_checks_passed"],"authorization":"run_phase1406_c065_holdout_factorial_swaps" if summary["all_checks_passed"] else "close_c065_at_field_gate"});print(json.dumps({k:v for k,v in summary.items() if k!="candidate_preview"},indent=2));print(json.dumps({"candidates":candidates},indent=2))
 finally:
  if model is not None:release_bf16(model)
if __name__=="__main__":main()
