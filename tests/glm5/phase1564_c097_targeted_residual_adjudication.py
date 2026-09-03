#!/usr/bin/env python3
"""Phase1564: adjudicate C097-A with pooled, split, and bootstrap evidence."""
from __future__ import annotations
import json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; RESULT=TESTS/"result"; CONTRACT=RESULT/"phase1562_c097_targeted_residual_contract"; PARENT=RESULT/"phase1563_c097_targeted_residual_capture"; C095=RESULT/"phase1554_c095_triadic_interaction_and_field_atlas"; C096=RESULT/"phase1559_c096_fresh_prediction_atlas_and_adjudication"; OUT=RESULT/"phase1564_c097_targeted_residual_adjudication"
sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core
from phase1561_c097_common import cosine,FOCUS_STATES,ROLES
def old_stack(directory,prefix):
 arr=np.load(directory/f"raw/{prefix}.float16.npy",mmap_mode="r"); idx=core.rows(directory/f"raw/{prefix}_index.jsonl"); ids=[r["row_index"] for r in idx if r["surface"]=="postquery" and r["concreteness"]=="concrete" and r["family_a"]=="similarity" and r["family_b"]=="class_inclusion"]
 return np.asarray(arr[ids],dtype=np.float32)
def main():
 if (OUT/"analysis/final.json").exists(): raise RuntimeError("Phase1564 exists")
 pf=core.load(PARENT/"analysis/final.json"); pa=core.load(PARENT/"audit/independent_final_audit.json"); protocol=core.load(CONTRACT/"protocol/preregistration.json")
 if pf["authorization"]!="run_phase1564_c097_targeted_residual_adjudication" or not pa["all_checks_passed"]: raise RuntimeError("Phase1563 authorization missing")
 field=np.load(PARENT/"raw/c097a_all_role_field.float32.npy",mmap_mode="r"); index=core.rows(PARENT/"raw/c097a_field_index.jsonl"); pairs=core.rows(CONTRACT/"material/frozen_pairs.jsonl"); lookup={(r["pair_id"],r["query_family"]):r["row_index"] for r in index}; byfam={f:sorted([p for p in pairs if p["family"]==f],key=lambda p:(p["partition"],p["partition_rank"])) for f in protocol["families"]}
 individual=[]; rows=[]
 for rank,(a,b) in enumerate(zip(byfam["similarity"],byfam["class_inclusion"],strict=True)):
  aa=field[lookup[(a["pair_id"],"similarity")]]; bb=field[lookup[(b["pair_id"],"class_inclusion")]]; ab=field[lookup[(a["pair_id"],"class_inclusion")]]; ba=field[lookup[(b["pair_id"],"similarity")]]; individual.append(.5*(aa+bb-ab-ba)); rows.append({"row_index":rank,"pair_a":a["pair_id"],"pair_b":b["pair_id"],"split":a["partition"]})
 individual=np.asarray(individual,np.float32); raw=OUT/"raw"; raw.mkdir(parents=True,exist_ok=True); np.save(raw/"c097a_individual_interactions.float32.npy",individual); core.write_rows(raw/"c097a_individual_interactions_index.jsonl",rows)
 old1=old_stack(C095,"triadic_individual_interactions"); old2=old_stack(C096,"c096_triadic_individual_interactions"); rng=np.random.default_rng(1564); metrics=[]; bootrows=[]
 for state in FOCUS_STATES:
  new=individual[:,state,3].mean(0); one=old1[:,state,3].mean(0); two=old2[:,state,3].mean(0); split=cosine(individual[:7,state,3].mean(0),individual[7:,state,3].mean(0)); values1=[]; values2=[]
  for _ in range(2000):
   sample=individual[rng.integers(0,len(individual),len(individual)),state,3].mean(0); values1.append(cosine(sample,one)); values2.append(cosine(sample,two))
  row={"state":state,"new_vs_c091_pooled":cosine(new,one),"new_vs_c096_pooled":cosine(new,two),"split_half_cosine":split,"bootstrap_c091_q025":float(np.quantile(values1,.025)),"bootstrap_c091_median":float(np.median(values1)),"bootstrap_c096_q025":float(np.quantile(values2,.025)),"bootstrap_c096_median":float(np.median(values2))}; metrics.append(row); bootrows.extend({"state":state,"iteration":i,"c091_cosine":values1[i],"c096_cosine":values2[i]} for i in range(2000))
 core.write_rows(OUT/"analysis/targeted_metrics.jsonl",metrics); core.write_rows(OUT/"analysis/bootstrap_cosines.jsonl",bootrows)
 A1=min(min(r["new_vs_c091_pooled"],r["new_vs_c096_pooled"]) for r in metrics)>=.75; A2=min(min(r["bootstrap_c091_q025"],r["bootstrap_c096_q025"]) for r in metrics)>=.50; A3=min(r["split_half_cosine"] for r in metrics)>=.75
 decisions={"A1_pooled_cross_wave":{"passed":A1,"threshold":">=0.75","observed":min(min(r["new_vs_c091_pooled"],r["new_vs_c096_pooled"]) for r in metrics)},"A2_bootstrap_lower":{"passed":A2,"threshold":"2.5% >=0.50","observed":min(min(r["bootstrap_c091_q025"],r["bootstrap_c096_q025"]) for r in metrics)},"A3_split_half":{"passed":A3,"threshold":">=0.75","observed":min(r["split_half_cosine"] for r in metrics)}}
 status="pooled_stability_with_split_sensitivity" if A1 and A2 and not A3 else "pooled_and_split_stability" if all((A1,A2,A3)) else "persistent_targeted_residual_boundary"
 summary={"phase":1564,"campaign":"C097-A","status":status,"n_new_quartets":14,"n_c091_quartets":len(old1),"n_c096_quartets":len(old2),"metrics":metrics,"decisions":decisions,"interpretation":"A1/A2 support finite-small-cell fragility rather than a stable reversal only when both pass; A3 independently records split sensitivity. None proves a semantic mechanism.","files":{"individual_sha256":core.sha(raw/"c097a_individual_interactions.float32.npy"),"metrics_sha256":core.sha(OUT/"analysis/targeted_metrics.jsonl"),"bootstrap_sha256":core.sha(OUT/"analysis/bootstrap_cosines.jsonl")},"finished_at_utc":datetime.now(timezone.utc).isoformat(),"authorization":"run_phase1565_c097_wordnet_independent_contract"}; core.save(OUT/"analysis/c097a_adjudication.json",summary); core.save(OUT/"analysis/final.json",{"phase":1564,"campaign":"C097-A","status":status,"authorization":summary["authorization"]}); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__": main()

