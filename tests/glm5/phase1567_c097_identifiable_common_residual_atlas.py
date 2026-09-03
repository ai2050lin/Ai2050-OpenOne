#!/usr/bin/env python3
"""Phase1567: identify the WordNet contrast mean/residual atlas with baselines."""
from __future__ import annotations
import json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; RESULT=TESTS/"result"; CONTRACT=RESULT/"phase1565_c097_wordnet_independent_contract"; PARENT=RESULT/"phase1566_c097_wordnet_capture"; C096=RESULT/"phase1559_c096_fresh_prediction_atlas_and_adjudication"; OUT=RESULT/"phase1567_c097_identifiable_common_residual_atlas"
sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core
from phase1561_c097_common import FAMILIES,FAMILY_PAIRS,SURFACES,ROLES,FOCUS_STATES,cosine,top_indices,decompose_contrasts
PARTITIONS=("response_discovery","confirmation","lockbox")
def bootstrap_common(stacks,rng,n=2000):
 values=[]
 for start in range(0,n,100):
  size=min(100,n-start); ids=rng.integers(0,stacks[0].shape[0],(size,stacks[0].shape[0])); means=np.stack([stack[ids].mean(axis=1) for stack in stacks],axis=1); common=means.mean(axis=1); total=np.sum(means*means,axis=(1,2)); common_energy=3*np.sum(common*common,axis=1); values.extend(np.divide(common_energy,total,out=np.zeros_like(common_energy),where=total>0).tolist())
 return values
def main():
 if (OUT/"analysis/final.json").exists(): raise RuntimeError("Phase1567 exists")
 pf=core.load(PARENT/"analysis/final.json"); pa=core.load(PARENT/"audit/independent_final_audit.json"); protocol=core.load(CONTRACT/"protocol/preregistration.json")
 if pf["authorization"]!="run_phase1567_c097_identifiable_common_residual_atlas" or not pa["all_checks_passed"]: raise RuntimeError("Phase1566 authorization missing")
 field=np.load(PARENT/"raw/c097b_all_role_field.float32.npy",mmap_mode="r"); index=core.rows(PARENT/"raw/c097b_field_index.jsonl"); pairs=core.rows(CONTRACT/"material/frozen_wordnet_pairs.jsonl"); lookup={(r["pair_id"],r["surface"],r["query_family"]):r["row_index"] for r in index}
 raw=OUT/"raw"; raw.mkdir(parents=True,exist_ok=True); individual=np.lib.format.open_memmap(raw/"c097b_individual_interactions.float32.npy",mode="w+",dtype=np.float32,shape=(180,37,4,2560)); individual_index=[]; groups={}; cursor=0
 for pi,partition in enumerate(PARTITIONS):
  for si,surface in enumerate(SURFACES):
   family_rows={f:sorted([p for p in pairs if p["partition"]==partition and p["family"]==f],key=lambda p:(p["source_set_id"],p["pair_id"])) for f in FAMILIES}
   for fi,(fa,fb) in enumerate(FAMILY_PAIRS):
    ids=[]
    for rank,(a,b) in enumerate(zip(family_rows[fa],family_rows[fb],strict=True)):
     aa=field[lookup[(a["pair_id"],surface,fa)]]; bb=field[lookup[(b["pair_id"],surface,fb)]]; ab=field[lookup[(a["pair_id"],surface,fb)]]; ba=field[lookup[(b["pair_id"],surface,fa)]]; individual[cursor]=.5*(aa+bb-ab-ba); ids.append(cursor); individual_index.append({"row_index":cursor,"partition":partition,"surface":surface,"family_a":fa,"family_b":fb,"rank":rank,"pair_a":a["pair_id"],"pair_b":b["pair_id"]}); cursor+=1
    groups[(pi,si,fi)]=ids
 individual.flush(); core.write_rows(raw/"c097b_individual_interactions_index.jsonl",individual_index)
 common=np.lib.format.open_memmap(raw/"c097b_common_contrast_field.float32.npy",mode="w+",dtype=np.float32,shape=(3,2,37,4,2560)); residual=np.lib.format.open_memmap(raw/"c097b_residual_contrast_field.float32.npy",mode="w+",dtype=np.float32,shape=(3,2,3,37,4,2560)); atlas=[]; boot=[]; rng=np.random.default_rng(1567)
 for pi,partition in enumerate(PARTITIONS):
  for si,surface in enumerate(SURFACES):
   centroids=[np.asarray(individual[groups[(pi,si,fi)]],np.float32).mean(0) for fi in range(3)]; g,rs,_=decompose_contrasts(centroids); common[pi,si]=g.astype(np.float32)
   for fi,r in enumerate(rs): residual[pi,si,fi]=r.astype(np.float32)
   for state in range(37):
    for ri,role in enumerate(ROLES):
     vectors=[v[state,ri] for v in centroids]; _,_,d=decompose_contrasts(vectors); cosines=[cosine(vectors[0],vectors[1]),cosine(vectors[0],vectors[2]),cosine(vectors[1],vectors[2])]; atlas.append({"partition":partition,"surface":surface,"state":state,"role":role,"common_fraction":d["common_fraction"],"common_energy":d["common_energy"],"residual_energy":d["residual_energy"],"energy_identity_error":d["energy_identity_error"],"residual_sum_max_abs":d["residual_sum_max_abs"],"minimum_pairwise_cosine":min(cosines),"median_pairwise_cosine":float(np.median(cosines))})
   for state in FOCUS_STATES:
    stacks=[np.asarray(individual[groups[(pi,si,fi)],state,3],np.float32) for fi in range(3)]; values=bootstrap_common(stacks,rng); boot.append({"partition":partition,"surface":surface,"state":state,"n":len(values),"q025":float(np.quantile(values,.025)),"median":float(np.median(values)),"q975":float(np.quantile(values,.975))})
 common.flush(); residual.flush(); core.write_rows(OUT/"analysis/common_residual_atlas.jsonl",atlas); core.write_rows(OUT/"analysis/common_fraction_bootstrap.jsonl",boot)
 cross_partition=[]
 for si,surface in enumerate(SURFACES):
  for state in FOCUS_STATES:
   for left,right in ((0,1),(0,2),(1,2)): cross_partition.append({"surface":surface,"state":state,"left":PARTITIONS[left],"right":PARTITIONS[right],"cosine":cosine(common[left,si,state,3],common[right,si,state,3])})
 core.write_rows(OUT/"analysis/common_cross_partition.jsonl",cross_partition)
 chinese=np.load(C096/"raw/c096_triadic_interaction_centroids.float32.npy",mmap_mode="r"); cross_language=[]
 for pi,partition in enumerate(PARTITIONS):
  for si,surface in enumerate(SURFACES):
   for ci,concreteness in enumerate(("concrete","abstract")):
    for state in FOCUS_STATES:
     cg=np.asarray(chinese[pi,si,ci,:,state,3],np.float64).mean(0); cross_language.append({"partition":partition,"surface":surface,"concreteness":concreteness,"state":state,"cosine":cosine(common[pi,si,state,3],cg)})
 core.write_rows(OUT/"analysis/english_chinese_common_cosines.jsonl",cross_language)
 coordinate=[]; perm_rng=np.random.default_rng(15670)
 for si,surface in enumerate(SURFACES):
  for state in FOCUS_STATES:
   reference=np.asarray(common[0,si,state,3],np.float64); idx=top_indices(reference,64)
   for target_pi in (1,2):
    target=np.asarray(common[target_pi,si,state,3],np.float64); actual_cos=cosine(reference[idx],target[idx]); actual_sign=float(np.mean(np.sign(reference[idx])==np.sign(target[idx]))); null_cos=[]; null_sign=[]
    for _ in range(1000):
     perm=target[perm_rng.permutation(target.size)]; null_cos.append(cosine(reference[idx],perm[idx])); null_sign.append(float(np.mean(np.sign(reference[idx])==np.sign(perm[idx]))))
    coordinate.append({"surface":surface,"state":state,"target_partition":PARTITIONS[target_pi],"actual_restricted_cosine":actual_cos,"actual_sign_agreement":actual_sign,"permutation_cosine_q99":float(np.quantile(null_cos,.99)),"permutation_sign_q99":float(np.quantile(null_sign,.99)),"beats_both":bool(actual_cos>np.quantile(null_cos,.99) and actual_sign>np.quantile(null_sign,.99))})
 core.write_rows(OUT/"analysis/top64_coordinate_permutation_baseline.jsonl",coordinate)
 focus=[r for r in atlas if r["role"]=="boundary" and r["state"] in FOCUS_STATES]; B1=float(np.median([r["common_fraction"] for r in focus])); B2=min(r["cosine"] for r in cross_partition); B3=float(np.median([r["cosine"] for r in cross_language])); B4=all(r["beats_both"] for r in coordinate)
 decisions={"B1_common_fraction":{"passed":B1>=.5,"threshold":"median >=0.50","observed":B1},"B2_cross_partition_common":{"passed":B2>=.5,"threshold":"minimum >=0.50","observed":B2},"B3_cross_language_common":{"passed":B3>=.5,"threshold":"median >=0.50","observed":B3},"B4_coordinate_alignment":{"passed":B4,"threshold":"all actual top64 cosine and sign > permutation q99","observed":{"passed_cells":sum(r["beats_both"] for r in coordinate),"total_cells":len(coordinate)}}}
 behavior=core.load(PARENT/"analysis/capture_summary.json")["family_behavior"]; behavior_type={f:("behavior_qualified" if v["balanced_accuracy"]>=.8 and v["true_recall"]>=.75 and v["false_recall"]>=.75 else "M_BEHAVIOR") for f,v in behavior.items()}
 summary={"phase":1567,"campaign":"C097-B","status":"identifiable_common_residual_atlas_revealed","decisions":decisions,"passed_decisions":sum(v["passed"] for v in decisions.values()),"total_decisions":4,"behavior_type":behavior_type,"focus_common_fraction":{"min":min(r["common_fraction"] for r in focus),"median":B1,"max":max(r["common_fraction"] for r in focus),"bootstrap_q025_min":min(r["q025"] for r in boot)},"cross_partition_common":{"min":B2,"median":float(np.median([r["cosine"] for r in cross_partition]))},"cross_language_common":{"min":min(r["cosine"] for r in cross_language),"median":B3,"max":max(r["cosine"] for r in cross_language)},"coordinate_baseline":{"passed":sum(r["beats_both"] for r in coordinate),"total":len(coordinate)},"exactness":{"max_energy_identity_error":max(r["energy_identity_error"] for r in atlas),"max_residual_sum_abs":max(r["residual_sum_max_abs"] for r in atlas)},"claim":"A dependent three-contrast mean is identifiable and may repeat as geometry; it is not a purified semantic comparator, an independent triple replication, or a causal object.","files":{"individual":core.sha(raw/"c097b_individual_interactions.float32.npy"),"common":core.sha(raw/"c097b_common_contrast_field.float32.npy"),"residual":core.sha(raw/"c097b_residual_contrast_field.float32.npy"),"atlas":core.sha(OUT/"analysis/common_residual_atlas.jsonl")},"finished_at_utc":datetime.now(timezone.utc).isoformat(),"authorization":"run_phase1568_c097_major_stage_closure_and_visualization_decision"}; core.save(OUT/"analysis/c097b_adjudication.json",summary); core.save(OUT/"analysis/final.json",{"phase":1567,"campaign":"C097-B","status":summary["status"],"authorization":summary["authorization"]}); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
