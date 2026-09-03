#!/usr/bin/env python3
"""Phase1493: one-shot prospective adjudication of frozen P084-1 through P084-6."""
from __future__ import annotations
import itertools,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; RESULT=TESTS/"result"; C=RESULT/"phase1489_c085_prospective_layered_contract"; A=RESULT/"phase1492_c085_stratified_factorial_atlas"; OUT=RESULT/"phase1493_c085_prospective_p084_adjudication"; sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
EFFECTS=["relation","entity","object","relation_entity","relation_object","entity_object","relation_entity_object"]
ORDERS={"relation":1,"entity":1,"object":1,"relation_entity":2,"relation_object":2,"entity_object":2,"relation_entity_object":3}
def cosine(a,b):
    d=float(np.linalg.norm(a)*np.linalg.norm(b)); return float(np.dot(a.astype(np.float64,copy=False),b.astype(np.float64,copy=False))/d) if d>1e-12 else 0.0
def pairwise(v): return [cosine(v[i],v[j]) for i,j in itertools.combinations(range(len(v)),2)]
def compact(v): return {"minimum":float(min(v)),"mean":float(np.mean(v)),"maximum":float(max(v))}
def top_mask(v,count):
    out=np.zeros(v.shape[-1],dtype=bool); out[np.argsort(-np.square(v.astype(np.float64,copy=False)),kind="stable")[:count]]=True; return out
def evaluate(atlas,counts,protocol):
    roles=protocol["roles"]; boundary=roles.index("boundary"); query_label=roles.index("query_label"); relation_mean=np.mean(atlas[0],axis=(1,2),dtype=np.float64).astype(np.float32)
    early=relation_mean[:,0,query_label]; early_pair=pairwise(early); early_ratio=float(np.linalg.norm(np.mean(early,axis=0))/np.mean(np.linalg.norm(early,axis=1)))
    late=relation_mean[:,35,boundary]; late_pair=pairwise(late); late_loo=[cosine(late[r],np.mean(np.delete(late,r,axis=0),axis=0)) for r in range(6)]; panel_loo=[]
    for r in range(6):
        panels=np.asarray(atlas[0,r,:,:,35,boundary],dtype=np.float32).reshape(6,2560)
        panel_loo.extend(cosine(panels[p],np.mean(np.delete(panels,p,axis=0),axis=0)) for p in range(6))
    support=[]
    for fraction,count in zip((0.005,0.01,0.02,0.05),(13,26,51,128)):
        masks=np.asarray([top_mask(late[r],count) for r in range(6)]); intersection=np.all(masks,axis=0); coords=np.flatnonzero(intersection); sign=np.sign(late[:,coords]); unanimous=bool(len(coords)>0 and np.all(np.all(sign==sign[0],axis=0) & (sign[0]!=0)))
        support.append({"fraction":fraction,"count_per_relation":count,"intersection_count":int(len(coords)),"same_sign_all_intersection_coordinates":unanimous})
    energy=[]; nuisance=[]
    for r in range(6):
        pooled=np.mean(atlas[:,r,:,:,35,boundary],axis=(1,2),dtype=np.float64); norms=np.linalg.norm(pooled,axis=1); beta=np.asarray([norms[i]/(2**ORDERS[e]) for i,e in enumerate(EFFECTS)]); frac=np.square(beta)/np.sum(np.square(beta)); ratios=beta[1:]/beta[0]
        energy.append(float(frac[0])); nuisance.append(float(np.max(ratios)))
    surface=[]
    for r,relation in enumerate(protocol["relations"]):
        for p,split in enumerate(protocol["partitions"]): surface.append({"relation":relation,"split":split,"cosine":cosine(np.asarray(atlas[0,r,p,0,35,boundary]),np.asarray(atlas[0,r,p,1,35,boundary]))})
    trajectory=[]
    for state in range(37):
        vectors=relation_mean[:,state,boundary]; pw=pairwise(vectors); loo=[cosine(vectors[r],np.mean(np.delete(vectors,r,axis=0),axis=0)) for r in range(6)]; trajectory.append({"state":state,"pairwise_minimum":float(min(pw)),"leave_one_relation_minimum":float(min(loo))})
    stable=[r["state"] for r in trajectory if r["pairwise_minimum"]>=.90 and r["leave_one_relation_minimum"]>=.95]; first=min(stable) if stable else None; evaluable=bool(np.all(counts[0]>=protocol["p084"]["minimum_sets_per_relation_split_surface"]))
    gates={
        "P084-1":evaluable and -.23<=float(np.mean(early_pair))<=-.17 and early_ratio<=.05,
        "P084-2":evaluable and min(late_pair)>=.90 and min(late_loo)>=.95 and min(panel_loo)>=.95,
        "P084-3":evaluable and all(row["intersection_count"]>=minimum and row["same_sign_all_intersection_coordinates"] for row,minimum in zip(support,(6,12,25,60))),
        "P084-4":evaluable and min(energy)>=.95 and max(nuisance)<=.15,
        "P084-5":evaluable and min(row["cosine"] for row in surface)>=.90,
        "P084-6":evaluable and first is not None and 20<=first<=26,
    }
    metrics={"evaluable":evaluable,"success_panel_minimum_set_count":int(np.min(counts[0])),"P084-1":{"pairwise":compact(early_pair),"centroid_norm_ratio":early_ratio},"P084-2":{"pairwise":compact(late_pair),"leave_one_relation":compact(late_loo),"leave_one_panel":compact(panel_loo)},"P084-3":{"supports":support},"P084-4":{"relation_coefficient_energy_fraction":compact(energy),"maximum_nonrelation_coefficient_norm_ratio":compact(nuisance)},"P084-5":{"cross_surface_cosine":compact([r["cosine"] for r in surface])},"P084-6":{"first_stable_state":first},"gates":gates,"all_six_passed":all(gates.values())}
    return metrics,surface,trajectory
def main():
    if (OUT/"analysis/final.json").exists(): raise RuntimeError("Phase1493 exists")
    final=core.load(A/"analysis/final.json"); audit=core.load(A/"audit/independent_final_audit.json"); protocol=core.load(C/"protocol/preregistration.json"); summary=core.load(A/"analysis/stratified_atlas_summary.json")
    if final["authorization"]!="run_phase1493_c085_prospective_p084_adjudication" or not audit["all_checks_passed"]: raise RuntimeError("Phase1492 authorization missing")
    atlas=np.load(A/"atlas/success_factorial_contrast_mean.float32.npy",mmap_mode="r"); counts=np.load(A/"atlas/stratum_sample_counts.int32.npy"); metrics,surface,trajectory=evaluate(atlas,counts,protocol)
    core.save(OUT/"analysis/p084_results.json",metrics); core.write_rows(OUT/"analysis/p084_surface_panels.jsonl",surface); core.write_rows(OUT/"analysis/p084_boundary_trajectory.jsonl",trajectory)
    verdict="prospectively_confirmed_in_fresh_controlled_task" if metrics["all_six_passed"] else "prospective_joint_gate_failed_with_component_results_retained"
    result={"phase":1493,"campaign":"C085","prediction_freeze_sha256":protocol["p084"]["freeze_sha256"],"verdict":verdict,"evaluable":metrics["evaluable"],"gates":metrics["gates"],"all_six_passed":metrics["all_six_passed"],"claim_boundary":"one Qwen3 explicit-label controlled task; no natural semantics, causality, cross-model law, or new mathematics","finished_at_utc":datetime.now(timezone.utc).isoformat(),"authorization":"run_phase1494_c085_stratum_diagnostics"}
    core.save(OUT/"analysis/final.json",result); print(json.dumps({"final":result,"metrics":metrics},indent=2))
if __name__=="__main__": main()
