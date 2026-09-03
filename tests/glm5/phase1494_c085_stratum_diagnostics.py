#!/usr/bin/env python3
"""Phase1494: compare success and mixed C085 relation fields; register failed as M2."""
from __future__ import annotations
import itertools,json,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; R=TESTS/"result"; A=R/"phase1492_c085_stratified_factorial_atlas"; P=R/"phase1493_c085_prospective_p084_adjudication"; B=R/"phase1490_c085_behavior_stratification"; C=R/"phase1489_c085_prospective_layered_contract"; OUT=R/"phase1494_c085_stratum_diagnostics"; sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
def cosine(a,b):
    d=float(np.linalg.norm(a)*np.linalg.norm(b)); return float(np.dot(a.astype(np.float64,copy=False),b.astype(np.float64,copy=False))/d) if d>1e-12 else 0.0
def compact(v): return {"minimum":float(min(v)),"mean":float(np.mean(v)),"maximum":float(max(v))} if v else None
def evaluate(field,counts,behavior,protocol):
    boundary=protocol["roles"].index("boundary"); valid=[(r,p,s) for r in range(6) for p in range(3) for s in range(2) if counts[1,r,p,s]>0]; rows=[]
    for r,p,s in valid:
        success=np.asarray(field[0,r,p,s,35,boundary],dtype=np.float32); mixed=np.asarray(field[1,r,p,s,35,boundary],dtype=np.float32)
        rows.append({"relation":protocol["relations"][r],"split":protocol["partitions"][p],"surface":protocol["surfaces"][s],"mixed_set_count":int(counts[1,r,p,s]),"success_mixed_cosine":cosine(success,mixed),"mixed_to_success_norm_ratio":float(np.linalg.norm(mixed)/np.linalg.norm(success))})
    trajectory=[]
    for state in range(37):
        values=[]
        for r,p,s in valid: values.append(cosine(np.asarray(field[0,r,p,s,state,boundary]),np.asarray(field[1,r,p,s,state,boundary])))
        trajectory.append({"state":state,"success_mixed_panel_cosine":compact(values)})
    relation_vectors=[]
    for r in range(6):
        panels=[np.asarray(field[1,r,p,s,35,boundary],dtype=np.float32) for p in range(3) for s in range(2) if counts[1,r,p,s]>0]
        if panels: relation_vectors.append((protocol["relations"][r],np.mean(panels,axis=0)))
    pair=[{"left":relation_vectors[i][0],"right":relation_vectors[j][0],"cosine":cosine(relation_vectors[i][1],relation_vectors[j][1])} for i,j in itertools.combinations(range(len(relation_vectors)),2)]
    errors=[r for r in behavior if not r["correct"]]; error_counts={"relation":dict(Counter(r["record_relation_id"] for r in errors)),"surface":dict(Counter(r["surface"] for r in errors)),"cell":dict(Counter(r["cell"] for r in errors)),"partition":dict(Counter(r["partition"] for r in errors)),"truth":dict(Counter(str(r["truth"]).lower() for r in errors))}
    summary={"valid_mixed_panels":len(valid),"mixed_relations":[x[0] for x in relation_vectors],"state35_success_mixed_cosine":compact([r["success_mixed_cosine"] for r in rows]),"state35_mixed_to_success_norm_ratio":compact([r["mixed_to_success_norm_ratio"] for r in rows]),"state35_mixed_relation_pairwise_cosine":compact([r["cosine"] for r in pair]),"first_state_with_mean_success_mixed_cosine_ge_0_90":next((r["state"] for r in trajectory if r["success_mixed_panel_cosine"]["mean"]>=.90),None),"error_count":len(errors),"error_counts":error_counts,"failed_stratum":"M2: absent; no comparison or zero-filled interpretation","interpretation":"descriptive diagnostic only; composition-set selection and unequal relation support prohibit a causal or failure-mechanism claim"}
    return summary,rows,trajectory,pair
def main():
    if (OUT/"analysis/final.json").exists(): raise RuntimeError("Phase1494 exists")
    pf=core.load(P/"analysis/final.json"); pa=core.load(P/"audit/independent_final_audit.json"); protocol=core.load(C/"protocol/preregistration.json")
    if pf["authorization"]!="run_phase1494_c085_stratum_diagnostics" or not pa["all_checks_passed"]: raise RuntimeError("Phase1493 authorization missing")
    field=np.load(A/"atlas/stratum_relation_effect.float32.npy",mmap_mode="r"); counts=np.load(A/"atlas/stratum_sample_counts.int32.npy"); behavior=core.rows(B/"raw/behavior.jsonl"); summary,panels,trajectory,pairs=evaluate(field,counts,behavior,protocol)
    core.save(OUT/"analysis/stratum_diagnostic_summary.json",summary); core.write_rows(OUT/"analysis/success_mixed_panel_comparison.jsonl",panels); core.write_rows(OUT/"analysis/success_mixed_boundary_trajectory.jsonl",trajectory); core.write_rows(OUT/"analysis/mixed_relation_pairwise.jsonl",pairs)
    checks={"mixed_panels":len(panels)==18,"mixed_relations":summary["mixed_relations"]==["support","visit","praise"],"errors":summary["error_count"]==57,"failed_m2":bool(np.all(counts[2]==0)),"finite":all(np.isfinite(r["success_mixed_cosine"]) and np.isfinite(r["mixed_to_success_norm_ratio"]) for r in panels)}
    if not all(checks.values()): raise RuntimeError(checks)
    final={"phase":1494,"campaign":"C085","status":"success_mixed_diagnostic_complete_failed_m2","checks":checks,"summary":summary,"claim_boundary":"diagnostic association only; no causal failure mechanism","finished_at_utc":datetime.now(timezone.utc).isoformat(),"authorization":"run_phase1495_c085_major_stage_closure"}; core.save(OUT/"analysis/final.json",final); print(json.dumps(final,indent=2))
if __name__=="__main__": main()
