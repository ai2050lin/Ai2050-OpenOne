#!/usr/bin/env python3
"""Independent audit for Phase1567."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; OUT=TESTS/"result/phase1567_c097_identifiable_common_residual_atlas"; sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core
from phase1561_c097_common import decompose_contrasts
def main():
 s=core.load(OUT/"analysis/c097b_adjudication.json"); i=np.load(OUT/"raw/c097b_individual_interactions.float32.npy",mmap_mode="r"); g=np.load(OUT/"raw/c097b_common_contrast_field.float32.npy",mmap_mode="r"); r=np.load(OUT/"raw/c097b_residual_contrast_field.float32.npy",mmap_mode="r"); f=core.load(OUT/"analysis/final.json"); _,_,d=decompose_contrasts([g[0,0,31,3]+r[0,0,k,31,3] for k in range(3)])
 checks={"individual_shape":list(i.shape)==[180,37,4,2560],"common_shape":list(g.shape)==[3,2,37,4,2560],"residual_shape":list(r.shape)==[3,2,3,37,4,2560],"finite":bool(np.isfinite(g).all() and np.isfinite(r).all()),"zero_sum":d["residual_sum_max_abs"]<1e-5,"energy_identity":d["energy_identity_error"]<1e-4,"decisions":len(s["decisions"])==4,"bootstrap":len(core.rows(OUT/"analysis/common_fraction_bootstrap.jsonl"))==12,"permutation":len(core.rows(OUT/"analysis/top64_coordinate_permutation_baseline.jsonl"))==8,"hash":s["files"]["common"]==core.sha(OUT/"raw/c097b_common_contrast_field.float32.npy"),"authorization":f["authorization"]=="run_phase1568_c097_major_stage_closure_and_visualization_decision"}; result={"phase":1567,"checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}; core.save(OUT/"audit/independent_final_audit.json",result)
 if not result["all_checks_passed"]: raise RuntimeError(result)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
