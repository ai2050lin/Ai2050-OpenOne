#!/usr/bin/env python3
"""Independent audit for Phase1493."""
from __future__ import annotations
import json,py_compile,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; R=TESTS/"result"; OUT=R/"phase1493_c085_prospective_p084_adjudication"; A=R/"phase1492_c085_stratified_factorial_atlas"; C=R/"phase1489_c085_prospective_layered_contract"; sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
import phase1493_c085_prospective_p084_adjudication as phase
def main():
    saved=core.load(OUT/"analysis/p084_results.json"); final=core.load(OUT/"analysis/final.json"); protocol=core.load(C/"protocol/preregistration.json"); atlas=np.load(A/"atlas/success_factorial_contrast_mean.float32.npy",mmap_mode="r"); counts=np.load(A/"atlas/stratum_sample_counts.int32.npy"); recomputed,surface,trajectory=phase.evaluate(atlas,counts,protocol); py_compile.compile(str(TESTS/"phase1493_c085_prospective_p084_adjudication.py"),doraise=True)
    checks={"metrics":saved==recomputed,"gates":final["gates"]==recomputed["gates"],"joint":final["all_six_passed"]==all(final["gates"].values()),"freeze":final["prediction_freeze_sha256"]==protocol["p084"]["freeze_sha256"],"surface_rows":len(core.rows(OUT/"analysis/p084_surface_panels.jsonl"))==18,"trajectory_rows":len(core.rows(OUT/"analysis/p084_boundary_trajectory.jsonl"))==37,"authorization":final["authorization"]=="run_phase1494_c085_stratum_diagnostics"}
    result={"phase":1493,"campaign":"C085","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    core.save(OUT/"audit/independent_final_audit.json",result); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
