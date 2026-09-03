#!/usr/bin/env python3
"""Independent audit for Phase1494."""
from __future__ import annotations
import json,py_compile,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; R=TESTS/"result"; OUT=R/"phase1494_c085_stratum_diagnostics"; A=R/"phase1492_c085_stratified_factorial_atlas"; B=R/"phase1490_c085_behavior_stratification"; C=R/"phase1489_c085_prospective_layered_contract"; sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
import phase1494_c085_stratum_diagnostics as phase
def main():
    field=np.load(A/"atlas/stratum_relation_effect.float32.npy",mmap_mode="r"); counts=np.load(A/"atlas/stratum_sample_counts.int32.npy"); behavior=core.rows(B/"raw/behavior.jsonl"); protocol=core.load(C/"protocol/preregistration.json"); recomputed,panels,traj,pairs=phase.evaluate(field,counts,behavior,protocol); saved=core.load(OUT/"analysis/stratum_diagnostic_summary.json"); final=core.load(OUT/"analysis/final.json"); py_compile.compile(str(TESTS/"phase1494_c085_stratum_diagnostics.py"),doraise=True)
    checks={"summary":saved==recomputed,"panels":core.rows(OUT/"analysis/success_mixed_panel_comparison.jsonl")==panels,"trajectory":core.rows(OUT/"analysis/success_mixed_boundary_trajectory.jsonl")==traj,"pairs":core.rows(OUT/"analysis/mixed_relation_pairwise.jsonl")==pairs,"phase_checks":all(final["checks"].values()),"authorization":final["authorization"]=="run_phase1495_c085_major_stage_closure"}
    result={"phase":1494,"campaign":"C085","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    core.save(OUT/"audit/independent_final_audit.json",result); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
