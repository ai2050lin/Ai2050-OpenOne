#!/usr/bin/env python3
"""Independent closure audit for Phase1495."""
from __future__ import annotations
import json,py_compile,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; R=TESTS/"result"; OUT=R/"phase1495_c085_major_stage_closure"; sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
def main():
    f=core.load(OUT/"analysis/final.json"); p=core.load(R/"phase1493_c085_prospective_p084_adjudication/analysis/p084_results.json"); b=core.load(R/"phase1490_c085_behavior_stratification/analysis/behavior_stratification_summary.json"); d=core.load(R/"phase1494_c085_stratum_diagnostics/analysis/stratum_diagnostic_summary.json"); py_compile.compile(str(TESTS/"phase1495_c085_major_stage_closure.py"),doraise=True)
    checks={"phase_checks":all(f["checks"].values()),"p084":p["all_six_passed"] and f["status"].endswith("prospectively_confirmed"),"behavior":b["stratum_counts"]=={"success":175,"mixed":41},"diagnostic":d["valid_mixed_panels"]==18 and d["error_count"]==57,"scope":"no causal" in f["claim_scope"],"puzzle":f["core_puzzle"]["id"]=="K262","authorization":f["authorization"]=="preregister_c086_label_carrier_withdrawal_layered_observation"}
    result={"phase":1495,"campaign":"C085","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    core.save(OUT/"audit/independent_final_audit.json",result); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
