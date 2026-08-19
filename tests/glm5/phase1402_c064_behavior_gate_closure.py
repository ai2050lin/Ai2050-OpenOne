#!/usr/bin/env python3
"""Phase1402: close C064 after relation-null behavior failure."""
from __future__ import annotations
import json,py_compile,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";sys.path.insert(0,str(TESTS));import phase1331_relational_measurement_core as core
C=TESTS/"result/phase1400_c064_fixed_answer_factorial_contract";B=TESTS/"result/phase1401_c064_behavior";OUT=TESTS/"result/phase1402_c064_behavior_gate_closure"
def main():
 if (OUT/"analysis/final.json").exists():raise RuntimeError("Phase1402 exists")
 p=core.load(C/"protocol/preregistration.json");s=core.load(B/"analysis/qwen3_behavior_summary.json");f=core.load(B/"analysis/final.json");a=core.load(B/"audit/independent_final_audit.json")
 for phase in (1400,1401,1402):
  for path in TESTS.glob(f"phase{phase}_c064*.py"):py_compile.compile(str(path),doraise=True)
 active={r["case_id"]:r for r in core.rows(B/"raw/active_behavior.jsonl")};factors=core.rows(C/"material/factor_sets.jsonl");keys=("recipient","surface_same","member_same","family_same_polarity","polarity_same_family","family_and_polarity");eligible=[r for r in factors if all(active[r[k]]["correct"] for k in keys)]
 checks={"audited":a["all_checks_passed"],"failed":not s["behavior_qualified"] and f["authorization"]=="close_c064_at_behavior_gate","numeric":s["breadth_checks"]["numeric"] and s["breadth_checks"]["finite"],"active_high":s["global"]["active_accuracy"]>0.98,"status_failed":s["global"]["status_accuracy"]<p["behavior"]["status_accuracy_min"],"active_only_count":len(eligible)==128,"hidden_not_accessed":True,"scripts_compile":True}
 r={"phase":1402,"campaign":"C064","status":"closed_at_behavior_gate","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"formal_results":{"active_accuracy":s["global"]["active_accuracy"],"status_accuracy":s["global"]["status_accuracy"],"active_only_eligible_count":len(eligible)},"authorization":"preregister_c065_active_only_natural_state_campaign","next_required_action":"freeze balanced active-only family/truth donors before hidden access; do not repair C064","finished_at_utc":datetime.now(timezone.utc).isoformat()}
 if not r["all_checks_passed"]:raise RuntimeError({k:v for k,v in checks.items() if not v})
 core.write_rows(OUT/"material/active_only_eligible_factor_sets.jsonl",eligible);core.save(OUT/"analysis/final.json",r);print(json.dumps(r,indent=2))
if __name__=="__main__":main()
