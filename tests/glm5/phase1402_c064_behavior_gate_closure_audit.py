#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";sys.path.insert(0,str(TESTS));import phase1331_relational_measurement_core as core
OUT=TESTS/"result/phase1402_c064_behavior_gate_closure"
def main():
 f=core.load(OUT/"analysis/final.json");e=core.rows(OUT/"material/active_only_eligible_factor_sets.jsonl");checks={"closed":f["status"]=="closed_at_behavior_gate","checks":f["all_checks_passed"],"eligible":len(e)==128,"new_contract":f["authorization"]=="preregister_c065_active_only_natural_state_campaign"}
 r={"phase":1402,"campaign":"C064","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())};core.save(OUT/"audit/independent_final_audit.json",r)
 if not r["all_checks_passed"]:raise RuntimeError(checks)
 print(json.dumps(r,indent=2))
if __name__=="__main__":main()
