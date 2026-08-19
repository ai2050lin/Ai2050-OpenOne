#!/usr/bin/env python3
"""Independent audit for Phase1396."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
OUT=TESTS/"result/phase1396_c062_campaign_closure"
def main():
 f=core.load(OUT/"analysis/final.json")
 checks={"closed":f["status"]=="closed_after_all_frozen_eligible_routes","internal":f["all_checks_passed"] and f["passed"]==f["total"],
         "audits":all(v["all_checks_passed"] for v in f["phase_audits"].values()),"positive_scoped":len(f["claim_boundary"]["supported"])==4,
         "negative_scoped":len(f["claim_boundary"]["not_supported"])==5,"no_forbidden":not f["forbidden_hits"],
         "no_auto_unregistered":not f["automatic_next_phase"],"next_new_contract":"C063 contract" in f["next_required_action"]}
 result={"phase":1396,"checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}
 core.save(OUT/"audit/independent_final_audit.json",result);print(json.dumps(result,indent=2))
 if not result["all_checks_passed"]:raise SystemExit(1)
if __name__=="__main__":main()
