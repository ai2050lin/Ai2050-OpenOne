#!/usr/bin/env python3
from __future__ import annotations
import json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";sys.path.insert(0,str(TESTS));import phase1331_relational_measurement_core as core
C=TESTS/"result/phase1403_c065_active_only_natural_state_contract";OUT=TESTS/"result/phase1405_c065_natural_discovery_field"
def main():
 p=core.load(C/"protocol/preregistration.json");s=core.load(OUT/"analysis/field_summary.json");f=core.load(OUT/"analysis/final.json");r=core.rows(OUT/"raw/natural_full_field.jsonl");c=core.load(OUT/"protocol/frozen_natural_event_candidates.json")
 checks={"cases":s["case_count"]==18,"records":s["record_count"]==len(r),"states":set(x["state_index"] for x in r)==set(range(37)),"finite":all(math.isfinite(x[k]) for x in r for k in ("family_identity_score","joint_polarity_score")),"candidates":len(c["candidates"])==18 and all(x["role"]!="physical" for x in c["candidates"]),"hash":c["source_field_sha256"]==core.sha(OUT/"raw/natural_full_field.jsonl"),"decision":f["authorization"]==("run_phase1406_c065_holdout_factorial_swaps" if s["all_checks_passed"] else "close_c065_at_field_gate")}
 result={"phase":1405,"campaign":"C065","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())};core.save(OUT/"audit/independent_final_audit.json",result)
 if not result["all_checks_passed"]:raise RuntimeError(checks)
 print(json.dumps(result,indent=2))
if __name__=="__main__":main()
