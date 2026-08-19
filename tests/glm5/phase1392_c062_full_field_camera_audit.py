#!/usr/bin/env python3
"""Independent audit for Phase1392."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
OUT=TESTS/"result/phase1392_c062_full_field_camera"
def main():
 s=core.load(OUT/"analysis/camera_summary.json");f=core.load(OUT/"analysis/final.json");kt=core.rows(OUT/"raw/known_truth_systems.jsonl");q=core.rows(OUT/"raw/qwen_identity_camera.jsonl")
 expected="run_phase1393_c062_discovery_full_field" if s["camera_qualified"] else "close_c062_at_camera_gate"
 checks={"flag":s["camera_qualified"]==all(s["checks"].values()),"authorization":f["authorization"]==expected,
         "known_count":len(kt)==256,"qwen_count":len(q)==24,"all_states":all(r["state_count"]==37 for r in q),
         "zero_exact":s["qwen_output_max_abs_diff"]==0.0 and s["qwen_all_state_relative_l2_max"]==0.0,
         "scope":True}
 result={"phase":1392,"checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}
 core.save(OUT/"audit/independent_final_audit.json",result);print(json.dumps(result,indent=2))
 if not result["all_checks_passed"]:raise SystemExit(1)
if __name__=="__main__":main()
