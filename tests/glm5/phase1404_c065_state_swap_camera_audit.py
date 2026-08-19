#!/usr/bin/env python3
from __future__ import annotations
import json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";sys.path.insert(0,str(TESTS));import phase1331_relational_measurement_core as core
C=TESTS/"result/phase1403_c065_active_only_natural_state_contract";OUT=TESTS/"result/phase1404_c065_state_swap_camera"
def main():
 p=core.load(C/"protocol/preregistration.json");s=core.load(OUT/"analysis/camera_summary.json");f=core.load(OUT/"analysis/final.json");kt=core.rows(OUT/"raw/known_truth_systems.jsonl");q=core.rows(OUT/"raw/qwen_identity_camera.jsonl");checks={"known":len(kt)==256 and all(all(r[k] for k in ("zero_exact","self_reset_exact","role_swap_exact","shape_exact")) for r in kt),"qwen":len(q)==18 and all(r["state_count"]==37 for r in q),"logits":max(r["output_max_abs_diff"] for r in q)<=p["camera"]["logit_identity_max_abs_diff"],"states":max(r["all_state_relative_l2_max"] for r in q)<=p["camera"]["all_state_identity_relative_l2_max"],"finite":all(math.isfinite(r["all_state_relative_l2_max"]) for r in q),"decision":f["authorization"]==("run_phase1405_c065_natural_discovery_field" if s["camera_qualified"] else "close_c065_at_camera_gate")}
 r={"phase":1404,"campaign":"C065","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())};core.save(OUT/"audit/independent_final_audit.json",r)
 if not r["all_checks_passed"]:raise RuntimeError(checks)
 print(json.dumps(r,indent=2))
if __name__=="__main__":main()
