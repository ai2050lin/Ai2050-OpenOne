#!/usr/bin/env python3
"""Independent audit for C136."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";OUT=TESTS/"result/phase1670_c136_chinese_pattern_composition_field";sys.path.insert(0,str(TESTS));import phase1331_relational_measurement_core as core
def contract():
 p=core.load(OUT/"protocol/preregistration.json");rows=core.rows(OUT/"compiled/qwen3.jsonl");checks={"internal":core.load(OUT/"audit/internal_contract_audit.json")["all_checks_passed"],"rows":len(rows)==512,"tasks":len(set(r["task"] for r in rows))==8,"roles":all(len(r["role_positions"])==5 for r in rows),"hashes":all(core.sha(Path(v))==p["source_hashes"][k] for k,v in p["source_paths"].items()),"boundary":"machine audit only" in p["naturalness_boundary"]};r={"checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"authorization":"run_c136_behavior"};core.save(OUT/"audit/independent_contract_audit.json",r);print(json.dumps(r,indent=2))
def final():
 b=core.load(OUT/"analysis/behavior.json");checks={"contract":core.load(OUT/"audit/independent_contract_audit.json")["all_checks_passed"],"behavior":core.load(OUT/"audit/internal_behavior_audit.json")["all_checks_passed"],"closure":core.load(OUT/"audit/internal_closure_audit.json")["all_checks_passed"]}
 if b["gate_passed"]:
  raw=np.load(OUT/"raw/qwen3_pattern_role_field.bf16.npy",mmap_mode="r");checks.update({"raw_shape":list(raw.shape)==[512,5,38,2560],"capture":core.load(OUT/"audit/internal_capture_audit.json")["all_checks_passed"],"freeze":core.load(OUT/"protocol/frozen_patterns.json")["confirmation_unread"],"confirmation":core.load(OUT/"audit/internal_confirmation_audit.json")["all_checks_passed"]})
 else:checks["no_hiddenstate"]=not (OUT/"raw/qwen3_pattern_role_field.bf16.npy").exists()
 r={"checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"scientific_gate_passed":core.load(OUT/"audit/internal_closure_audit.json")["scientific_gate_passed"],"authorization":"start_route_D_C137"};core.save(OUT/"audit/independent_closure_audit.json",r);print(json.dumps(r,indent=2))
if __name__=="__main__":{"contract":contract,"final":final}[sys.argv[1]]()
