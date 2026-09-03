#!/usr/bin/env python3
"""Independent C134 route-A audit."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";OUT=TESTS/"result/phase1668_c134_route_a_directed_composition";sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
import phase1668_c134_route_a_directed_composition as c134

def contract_audit():
    p=core.load(OUT/"protocol/preregistration.json");cases=core.rows(OUT/"material/cases.jsonl");compiled=core.rows(OUT/"compiled/qwen3.jsonl")
    checks={"internal":core.load(OUT/"audit/internal_contract_audit.json")["all_checks_passed"],"counts":len(cases)==len(compiled)==1024,"routes":set(row["route_type"] for row in cases)==set((*c134.ROUTE_TYPES,"edge_factorial")),"roles":all(set(row["role_positions"])==set(c134.ROLES) for row in compiled),"source_hashes":all(core.sha(Path(p["source_paths"][n]))==d for n,d in p["source_hashes"].items()),"confirmation_policy":"discovery code reads only discovery" in p["confirmation_policy"],"boundary":"not a universal" in p["claim_boundary"]}
    r={"phase":1668,"campaign":"C134","stage":"contract","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"authorization":"run_c134_behavior" if all(checks.values()) else "stop"};core.save(OUT/"audit/independent_contract_audit.json",r);print(json.dumps(r,indent=2))

def final_audit():
    b=core.load(OUT/"analysis/behavior.json");cl=core.load(OUT/"analysis/closure.json");checks={"contract":core.load(OUT/"audit/independent_contract_audit.json")["all_checks_passed"],"behavior_integrity":core.load(OUT/"audit/internal_behavior_audit.json")["all_integrity_checks_passed"],"closure":core.load(OUT/"audit/internal_closure_audit.json")["all_checks_passed"]}
    if b["gate_passed"]:
        raw=np.load(OUT/"raw/qwen3_role_typed_checkpoints.bf16.npy",mmap_mode="r");freeze=core.load(OUT/"protocol/frozen_predictions.json");confirm=core.load(OUT/"analysis/confirmation.json");checks.update({"raw_shape":list(raw.shape)==[1024,5,38,2560],"capture":core.load(OUT/"audit/internal_capture_audit.json")["all_checks_passed"],"freeze":freeze["confirmation_unread"] and freeze["authorization"]=="validate_c134_confirmation","confirmation":core.load(OUT/"audit/internal_confirmation_audit.json")["all_checks_passed"],"six_predictions":len(confirm["route_predictions"])==6})
    else:checks["no_hiddenstate"]=not (OUT/"raw/qwen3_role_typed_checkpoints.bf16.npy").exists()
    r={"phase":1668,"campaign":"C134","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"scientific_gate_passed":core.load(OUT/"audit/internal_closure_audit.json")["scientific_gate_passed"],"authorization":"start_route_B_C135" if all(checks.values()) else "stop"};core.save(OUT/"audit/independent_closure_audit.json",r);print(json.dumps(r,indent=2))

if __name__=="__main__":
    if len(sys.argv)!=2 or sys.argv[1] not in ("contract","final"):raise SystemExit("contract|final")
    {"contract":contract_audit,"final":final_audit}[sys.argv[1]]()
