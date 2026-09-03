#!/usr/bin/env python3
"""Independent audit for Phase1489."""
from __future__ import annotations
import json, py_compile, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; OUT=TESTS/"result/phase1489_c085_prospective_layered_contract"; sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
def main():
    p=core.load(OUT/"protocol/preregistration.json"); f=core.load(OUT/"analysis/final.json"); a=core.load(OUT/"audit/pre_model_semantic_naturalness_zero_model_audit.json")
    py_compile.compile(str(TESTS/"phase1489_c085_prospective_layered_contract.py"),doraise=True)
    checks={"preaudit":a["all_checks_passed"],"active_hash":core.sha(OUT/"material/active_cases.jsonl")==p["material"]["active_sha256"],"compiled_hash":core.sha(OUT/"compiled/qwen3_active.jsonl")==p["material"]["compiled_sha256"],"composition_hash":core.sha(OUT/"material/composition_sets.jsonl")==p["material"]["composition_sha256"],"contract_hash":core.digest({k:v for k,v in p.items() if k not in ("contract_sha256","authorization")})==p["contract_sha256"],"prediction_hash":p["p084"]["freeze_sha256"]==f["prediction_freeze_sha256"],"strata":set(p["behavior_strata"])>={"success","mixed","failed"},"observables":p["allowed_observables"]==["input embeddings","all full-dimensional Hidden States","yes/no logits"],"authorization":f["authorization"]=="run_phase1490_c085_behavior_stratification"}
    result={"phase":1489,"campaign":"C085","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    core.save(OUT/"audit/independent_final_audit.json",result); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
