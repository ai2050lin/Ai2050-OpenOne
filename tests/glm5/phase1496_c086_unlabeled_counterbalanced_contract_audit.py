#!/usr/bin/env python3
"""Independent audit for Phase1496."""
from __future__ import annotations
import json,py_compile,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; OUT=TESTS/"result/phase1496_c086_unlabeled_counterbalanced_contract"; sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
def main():
 p=core.load(OUT/"protocol/preregistration.json"); f=core.load(OUT/"analysis/final.json"); a=core.load(OUT/"audit/pre_model_semantic_naturalness_zero_model_audit.json"); rows=core.rows(OUT/"material/active_cases.jsonl"); py_compile.compile(str(TESTS/"phase1496_c086_unlabeled_counterbalanced_contract.py"),doraise=True)
 checks={"preaudit":a["all_checks_passed"],"counts":len(rows)==6912 and len(core.rows(OUT/"material/composition_sets.jsonl"))==216,"hashes":core.sha(OUT/"material/active_cases.jsonl")==p["material"]["active_sha256"] and core.sha(OUT/"compiled/qwen3_active.jsonl")==p["material"]["compiled_sha256"] and core.sha(OUT/"material/composition_sets.jsonl")==p["material"]["composition_sha256"],"contract":core.digest({k:v for k,v in p.items() if k not in ("contract_sha256","authorization")})==p["contract_sha256"],"counterbalance":all(r["output_yes"]==(r["relation_match"]==(r["code_sign"]==1)) for r in rows),"label_absence":all("relation label" not in r["prompt"].lower() for r in rows),"observables":p["allowed_observables"]==["input embeddings","all full-dimensional Hidden States","yes/no logits"],"authorization":f["authorization"]=="run_phase1497_c086_behavior_stratification"}
 result={"phase":1496,"campaign":"C086","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}
 if not result["all_checks_passed"]: raise RuntimeError(checks)
 core.save(OUT/"audit/independent_final_audit.json",result); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
