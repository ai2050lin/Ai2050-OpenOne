#!/usr/bin/env python3
"""Independent audit for Phase1400."""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
OUT=TESTS/"result/phase1400_c064_fixed_answer_factorial_contract"
def main():
 p=core.load(OUT/"protocol/preregistration.json");pre=core.load(OUT/"audit/pre_model_semantic_naturalness_zero_model_audit.json")
 a=core.rows(OUT/"material/active_cases.jsonl");s=core.rows(OUT/"material/status_cases.jsonl");f=core.rows(OUT/"material/factor_sets.jsonl")
 checks={"preaudit":pre["all_checks_passed"] and pre["passed"]==pre["total"],
 "hashes":p["material"]["active_sha256"]==core.sha(OUT/"material/active_cases.jsonl") and p["material"]["status_sha256"]==core.sha(OUT/"material/status_cases.jsonl") and p["material"]["factor_sha256"]==core.sha(OUT/"material/factor_sets.jsonl"),
 "active_balance":len(a)==864 and Counter(x["truth"] for x in a)=={True:432,False:432},
 "status_balance":len(s)==288 and Counter(x["truth"] for x in s)=={True:144,False:144},
 "factor_balance":len(f)==144 and Counter(x["partition"] for x in f)=={q:48 for q in p["material"]["partitions"]},
 "scope":all(x in p["forbidden"] for x in ("attention","MLP","gradient","PCA","learned probe")),
 "authorization":p["authorization"]=="run_phase1401_c064_behavior","human_disclosed":not p["material"]["human_naturalness_lock"]}
 r={"phase":1400,"campaign":"C064","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())};core.save(OUT/"audit/independent_final_audit.json",r)
 if not r["all_checks_passed"]:raise RuntimeError({k:v for k,v in checks.items() if not v})
 print(json.dumps(r,indent=2))
if __name__=="__main__":main()
