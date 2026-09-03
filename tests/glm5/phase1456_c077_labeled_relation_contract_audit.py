#!/usr/bin/env python3
"""Independent audit for Phase1456 C077 contract."""
from __future__ import annotations
import json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core
PHASE,CAMPAIGN=1456,"C077"; OUT=TESTS/"result/phase1456_c077_labeled_relation_contract"
def main():
    active=core.rows(OUT/"material/active_cases.jsonl"); comp=core.rows(OUT/"material/composition_sets.jsonl"); compiled=core.rows(OUT/"compiled/qwen3_active.jsonl"); pre=core.load(OUT/"audit/pre_model_semantic_naturalness_zero_model_audit.json"); p=core.load(OUT/"protocol/preregistration.json"); final=core.load(OUT/"analysis/final.json")
    checks={"preaudit":pre["all_checks_passed"],"active":len(active)==3456,"truth":Counter(r["truth"] for r in active)=={True:1728,False:1728},"semantic":all(r["truth"]==(r["record_label"]==r["query_label"]) for r in active),"composition":len(comp)==216 and Counter(r["partition"] for r in comp)=={"response_discovery":72,"confirmation":72,"lockbox":72},"compiled":len(compiled)==3456 and all(len(r["role_positions"])==9 and all(len(v)==1 for v in r["role_positions"].values()) for r in compiled),"hashes":core.sha(OUT/"material/active_cases.jsonl")==p["material"]["active_sha256"] and core.sha(OUT/"material/composition_sets.jsonl")==p["material"]["composition_sha256"],"raw":p["discovery_capture"]["expected_case_count"]==1152 and p["discovery_capture"]["role_slot_count"]==9 and p["discovery_capture"]["no_holdout_access"],"scope":p["discovery_description"]["label_scope_only"] and "unlabeled natural relation mechanism" in p["claim_boundary"]["forbidden"],"forbidden":all(x in p["forbidden"] for x in ("attention","MLP","parameters","gradients","PCA","learned probe")),"hidden":pre["checks"]["hidden_not_accessed"],"authorization":final["authorization"]=="run_phase1457_c077_behavior" and final["contract_sha256"]==p["contract_sha256"]}
    result={"phase":PHASE,"campaign":CAMPAIGN,"checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}; core.save(OUT/"audit/independent_final_audit.json",result); print(json.dumps(result,indent=2));
    if not result["all_checks_passed"]: raise SystemExit(1)
if __name__=="__main__": main()
