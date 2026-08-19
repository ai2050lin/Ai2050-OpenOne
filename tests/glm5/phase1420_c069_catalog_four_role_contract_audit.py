#!/usr/bin/env python3
"""Independent audit for Phase1420."""
from __future__ import annotations
import json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";sys.path.insert(0,str(TESTS));import phase1331_relational_measurement_core as core
OUT=TESTS/"result/phase1420_c069_catalog_four_role_contract"
def main():
 p=core.load(OUT/"protocol/preregistration.json");f=core.load(OUT/"analysis/final.json");pre=core.load(OUT/"audit/pre_model_semantic_naturalness_zero_model_audit.json");a=core.rows(OUT/"material/active_cases.jsonl");c=core.rows(OUT/"material/composition_sets.jsonl");q=core.rows(OUT/"compiled/qwen3_active.jsonl");roles={"record_target","record_family","query_target","query_family"}
 checks={"preaudit":pre["all_checks_passed"],"active":len(a)==2880 and Counter(r["cell"] for r in a)=={x:360 for x in ("aa","ab","ac","ad","bb","ba","bc","bd")},"composition":len(c)==72 and Counter(r["partition"] for r in c)=={x:24 for x in p["material"]["partitions"]},"compiled":len(q)==2880 and all(all(len(r["role_positions"][x])==1 for x in roles) for r in q),"hashes":p["material"]["active_sha256"]==core.sha(OUT/"material/active_cases.jsonl") and p["material"]["composition_sha256"]==core.sha(OUT/"material/composition_sets.jsonl"),"catalog_scope":p["behavior"]["ordinary_is_required_set_control_not_family_gate"] and p["material"]["mechanism_surface"]=="catalog","mechanism_unchanged":p["mechanism"]["state_index"]==16 and len(p["mechanism"]["arms"])==9,"no_search":all(x in p["forbidden"] for x in ("layer search","subset search","candidate search")),"hidden_not_accessed":pre["checks"]["hidden_not_accessed"],"authorization":f["authorization"]=="run_phase1421_c069_catalog_behavior"}
 r={"phase":1420,"campaign":"C069","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())};core.save(OUT/"audit/independent_final_audit.json",r);print(json.dumps(r,indent=2))
 if not r["all_checks_passed"]:raise SystemExit(1)
if __name__=="__main__":main()
