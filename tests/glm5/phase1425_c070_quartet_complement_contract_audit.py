#!/usr/bin/env python3
"""Independent audit for Phase1425."""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core
OUT=TESTS/"result/phase1425_c070_quartet_complement_contract"


def main():
    p=core.load(OUT/"protocol/preregistration.json"); f=core.load(OUT/"analysis/final.json"); pre=core.load(OUT/"audit/pre_model_semantic_naturalness_zero_model_audit.json")
    a=core.rows(OUT/"material/active_cases.jsonl"); c=core.rows(OUT/"material/composition_sets.jsonl"); q=core.rows(OUT/"compiled/qwen3_active.jsonl"); roles=("record_target","record_family","query_target","query_family")
    checks={
        "preaudit":pre["all_checks_passed"], "active":len(a)==1440 and Counter(r["cell"] for r in a)=={x:180 for x in ("aa","ab","ac","ad","bb","ba","bc","bd")},
        "composition":len(c)==72 and Counter(r["partition"] for r in c)=={x:24 for x in p["material"]["partitions"]},
        "same_shape":len({len(r["prompt_ids"]) for r in q})==1 and len({tuple((x,tuple(r["role_positions"][x])) for x in roles) for r in q})==1,
        "quartet_singleton":all(all(len(r["role_positions"][x])==1 for x in roles) for r in q),
        "hashes":p["material"]["active_sha256"]==core.sha(OUT/"material/active_cases.jsonl") and p["material"]["composition_sha256"]==core.sha(OUT/"material/composition_sets.jsonl"),
        "all_family_cells":all(sum(r["record_family"]==family and r["cell"]==cell for r in a)==30 for family in p["material"]["families"] for cell in ("aa","ab","ac","ad","bb","ba","bc","bd")),
        "partition_object":p["research_object"]=="state16 quartet-versus-complement causal support partition" and p["mechanism"]["state_index"]==16,
        "five_arms":p["mechanism"]["arms"]==["self","quartet_only","complement_only","full_state","wrong_full_state"],
        "forbidden":all(x in p["forbidden"] for x in ("attention","MLP","gradients","PCA","learned probe","layer search")),
        "hidden_not_accessed":pre["checks"]["hidden_not_accessed"], "authorization":f["authorization"]=="run_phase1426_c070_behavior",
    }
    result={"phase":1425,"campaign":"C070","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}; core.save(OUT/"audit/independent_final_audit.json",result); print(json.dumps(result,indent=2))
    if not result["all_checks_passed"]: raise SystemExit(1)


if __name__=="__main__":main()
