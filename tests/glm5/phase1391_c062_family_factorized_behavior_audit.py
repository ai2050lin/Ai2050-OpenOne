#!/usr/bin/env python3
"""Independent audit for Phase1391."""
from pathlib import Path
import json, sys
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5";sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
OUT=TESTS/"result/phase1391_c062_family_factorized_behavior"


def main():
    s=core.load(OUT/"analysis/qwen3_family_behavior_summary.json"); f=core.load(OUT/"analysis/final.json")
    rows=core.rows(OUT/"material/eligible_pairs.jsonl")
    expected="run_phase1392_c062_full_field_camera" if s["behavior_qualified"] else "close_c062_at_factorized_behavior_breadth_gate"
    checks={
        "family_flags": all(v["qualified"]==all(v["checks"].values()) for v in s["family_results"].values()),
        "breadth_flag": s["behavior_qualified"]==all(s["breadth_checks"].values()),
        "authorization": f["authorization"]==expected,
        "selected_exact": len(rows)==s["selected_count"],
        "selected_only_qualified": set(r["target_family"] for r in rows)<=set(s["qualified_families"]),
        "per_family_selected": all(sum(r["target_family"]==fam for r in rows)==54 for fam in s["qualified_families"]),
        "per_partition_selected": all(v==18*len(s["qualified_families"]) for v in s["selected_partition_counts"].values()),
        "bf16": s["runtime"]["quantization"]["has_bf16_parameters"],
        "not_quantized": not s["runtime"]["quantization"]["has_quantized_modules"],
    }
    result={"phase":1391,"checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}
    core.save(OUT/"audit/independent_final_audit.json",result);print(json.dumps(result,indent=2))
    if not result["all_checks_passed"]:raise SystemExit(1)


if __name__=="__main__":main()
