#!/usr/bin/env python3
"""Independent audit for Phase1388."""
from pathlib import Path
import json, math, sys
ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"; sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
OUT = TESTS / "result/phase1388_c061_qwen_behavior_qualification"


def main() -> None:
    summary = core.load(OUT / "analysis/qwen3_behavior_summary.json")
    final = core.load(OUT / "analysis/final.json")
    eligible = core.rows(OUT / "material/eligible_pairs.jsonl")
    expected_authorization = ("run_phase1389_c061_full_field_camera" if summary["behavior_qualified"]
                              else "close_c061_behavior_unqualified_before_hidden_access")
    checks = {
        "behavior_flag_exact": summary["behavior_qualified"] == all(summary["checks"].values()),
        "authorization": final["authorization"] == expected_authorization,
        "active_count": summary["active"]["count"] == 1728,
        "status_count": summary["status"]["count"] == 576,
        "selected_count_exact": len(eligible) == summary["selected_pair_count"],
        "failure_closed_before_hidden": summary["behavior_qualified"] or len(eligible) == 0,
        "failed_gate_present": summary["behavior_qualified"] or not all(summary["checks"].values()),
        "finite": math.isfinite(summary["numeric_same_shape_max_abs_diff"]),
        "bf16": summary["runtime"]["quantization"]["has_bf16_parameters"],
        "not_quantized": not summary["runtime"]["quantization"]["has_quantized_modules"],
    }
    result = {"phase": 1388, "checks": checks, "passed": sum(checks.values()), "total": len(checks),
              "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
