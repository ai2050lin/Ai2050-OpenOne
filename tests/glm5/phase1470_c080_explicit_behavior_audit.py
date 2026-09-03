#!/usr/bin/env python3
"""Independent audit for Phase1470."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
CONTRACT = TESTS / "result/phase1469_c080_balanced_interaction_contract"
OUT = TESTS / "result/phase1470_c080_explicit_behavior"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1470_c080_explicit_behavior as phase


def main() -> None:
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    saved = core.load(OUT / "analysis/behavior_summary.json")
    rows = core.rows(OUT / "raw/explicit_behavior.jsonl")
    sets = core.rows(CONTRACT / "material/explicit_interaction_sets.jsonl")
    eligible_saved = core.rows(OUT / "material/eligible_interaction_sets.jsonl")
    recomputed, eligible = phase.evaluate(rows, sets, protocol, saved["numeric_repeat_max_abs_diff"], saved["runtime"]["quantization"])
    py_compile.compile(str(TESTS / "phase1470_c080_explicit_behavior.py"), doraise=True)
    checks = {
        "count": len(rows) == 10368,
        "case_ids": len({row["case_id"] for row in rows}) == len(rows),
        "predictions": all(row["prediction"] == max(range(2), key=lambda index: row["scores"][index]) for row in rows),
        "correct": all(row["correct"] == (row["prediction"] == row["gold_position"]) for row in rows),
        "metrics": all(abs(saved[key] - recomputed[key]) < 1e-12 for key in ("global_accuracy", "global_balanced_accuracy")),
        "submetrics": all(saved[key] == recomputed[key] for key in ("surface", "nuisance_surface", "equal_label_surface", "unequal_pair_surface")),
        "checks": saved["checks"] == recomputed["checks"],
        "eligible": [row["set_id"] for row in eligible_saved] == [row["set_id"] for row in eligible],
        "qualification": final["behavior_qualified"] == saved["behavior_qualified"] == all(saved["checks"].values()),
        "authorization": final["authorization"] == ("run_phase1471_c080_explicit_discovery_capture" if saved["behavior_qualified"] else "close_c080_explicit_at_behavior_gate"),
        "numeric": saved["runtime"]["quantization"]["has_bf16_parameters"] and not saved["runtime"]["quantization"]["has_quantized_modules"],
        "hidden_not_accessed": saved["hidden_state_accessed"] is False,
    }
    result = {"phase": 1470, "campaign": "C080", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
