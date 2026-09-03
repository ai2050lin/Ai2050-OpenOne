#!/usr/bin/env python3
"""Independent audit for Phase1480."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
CONTRACT = TESTS / "result/phase1479_c083_fresh_validation_contract"
OUT = TESTS / "result/phase1480_c083_behavior"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1480_c083_behavior as phase


def main() -> None:
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    saved = core.load(OUT / "analysis/behavior_summary.json")
    final = core.load(OUT / "analysis/final.json")
    rows = core.rows(OUT / "raw/behavior.jsonl")
    composition = core.rows(CONTRACT / "material/composition_sets.jsonl")
    eligible_saved = core.rows(OUT / "material/eligible_composition_sets.jsonl")
    recomputed, eligible = phase.evaluate(rows, composition, protocol, saved["numeric_repeat_max_abs_diff"], saved["runtime"]["quantization"])
    py_compile.compile(str(TESTS / "phase1480_c083_behavior.py"), doraise=True)
    checks = {
        "count": len(rows) == 3456,
        "predictions": all(row["prediction"] == max(range(2), key=lambda index: row["scores"][index]) for row in rows),
        "correct": all(row["correct"] == (row["prediction"] == row["gold_position"]) for row in rows),
        "metrics": all(saved[key] == recomputed[key] for key in ("surface", "relation_surface", "checks")),
        "eligible": [row["set_id"] for row in eligible_saved] == [row["set_id"] for row in eligible],
        "qualification": final["behavior_qualified"] == saved["behavior_qualified"] == all(saved["checks"].values()),
        "authorization": final["authorization"] == ("run_phase1481_c083_discovery_capture" if saved["behavior_qualified"] else "close_c083_at_behavior_gate"),
        "numeric": saved["checks"]["repeat"] and saved["checks"]["finite"] and saved["checks"]["bf16"] and saved["checks"]["not_quantized"],
        "hidden_not_accessed": not saved["hidden_state_accessed"],
    }
    result = {"phase": 1480, "campaign": "C083", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
