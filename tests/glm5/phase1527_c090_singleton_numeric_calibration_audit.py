#!/usr/bin/env python3
"""Independent audit for Phase1527."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1527_c090_singleton_numeric_calibration"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    summary = core.load(OUT / "analysis/singleton_numeric_calibration.json")
    pairs = core.rows(OUT / "analysis/singleton_causal_prefix_pairs.jsonl")
    comparisons = core.rows(OUT / "analysis/left_batch_vs_singleton.jsonl")
    field = np.load(OUT / "raw/singleton_calibration_field.float32.npy", mmap_mode="r")
    py_compile.compile(str(TESTS / "phase1527_c090_singleton_numeric_calibration.py"), doraise=True)
    checks = {
        "status": final["status"] == "singleton_engine_failed_causal_prefix_gate",
        "shape": list(field.shape) == [72, 37, 4, 2560] and bool(np.isfinite(np.asarray(field)).all()),
        "hash": core.sha(OUT / "raw/singleton_calibration_field.float32.npy") == summary["files"]["field"]["sha256"],
        "pairs": len(pairs) == 36 and max(row["max_relative_l2"] for row in pairs) == summary["singleton_causal_prefix_max_relative_l2"],
        "comparisons": len(comparisons) == 72 and max(row["field_max_abs"] for row in comparisons) == summary["left_batch_vs_singleton"]["field_max_abs"],
        "repeat": summary["singleton_repeat_hidden_max_abs"] <= summary["thresholds"]["singleton_repeat_max_abs"],
        "causal_failure": summary["singleton_causal_prefix_max_relative_l2"] > summary["thresholds"]["singleton_causal_prefix_relative_l2"],
        "canonical_failure": not summary["canonical_singleton_engine_pass"],
        "recorded_checks": summary["checks"]["repeat"] and not summary["checks"]["causal_prefix"] and not summary["checks"]["canonical"],
        "authorization": final["authorization"] == "preregister_phase1528_c090_right_padded_group_calibration",
    }
    result = {"phase": 1527, "campaign": "C090", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
