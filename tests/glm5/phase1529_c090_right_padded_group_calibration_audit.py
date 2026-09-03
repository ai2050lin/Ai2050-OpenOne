#!/usr/bin/env python3
"""Independent audit for Phase1529."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1529_c090_right_padded_group_calibration"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    summary = core.load(OUT / "analysis/right_padded_group_calibration.json")
    pairs = core.rows(OUT / "analysis/right_padded_causal_prefix_pairs.jsonl")
    field = np.load(OUT / "raw/right_padded_calibration_field.float32.npy", mmap_mode="r")
    py_compile.compile(str(TESTS / "phase1529_c090_right_padded_group_calibration.py"), doraise=True)
    checks = {
        "status": final["status"] == "right_padded_group_engine_calibrated",
        "shape": list(field.shape) == [72, 37, 4, 2560] and bool(np.isfinite(np.asarray(field)).all()),
        "hash": core.sha(OUT / "raw/right_padded_calibration_field.float32.npy") == summary["files"]["field"]["sha256"],
        "pairs": len(pairs) == 36 and max(row["max_relative_l2"] for row in pairs) == summary["causal_prefix_max_relative_l2"],
        "repeat": summary["repeat_hidden_max_abs"] <= summary["gates"]["repeat_hidden_max_abs"] and summary["repeat_logit_max_abs"] <= summary["gates"]["repeat_logit_max_abs"],
        "causal": summary["causal_prefix_max_relative_l2"] <= summary["gates"]["causal_prefix_relative_l2"],
        "canonical": summary["canonical_right_padded_engine_pass"] and all(summary["checks"].values()),
        "authorization": final["authorization"] == "run_phase1530_c090_canonical_full_recapture",
    }
    result = {"phase": 1529, "campaign": "C090", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
