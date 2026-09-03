#!/usr/bin/env python3
"""Independent audit for Phase1530."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1530_c090_canonical_full_recapture"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    summary = core.load(OUT / "analysis/canonical_behavior_and_capture_summary.json")
    rows = core.rows(OUT / "raw/canonical_all_role_field_index.jsonl")
    groups = core.rows(OUT / "material/canonical_stratified_relation_sets.jsonl")
    field = np.load(OUT / "raw/canonical_all_role_field.float16.npy", mmap_mode="r")
    py_compile.compile(str(TESTS / "phase1530_c090_canonical_full_recapture.py"), doraise=True)
    checks = {
        "status": final["status"] == "canonical_behavior_and_full_state_recapture_complete",
        "shape": list(field.shape) == [360, 37, 4, 2560] and bool(np.isfinite(np.asarray(field)).all()),
        "index": len(rows) == 360 and all(row["row_index"] == i for i, row in enumerate(rows)),
        "groups": len(groups) == 45 and sum(row["case_count"] for row in groups) == 360,
        "hashes": core.sha(OUT / "raw/canonical_all_role_field.float16.npy") == summary["files"]["field"]["sha256"] and core.sha(OUT / "raw/canonical_all_role_field_index.jsonl") == summary["files"]["index"]["sha256"],
        "repeat": summary["repeat_hidden_max_abs"] == 0.0 and summary["repeat_logit_max_abs"] == 0.0,
        "causal": summary["source_causal_prefix_max_abs"] == 0.0,
        "behavior": sum(row["correct"] for row in rows) / len(rows) == summary["global_accuracy"],
        "checks": all(summary["checks"].values()),
        "authorization": final["authorization"] == "run_phase1531_c090_canonical_truth_contrast_atlas",
    }
    result = {"phase": 1530, "campaign": "C090", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
