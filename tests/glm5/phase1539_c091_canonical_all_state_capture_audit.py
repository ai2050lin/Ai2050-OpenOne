#!/usr/bin/env python3
"""Independent audit for Phase1539."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1539_c091_canonical_all_state_capture"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    report = core.load(OUT / "analysis/canonical_capture_summary.json")
    field_path = OUT / "raw/canonical_all_role_field.float16.npy"
    index_path = OUT / "raw/canonical_all_role_field_index.jsonl"
    field = np.load(field_path, mmap_mode="r")
    index = core.rows(index_path)
    checks = {
        "field_hash": core.sha(field_path) == report["files"]["field"]["sha256"],
        "field_bytes": field_path.stat().st_size == report["files"]["field"]["bytes"],
        "index_hash": core.sha(index_path) == report["files"]["index"]["sha256"],
        "shape": list(field.shape) == [540, 37, 4, 2560],
        "index": len(index) == 540 and sorted(row["row_index"] for row in index) == list(range(540)),
        "numeric_checks": all(report["checks"][key] for key in ("repeat_hidden", "repeat_logits", "behavior_logit_replay", "postquery_causal_identity", "prequery_causal_identity")),
        "qualified_scope": report["qualified_families"] == ["whole_part"],
        "finite_sample": bool(np.isfinite(np.asarray(field[::17])).all()),
        "authorization": core.load(OUT / "analysis/final.json")["authorization"] == "run_phase1540_c091_discovery_timing_atlas",
    }
    result = {
        "phase": 1539,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "checks": checks,
    }
    core.save(OUT / "audit/independent_final_audit.json", result)
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
