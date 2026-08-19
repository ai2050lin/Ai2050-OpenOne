#!/usr/bin/env python3
"""Independent arithmetic audit for the Phase1246 descriptive atlas."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result/phase1246_c001_wp01_typed_behavior_qualification"
RAW = RESULT / "behavior/qwen3/raw_behavior.jsonl"
ATLAS = RESULT / "analysis/descriptive_failure_atlas.json"
OUT = RESULT / "audit/independent_failure_atlas_audit.json"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    raw = read_jsonl(RAW)
    atlas = read_json(ATLAS)
    candidate_errors = sum(not row["candidate_correct"] for row in raw)
    cache_rows = [row for row in raw if row["cache_full_recompute"] is not None]
    mismatch_steps = sum(
        row["cache_full_recompute"]["step_count"] - row["cache_full_recompute"]["match_count"]
        for row in cache_rows
    )
    mismatch_trajectories = sum(row["cache_full_recompute"]["agreement"] != 1.0 for row in cache_rows)
    expected_partition = {}
    for partition in ("calibration", "discovery", "selection", "confirmation"):
        rows = [row for row in raw if row["partition"] == partition]
        expected_partition[partition] = sum(row["candidate_correct"] for row in rows) / len(rows)
    checks = {
        "atlas_digest": atlas["atlas_digest"] == digest({k: v for k, v in atlas.items() if k != "atlas_digest"}),
        "raw_digest": atlas["source_raw_digest"] == digest(raw),
        "candidate_error_count": atlas["candidate_selection"]["error_count"] == candidate_errors,
        "candidate_partition_accuracy": atlas["candidate_selection"]["partition_accuracy"] == expected_partition,
        "cache_trajectory_count": atlas["cache_recompute"]["trajectory_count"] == len(cache_rows),
        "cache_mismatch_steps": atlas["cache_recompute"]["mismatch_step_count"] == mismatch_steps,
        "cache_mismatch_trajectories": atlas["cache_recompute"]["mismatch_trajectory_count"] == mismatch_trajectories,
        "generation_category_totals": all(sum(counts.values()) == len(raw) for counts in atlas["generation"]["category_counts"].values()),
        "non_authorizing_boundary": len(atlas["interpretation_boundary"]) >= 3,
    }
    value: dict[str, Any] = {
        "phase": 1246,
        "schema_version": "phase1246.independent_failure_atlas_audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "check_count": len(checks),
        "all_checks_passed": all(checks.values()),
        "atlas_digest": atlas["atlas_digest"],
    }
    value["audit_digest"] = digest(value)
    OUT.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(canonical_json({"status": "phase1246_failure_atlas_audit", "passed": value["all_checks_passed"], "checks": len(checks), "digest": value["audit_digest"]}))
    if not value["all_checks_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
