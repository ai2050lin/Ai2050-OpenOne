#!/usr/bin/env python3
"""Post-decision representation audit for the Phase1171 endpoint.

The frozen independent audit compared in-memory operation tuples with the JSON
list representation stored in score.json.  This script preserves that 32/33
audit and independently checks whether canonical JSON normalization makes the
recomputed endpoint exactly equal.  It cannot amend the primary result or
authorize continuation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1171_fixed_dimension_formation_trajectory_tomography as phase  # noqa: E402


SCRIPT = Path(__file__).resolve()


def normalized(value):
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def main() -> None:
    root = phase.OUT_ROOT
    original_audit_path = root / "audit/independent_audit.json"
    original_audit = phase.base.read_json(original_audit_path)
    score = phase.base.read_json(root / "analysis/score.json")
    final = phase.base.read_json(root / "analysis/final.json")
    holdout = phase.base.read_jsonl(root / "runs/holdout/holdout_metrics.jsonl")
    failed_checks = [key for key, value in original_audit["checks"].items() if not value]
    grouped: dict[str, list[dict]] = {}
    for row in holdout:
        grouped.setdefault(row["trajectory_id"], []).append(row)
    trajectories = [phase.trajectory_summary(rows) for rows in grouped.values()]
    recomputed = phase.endpoint_decision(trajectories)
    normalized_endpoint_match = all(normalized(recomputed[key]) == normalized(score[key]) for key in recomputed)
    report = {
        "phase": phase.PHASE,
        "audited_at_utc": phase.base.utc_now(),
        "script_sha256": phase.base.sha256_file(SCRIPT),
        "original_audit_sha256": phase.base.sha256_file(original_audit_path),
        "original_audit_passed": original_audit["passed"],
        "original_audit_total": original_audit["total"],
        "original_failed_checks": failed_checks,
        "normalized_endpoint_match": normalized_endpoint_match,
        "recomputed_global_regime_counts": recomputed["global_regime_counts"],
        "recomputed_mixed_task_count": recomputed["mixed_task_count"],
        "recomputed_primary_endpoint_pass": recomputed["primary_endpoint_pass"],
        "final_primary_endpoint_match": final["decision"]["primary_endpoint_pass"] == recomputed["primary_endpoint_pass"],
        "continuation_remains_denied": final["decision"]["auto_continue"] is False,
        "claim_scope": "Representation-normalization audit only; it does not rewrite the frozen 32/33 audit, thresholds, regimes, primary endpoint, or continuation branch.",
    }
    report["overall_pass"] = (
        report["original_audit_passed"] == 32
        and report["original_audit_total"] == 33
        and report["original_failed_checks"] == ["endpoint_recompute"]
        and report["normalized_endpoint_match"]
        and report["recomputed_global_regime_counts"] == {"delayed": 64, "direct_left_censored": 0, "nonstable": 0, "unfit": 0}
        and report["recomputed_mixed_task_count"] == 0
        and report["recomputed_primary_endpoint_pass"] is False
        and report["final_primary_endpoint_match"]
        and report["continuation_remains_denied"]
    )
    report["report_digest"] = phase.base.digest(report)
    phase.base.write_json(root / "audit/endpoint_representation_recompute.json", report)
    print(json.dumps({
        "overall_pass": report["overall_pass"],
        "normalized_endpoint_match": report["normalized_endpoint_match"],
        "global_regime_counts": report["recomputed_global_regime_counts"],
        "report_digest": report["report_digest"],
    }))
    if not report["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
