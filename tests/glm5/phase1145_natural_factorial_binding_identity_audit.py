#!/usr/bin/env python3
"""Independent record audit for Phase1145."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import phase1145_natural_factorial_binding_identity as phase


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = phase.OUT_ROOT
PRIMARY = ROOT / "tests/glm5/phase1145_natural_factorial_binding_identity.py"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def close(left: Any, right: Any, tolerance: float = 2e-7) -> bool:
    if left is None or right is None:
        return left is right
    return abs(float(left) - float(right)) <= tolerance


def audit_run(split: str, route: str, model: str, protocol_digest: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = OUT_ROOT / "runs" / split / route / model
    records = read_jsonl(root / "records.jsonl")
    stored_curves = read_jsonl(root / "curves.jsonl")
    stored_comparisons = read_jsonl(root / "comparisons.jsonl")
    summary = read_json(root / "summary.json")
    curves = phase.curves_from_records(records)
    comparisons = phase.comparisons_from_curves(curves)
    metrics, _, _ = phase.analyze_route(records)
    metric_names = [key for key, value in metrics.items() if isinstance(value, (int, float)) or value is None]
    mismatches = [{"metric": key, "stored": summary["metrics"].get(key), "recomputed": metrics[key]} for key in metric_names if not close(summary["metrics"].get(key), metrics[key])]
    named = {
        "protocol_digest": summary["protocol_digest"] == protocol_digest,
        "record_count": len(records) == phase.EXPECTED_RECORDS == summary["record_count"],
        "curve_count": len(curves) == len(stored_curves) == phase.EXPECTED_COMPARISONS * 2,
        "comparison_count": len(comparisons) == len(stored_comparisons) == phase.EXPECTED_COMPARISONS,
        "record_digest": digest(records) == summary["record_digest"],
        "curve_digest": digest(curves) == summary["metrics"]["curve_digest"],
        "comparison_digest": digest(comparisons) == summary["metrics"]["comparison_digest"],
        "metrics_recomputed": not mismatches,
        "all_records_finite": all(bool(row["finite"]) for row in records),
        "alpha0_exact": metrics["paired_alpha0_max_abs_margin_difference"] <= phase.THRESHOLDS["paired_alpha0_max_abs_margin_difference"],
    }
    return ([{"split": split, "route": route, "model": model, "name": name, "passed": bool(value)} for name, value in named.items()], {"split": split, "route": route, "model": model, "named_checks": named, "mismatches": mismatches})


def main() -> None:
    prereg = read_json(OUT_ROOT / "protocol/preregistration.json")
    body = dict(prereg)
    stored = body.pop("protocol_digest")
    checks = [
        {"name": "protocol_digest", "passed": digest(body) == stored},
        {"name": "primary_hash", "passed": sha256_file(PRIMARY) == prereg["source"]["script_sha256"]},
        {"name": "protocol_audit", "passed": bool(read_json(OUT_ROOT / "protocol/audit.json")["all_checks_passed"])},
    ]
    details = []
    for split in phase.SPLITS:
        for route in phase.ROUTES:
            for model in phase.MODELS:
                root = OUT_ROOT / "runs" / split / route / model
                if not root.exists():
                    continue
                more, detail = audit_run(split, route, model, stored)
                checks.extend(more)
                details.append(detail)
    final = read_json(OUT_ROOT / "analysis/final.json")
    checks.extend([
        {"name": "component_scope_matches_final", "passed": bool(final["component_search_authorized"]) == bool(final["confirmed"])},
        {"name": "semantic_claim_denied", "passed": not bool(final["semantic_payload_claim_authorized"])},
    ])
    audit = {"phase": phase.PHASE, "created_at_utc": datetime.now(timezone.utc).isoformat(), "check_count": len(checks), "passed_count": sum(bool(row["passed"]) for row in checks), "all_checks_passed": all(bool(row["passed"]) for row in checks), "checks": checks, "details": details, "protocol_digest": stored, "final_digest": final["final_digest"]}
    audit["audit_digest"] = digest(audit)
    write_json(OUT_ROOT / "audit/independent_result_audit.json", audit)
    print(canonical(audit))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
