#!/usr/bin/env python3
"""Independent result audit for the Phase 1293 C029 erratum."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
UPSTREAM = TEST_ROOT / "result/phase1292_c029_object_attribute_convergence_contract"
MATERIAL = UPSTREAM / "material/frozen_object_attribute_cases.jsonl"
PROTOCOL = UPSTREAM / "protocol/preregistration.json"
REPORT = TEST_ROOT / "result/phase1293_c029_naturalness_erratum/analysis/final.json"
AUDIT = TEST_ROOT / "result/phase1293_c029_naturalness_erratum/audit/independent_final_audit.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    protocol = load(PROTOCOL)
    report = load(REPORT)
    rows = [json.loads(line) for line in MATERIAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts: Counter[str] = Counter()
    for row in rows:
        if row["surface"] != "catalog_prose":
            continue
        text = row["candidate_prompt"]
        counts["indefinite_article_mismatch"] += len(re.findall(
            r"\ba ([aeiou][a-z-]*) (?:color|shape)\b", text, flags=re.IGNORECASE
        ))
        counts["unnatural_location_collocation"] += int("stored in the rooftop" in text.lower())
    counts = Counter({key: value for key, value in counts.items() if value})
    checks = {
        "upstream_pinned": report["upstream_protocol_digest"] == protocol["protocol_digest"] and report["upstream_material_sha256"] == sha(MATERIAL),
        "counts_recompute": report["issue_counts"] == dict(counts) and report["issue_count"] == sum(counts.values()),
        "defects_nonzero": sum(counts.values()) > 0,
        "no_model_weights": report["model_weights_loaded"] is False,
        "closure_exact": report["authorization"] == "close_c029_before_behavior",
        "scope_bounded": "does not provide model behavior" in report["scientific_scope"],
    }
    result = {
        "phase": 1293,
        "campaign": "C029",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "auditor_imports_main": False,
        "checks": checks,
        "passed_count": sum(checks.values()),
        "total_count": len(checks),
        "all_checks_passed": all(checks.values()),
        "authorization": "close_c029_before_behavior" if all(checks.values()) else "none",
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"phase": 1293, "passed": result["passed_count"], "total": result["total_count"], "authorization": result["authorization"]}, sort_keys=True))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
