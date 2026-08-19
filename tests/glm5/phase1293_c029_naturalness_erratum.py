#!/usr/bin/env python3
"""Phase 1293: independent naturalness erratum for frozen C029 material."""

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
UPSTREAM_AUDIT = UPSTREAM / "audit/independent_final_audit.json"
OUT = TEST_ROOT / "result/phase1293_c029_naturalness_erratum"
REPORT = OUT / "analysis/final.json"


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    protocol = read_json(PROTOCOL)
    audit = read_json(UPSTREAM_AUDIT)
    rows = read_jsonl(MATERIAL)
    issues: list[dict[str, str]] = []
    for row in rows:
        if row["surface"] != "catalog_prose":
            continue
        text = row["candidate_prompt"]
        for match in re.finditer(r"\ba ([aeiou][a-z-]*) (?:color|shape)\b", text, flags=re.IGNORECASE):
            issues.append({"case_id": row["case_id"], "kind": "indefinite_article_mismatch", "text": match.group(0)})
        if "stored in the rooftop" in text.lower():
            issues.append({"case_id": row["case_id"], "kind": "unnatural_location_collocation", "text": "stored in the rooftop"})

    counts = Counter(issue["kind"] for issue in issues)
    result = {
        "phase": 1293,
        "campaign": "C029",
        "schema_version": "phase1293.c029.naturalness_erratum.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_protocol_digest": protocol["protocol_digest"],
        "upstream_material_sha256": file_sha256(MATERIAL),
        "upstream_audit_had_passed": audit["all_checks_passed"],
        "model_weights_loaded": False,
        "contexts_reviewed": len(rows),
        "issue_count": len(issues),
        "issue_counts": dict(counts),
        "examples": issues[:24],
        "verdict": "pre_model_naturalness_gate_failed",
        "authorization": "close_c029_before_behavior",
        "scientific_scope": (
            "This invalidates the C029 material-naturalness authorization. It does not provide model behavior "
            "or mechanism evidence because no C029 model weights were loaded."
        ),
        "next_legal_step": (
            "A separately named campaign may retain the externally grounded lookup object only if it creates "
            "new material and freezes an article/collocation-aware independent audit before model loading."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "phase": 1293,
        "campaign": "C029",
        "issue_count": len(issues),
        "issue_counts": dict(counts),
        "authorization": result["authorization"],
    }, ensure_ascii=False, sort_keys=True))
    if not issues:
        raise SystemExit("erratum expected at least one independently detected naturalness defect")


if __name__ == "__main__":
    main()
