#!/usr/bin/env python3
"""Independent arithmetic and provenance audit for C621."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase2156_c622_output_failure_taxonomy_audit"
C621 = ROOT / "tests/glm5/result/phase2155_c621_output_failure_taxonomy"
C617 = ROOT / "tests/glm5/result/phase2151_c617_generation_timeline_causal_boundary"


def load(path: Path): return json.loads(path.read_text(encoding="utf-8"))
def rows(path: Path): return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
def sha(path: Path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    save(OUT / "protocol/preregistration.json", {"phase": 2156, "campaign": "C622",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "object": "independent C621 provenance and arithmetic audit"})
    final = load(C621 / "analysis/final.json")
    taxonomy = rows(C621 / "analysis/failure_taxonomy.jsonl")
    source = rows(C617 / "analysis/generation_timeline_records.jsonl")
    checks = [
        {"name": "c621_closed", "passed": final["status"] == "closed" and final["all_checks_passed"], "detail": final["status"]},
        {"name": "record_arithmetic", "passed": len(taxonomy) == len(source) * 8 == 96,
         "detail": {"source": len(source), "taxonomy": len(taxonomy)}},
        {"name": "mode_balance", "passed": all(sum(x["mode"] == m for x in taxonomy) == 12 for m in
            ("zero", "q16", "q24", "q32", "joint", "wrong_sign", "wrong_role", "wrong_operation")), "detail": "12 per mode"},
        {"name": "descriptive_boundary", "passed": "no new causal gate" in load(C621 / "protocol/preregistration.json")["claim_boundary"],
         "detail": "post-registered taxonomy"},
    ]
    headline = {"status": "taxonomy_audit_closed", "checks_passed": sum(x["passed"] for x in checks),
        "checks_total": len(checks), "checks": checks,
        "manifest": [{"path": str((C621 / "analysis/final.json").relative_to(ROOT)), "sha256": sha(C621 / "analysis/final.json")},
                     {"path": str((C621 / "analysis/failure_taxonomy.jsonl").relative_to(ROOT)), "sha256": sha(C621 / "analysis/failure_taxonomy.jsonl")}],
        "adjudication": {"same_exact_goal_next_stage": False,
            "reason": "The timing-duration explanation is adjudicated for the frozen response. A next experiment needs a new output-identity state object or external human-reviewed corpus."}}
    result = {"phase": 2156, "campaign": "C622", "status": "closed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "all_checks_passed": all(x["passed"] for x in checks),
        "headline": headline, "checks": {"all_audit_checks": all(x["passed"] for x in checks)},
        "next_authorization": "major_stage_closed_new_object_required"}
    save(OUT / "analysis/final.json", result); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
