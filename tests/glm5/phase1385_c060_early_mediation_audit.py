#!/usr/bin/env python3
"""Independent audit for Phase1385."""
from __future__ import annotations

import json
import math
import py_compile
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1385_c060_early_mediation"
SCRIPT = TESTS / "phase1385_c060_early_mediation.py"


def main() -> None:
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    summary = core.load(OUT / "analysis/qwen3_early_mediation_summary.json")
    final = core.load(OUT / "analysis/final.json")
    records = core.rows(OUT / "raw/qwen3_early_mediation.jsonl")
    py_compile.compile(str(SCRIPT), doraise=True)
    py_compile.compile(__file__, doraise=True)
    checks = {
        "record_count": len(records) == manifest["case_count"] == summary["record_count"] == 144,
        "partitions": Counter(r["partition"] for r in records) == {"confirmation": 72, "lockbox": 72},
        "arms": all(set(r["effects"]) == set(manifest["arms"]) for r in records),
        "finite": summary["runtime"]["all_finite"] and all(
            r["all_finite"] and all(math.isfinite(v) for v in r["effects"].values()) for r in records
        ),
        "self": max(abs(r["effects"]["upstream_self"]) for r in records) <= 1e-4,
        "split_counts": summary["splits"]["pooled"]["count"] == 144 and summary["splits"]["confirmation"]["count"] == 72 and summary["splits"]["lockbox"]["count"] == 72,
        "qualification_consistent": summary["mediation_qualified"] == all(v["qualified"] for v in summary["splits"].values()),
        "final_consistent": final["mediation_qualified"] == summary["mediation_qualified"],
        "authorization": final["authorization"] == "run_phase1386_c060_campaign_closure",
        "scripts_compile": True,
    }
    audit = {
        "phase": 1385,
        "campaign": "C060",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
