#!/usr/bin/env python3
"""Independent audit for C210."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import phase1739_c205_response_ecology_common as common

core = common.core
OUT = common.C210


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = final["headline"]
    checks = {"final": final["all_checks_passed"], "pairs": len(core.rows(OUT / "analysis/pair_index.jsonl")) == 36, "semantic_filter": report["eligible_pairs"] + report["descriptive_pairs"] == 36, "models_frozen": len(protocol["models"]) == 5, "fresh_programs": len(report["per_program_fresh"]) == 9, "operators_finite": all(np.isfinite(np.load(path)).all() for path in (OUT / "analysis/operators").glob("*.npy")), "producer_hash": core.sha(Path(__file__).with_name("phase1744_c210_natural_edit_trajectory.py")) == protocol["producer_sha256"]}
    audit = {"phase": 1744, "campaign": "C210", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
