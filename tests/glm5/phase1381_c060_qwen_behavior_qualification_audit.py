#!/usr/bin/env python3
"""Independent audit for Phase1381."""
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

CONTRACT = TESTS / "result/phase1380_c060_conditional_coalition_campaign_contract"
OUT = TESTS / "result/phase1381_c060_qwen_behavior_qualification"
SCRIPT = TESTS / "phase1381_c060_qwen_behavior_qualification.py"


def main() -> None:
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    summary = core.load(OUT / "analysis/qwen3_behavior_summary.json")
    final = core.load(OUT / "analysis/final.json")
    active = core.rows(OUT / "raw/active_behavior.jsonl")
    status = core.rows(OUT / "raw/status_behavior.jsonl")
    selected = core.rows(OUT / "material/eligible_pairs.jsonl")
    py_compile.compile(str(SCRIPT), doraise=True)
    py_compile.compile(__file__, doraise=True)
    qualified = bool(summary["behavior_qualified"])
    split_target = int(protocol["material"]["discovery_target"])
    per_cell = int(protocol["material"]["eligible_cases_per_cell"])
    expected_selected = int(protocol["material"]["eligible_case_target"])
    expected_auth = (
        "run_phase1382_c060_instrument_calibration"
        if qualified
        else "close_c060_behavior_unqualified_before_hidden_access"
    )
    checks = {
        "counts": len(active) == 864 and len(status) == 288,
        "finite": all(math.isfinite(r["margin"]) for r in active + status),
        "active_accuracy": abs(
            summary["active"]["accuracy"] - sum(r["correct"] for r in active) / len(active)
        ) < 1e-12,
        "status_accuracy": abs(
            summary["status"]["accuracy"] - sum(r["correct"] for r in status) / len(status)
        ) < 1e-12,
        "selected_matches_gate": (
            len(selected) == expected_selected
            and len({r["pair_id"] for r in selected}) == expected_selected
        ) if qualified else len(selected) == 0,
        "partition_matches_gate": (
            Counter(r["partition"] for r in selected)
            == {p: split_target for p in protocol["material"]["partitions"]}
        ) if qualified else True,
        "cell_matches_gate": (
            set(Counter((r["target_family"], r["partition"], r["surface"]) for r in selected).values())
            == {per_cell}
        ) if qualified else True,
        "checks_consistent": qualified == all(summary["checks"].values()),
        "final_consistent": final["behavior_qualified"] == qualified,
        "authorization": final["authorization"] == expected_auth,
        "scripts_compile": True,
    }
    audit = {
        "phase": 1381,
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
