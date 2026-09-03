#!/usr/bin/env python3
"""Independent audit for C140."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1674_c140_identifiability_and_master_contract"
RAW = TESTS / "result/phase1669_c135_all_token_coordinate_transmission/raw/qwen3_all_token_all_checkpoint.bf16.npy"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    internal = core.load(OUT / "audit/internal_contract_audit.json")
    ledger = core.load(OUT / "analysis/c133_c139_typed_ledger.json")
    arithmetic = core.load(OUT / "analysis/c135_archive_audit.json")
    raw = np.load(RAW, mmap_mode="r")
    checks = {
        "internal": internal["all_checks_passed"],
        "arithmetic": int(raw.size) == arithmetic["element_count"] == 113233920,
        "raw_hash": core.sha(RAW) == arithmetic["sha256"],
        "typed_missingness": ledger["A"]["discovery_prediction"] == "not-tested" and ledger["B"]["new_trajectory_prediction"] == "measured-fail",
        "five_arms": protocol["C141_design"]["total_cases"] == 1280 and len(protocol["C141_design"]["arms"]) == 5,
        "all_tokens_coordinates": protocol["C141_design"]["tokens"].startswith("all actual") and protocol["C141_design"]["coordinates"].startswith("all physical"),
        "observation_after_failure": "never blocks" in protocol["C141_design"]["behavior_policy"],
        "causal_last": protocol["stages"]["C148"].startswith("local symmetric"),
        "scope": all(term in protocol["forbidden"] for term in ("attention inspection", "MLP inspection", "weight inspection")),
    }
    report = {
        "phase": 1674,
        "campaign": "C140",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "authorization": "run_C141_contract_and_capture",
    }
    core.save(OUT / "audit/independent_contract_audit.json", report)
    print(json.dumps(report, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
