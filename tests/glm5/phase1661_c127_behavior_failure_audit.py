#!/usr/bin/env python3
"""Independent audit of the C127 behavior-only terminal branch."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1661_c127_typed_transition_language_family"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    behavior = core.load(OUT / "analysis/behavior_gate.json")
    closure = core.load(OUT / "analysis/closure.json")
    rows = core.rows(OUT / "raw/qwen3_behavior_index.jsonl")
    logits = np.load(OUT / "raw/qwen3_behavior_candidate_logits.float32.npy", mmap_mode="r")
    by = lambda key, value: float(np.mean([row["correct"] for row in rows if row[key] == value]))
    summary = {"global_accuracy": float(np.mean([row["correct"] for row in rows])), "by_partition": {value: by("partition", value) for value in ("discovery", "confirmation")}, "by_truth": {str(value): by("truth_factor", value) for value in (1, -1)}, "by_surface": {str(value): by("surface_factor", value) for value in (1, -1)}}
    checks = {
        "contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "behavior_integrity": core.load(OUT / "audit/internal_behavior_audit.json")["all_integrity_checks_passed"],
        "rows": len(rows) == 256 and list(logits.shape) == [256, 2] and bool(np.isfinite(logits).all()),
        "summary": summary == behavior["summary"],
        "failed_gate": behavior["gate_passed"] is False and closure["status"] == "closed_at_behavior_gate_without_hiddenstate_capture",
        "no_hiddenstate": not (OUT / "raw/qwen3_uniform_role_checkpoints.bf16.npy").exists(),
        "boundary": "no embedding" in closure["claim_boundary"] and "no embedding" not in protocol["claim_boundary"],
    }
    report = {"phase": 1661, "campaign": "C127", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": "append_memo_and_start_c128" if all(checks.values()) else "stop"}
    core.save(OUT / "audit/independent_behavior_failure_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
