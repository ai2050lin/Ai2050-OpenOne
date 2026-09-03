#!/usr/bin/env python3
"""Independent C128 behavior qualification audit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1662_c128_direct_precedence_behavior_qualification"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    behavior = core.load(OUT / "analysis/behavior_gate.json")
    closure = core.load(OUT / "analysis/closure.json")
    rows = core.rows(OUT / "raw/qwen3_behavior_index.jsonl")
    logits = np.load(OUT / "raw/qwen3_candidate_logits.float32.npy", mmap_mode="r")
    accuracy = lambda selected: float(np.mean([row["correct"] for row in selected]))
    summary = {"global_accuracy": accuracy(rows), "by_partition": {key: accuracy([row for row in rows if row["partition"] == key]) for key in ("discovery", "confirmation")}, "by_truth": {str(key): accuracy([row for row in rows if row["truth_factor"] == key]) for key in (1, -1)}, "by_surface": {str(key): accuracy([row for row in rows if row["surface_factor"] == key]) for key in (1, -1)}}
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "behavior": core.load(OUT / "audit/internal_behavior_audit.json")["all_integrity_checks_passed"], "closure": core.load(OUT / "audit/internal_closure_audit.json")["all_checks_passed"], "parent_hashes": all(core.sha(Path(protocol["parent_paths"][name])) == digest for name, digest in protocol["parent_hashes"].items()), "rows": len(rows) == 256 and list(logits.shape) == [256, 2] and bool(np.isfinite(logits).all()), "summary": summary == behavior["summary"], "no_hiddenstate": not (OUT / "raw/qwen3_uniform_role_checkpoints.bf16.npy").exists(), "boundary": "behavior only" in closure["claim_boundary"]}
    report = {"phase": 1662, "campaign": "C128", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_gate_passed": behavior["gate_passed"], "authorization": "start_c129" if all(checks.values()) and behavior["gate_passed"] else "stop"}
    core.save(OUT / "audit/independent_closure_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
