#!/usr/bin/env python3
"""Independent audit for C135."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1669_c135_all_token_coordinate_transmission"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core


def contract() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    anchors = core.rows(OUT / "material/anchors.jsonl")
    checks = {"internal": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "anchors": len(anchors) == 12, "split": sum(row["partition"] == "discovery" for row in anchors) == 6, "truth": sum(row["truth_factor"] == 1 for row in anchors) == 6, "hashes": all(core.sha(Path(path)) == protocol["source_hashes"][name] for name, path in protocol["source_paths"].items()), "scope": "not a complete" in protocol["claim_boundary"]}
    report = {"phase": 1669, "campaign": "C135", "stage": "contract", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": "capture_c135_all_token_field" if all(checks.values()) else "stop"}
    core.save(OUT / "audit/independent_contract_audit.json", report)
    print(json.dumps(report, indent=2))


def final() -> None:
    capture = core.load(OUT / "analysis/capture.json")
    field = np.load(OUT / "raw/qwen3_all_token_all_checkpoint.bf16.npy", mmap_mode="r")
    freeze = core.load(OUT / "protocol/frozen_transmission.json")
    confirmation = core.load(OUT / "analysis/confirmation.json")
    checks = {"contract": core.load(OUT / "audit/independent_contract_audit.json")["all_checks_passed"], "capture": core.load(OUT / "audit/internal_capture_audit.json")["all_checks_passed"], "raw_shape": list(field.shape) == capture["shape"] and field.shape[0] == 38 and field.shape[2] == 2560, "raw_hash": core.sha(OUT / "raw/qwen3_all_token_all_checkpoint.bf16.npy") == capture["sha256"], "freeze": freeze["confirmation_unread"] and freeze["gain_sha256"] == core.sha(OUT / "protocol/frozen_diagonal_gain.float32.npy"), "confirmation": core.load(OUT / "audit/internal_confirmation_audit.json")["all_checks_passed"], "edges": len(core.rows(OUT / "analysis/confirmation_top_coordinate_edges.jsonl")) == 4096, "closure": core.load(OUT / "audit/internal_closure_audit.json")["all_checks_passed"]}
    report = {"phase": 1669, "campaign": "C135", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_gate_passed": confirmation["prediction_gate_passed"], "authorization": "start_route_C_C136" if all(checks.values()) else "stop"}
    core.save(OUT / "audit/independent_closure_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    {"contract": contract, "final": final}[sys.argv[1]]()
