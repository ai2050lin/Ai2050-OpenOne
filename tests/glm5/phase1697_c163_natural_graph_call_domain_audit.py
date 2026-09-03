#!/usr/bin/env python3
"""Independent curve-selection and natural-control audit for C163."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1697_c163_natural_graph_call_domain"


def load(path):
    return json.loads((OUT / path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in (OUT / path).read_text(encoding="utf-8").splitlines()]


def main():
    protocol = load("protocol/preregistration.json")
    lock = load("protocol/nonce_checkpoint_selection_lock.json")
    report = load("analysis/call_domain.json")
    scores = np.load(OUT / "raw/control_logits.float32.npy", mmap_mode="r")
    best = max(lock["curve"], key=lambda row: (row["nonce_mean_gain"], -row["q"]))["q"]
    checks = {
        "contract": load("audit/internal_contract_audit.json")["all_checks_passed"],
        "curve": load("audit/internal_curve_run_audit.json")["all_checks_passed"],
        "controls": load("audit/internal_control_run_audit.json")["all_checks_passed"],
        "analysis": load("audit/internal_analysis_audit.json")["all_checks_passed"],
        "selection": best == lock["selected_checkpoint"] == report["selected_checkpoint"] == 32,
        "control_shape": list(scores.shape) == [8, 256, 2],
        "typed_failure": report["natural_call_gate_passed"] is False and not all(report["gates"].values()),
        "wrong_relation_boundary": report["paired_win_rates"]["wrong_relation"] < protocol["natural_gates"]["paired_win_over_each_wrong_control_min"],
        "generation": len(rows("raw/natural_free_generation.jsonl")) == 64,
        "incident_predata": load("audit/pre_data_execution_incident.json")["scientific_data_revealed"] is False,
        "scope": "not training-time formation" in report["claim_boundary"],
    }
    audit = {"phase": 1697, "campaign": "C163", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_natural_call_passed": report["natural_call_gate_passed"], "authorization": "memo_then_C164"}
    (OUT / "audit/independent_final_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
