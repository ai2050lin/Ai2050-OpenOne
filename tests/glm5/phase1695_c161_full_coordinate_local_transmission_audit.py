#!/usr/bin/env python3
"""Independent data and discovery-lock audit for C161."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1695_c161_full_coordinate_local_transmission"


def load(path):
    return json.loads((OUT / path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in (OUT / path).read_text(encoding="utf-8").splitlines()]


def main():
    response = np.load(OUT / "raw/q24_relation_to_q25_six_role_response.float16.npy", mmap_mode="r")
    interactions = np.load(OUT / "raw/confirmation_second_order_interactions.float16.npy", mmap_mode="r")
    eps = np.load(OUT / "raw/anchor_epsilons.float32.npy")
    anchors = rows("material/anchors.jsonl")
    lock = load("protocol/discovery_pair_selection_lock.json")
    report = load("analysis/transmission.json")
    discovery = [row["anchor_index"] for row in anchors if row["partition"] == "discovery"]
    outgoing = np.mean([np.linalg.norm(np.asarray(response[i], np.float32).reshape(2560, -1), axis=1) for i in discovery], axis=0)
    top = np.argsort(outgoing)[-16:][::-1].tolist()
    checks = {
        "contract": load("audit/internal_contract_audit.json")["all_checks_passed"],
        "first_run": load("audit/internal_first_order_run_audit.json")["all_checks_passed"],
        "second_run": load("audit/internal_second_order_run_audit.json")["all_checks_passed"],
        "analysis": load("audit/internal_analysis_audit.json")["all_checks_passed"],
        "response_shape": list(response.shape) == [16, 2560, 6, 2560],
        "interaction_shape": list(interactions.shape) == [8, 8, 6, 2560],
        "epsilons": list(eps.shape) == [16] and bool(np.all(eps > 0)),
        "discovery_only_lock": lock["selection_source"] == "discovery anchors only" and top == lock["top_coordinates"],
        "first_order_gate": report["first_order_replication_passed"],
        "generic_diagnostic": report["generic_transport_diagnostic"]["relation_specific_transport_supported"] is False,
        "identity_reported": report["generic_transport_diagnostic"]["median_same_coordinate_energy_fraction"] > 0.5,
        "scope": "not a unique circuit" in report["claim_boundary"],
    }
    audit = {"phase": 1695, "campaign": "C161", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_first_order_replication_passed": report["first_order_replication_passed"], "scientific_relation_specific_transport_supported": report["generic_transport_diagnostic"]["relation_specific_transport_supported"], "authorization": "memo_then_C162"}
    (OUT / "audit/independent_final_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
